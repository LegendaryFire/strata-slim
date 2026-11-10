"""Hugging Face: list a repository's files at a commit, and download one file with its size and SHA-256 checked.

Uses the public API only (no huggingface_hub). HF_TOKEN for gated or private repositories, HF_ENDPOINT for a mirror.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

from common import Progress, SetupError, size_text

log = logging.getLogger(__name__)


@dataclass
class RemoteFile:
    path: str              # path in the repository, e.g. UD-Q4_K_XL/x-00001-of-00004.gguf
    size: int
    sha256: str | None     # None for small files that Hugging Face does not keep in LFS

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


# ---- the .done mark ---------------------------------------------------------------------------------------------
# A finished, checked file has <file>.done beside it holding "sha256 <hex>" (the mark upstream's setup writes too).

def done_mark(path: Path) -> Path:
    return path.with_name(f"{path.name}.done")


def marked_sha(path: Path) -> str | None:
    """The SHA-256 in a file's .done mark; None when there is no mark or it holds no hash."""
    mark = done_mark(path)
    if not mark.is_file():
        return None
    m = re.search(r"\b([0-9a-f]{64})\b", mark.read_text(errors="replace"))
    return m.group(1) if m else None


def is_done(path: Path, want: RemoteFile) -> bool:
    """True when the file is there, marked done and is the one wanted. A marked file that is another one stops."""
    if not path.exists() or not done_mark(path).exists():
        return False
    if path.stat().st_size != want.size:
        raise SetupError(f"{path}: {path.stat().st_size} bytes, but the recorded version has {want.size}")
    have = marked_sha(path)
    if have and want.sha256 and have != want.sha256:
        raise SetupError(f"{path}: its .done mark says sha256 {have[:16]}.., the recorded version {want.sha256[:16]}..")
    return True


def sha256_of(path: Path, label: str) -> str:
    """The file's SHA-256, with a progress line while it reads (a 50 GB shard takes minutes)."""
    digest = hashlib.sha256()
    progress = Progress(f"{label}: checking SHA-256", path.stat().st_size)
    with path.open("rb") as f:
        chunk_size = 8 << 20
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
            progress.update(f.tell())
    progress.finish("checked")
    return digest.hexdigest()


# ---- the client -------------------------------------------------------------------------------------------------

class HuggingFace:
    def __init__(self, endpoint: str, token: str | None):
        self.endpoint = endpoint.rstrip("/")
        self.token = token

    def request(self, url: str, headers: dict | None = None) -> urllib.request.Request:
        all_headers = {"User-Agent": "strata-slim"}
        if headers:
            all_headers.update(headers)
        if self.token:
            all_headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(url, headers=all_headers)

    def api(self, path: str):
        """GET {endpoint}/api/{path} as JSON. A list answer is followed through its "next" pages and joined."""
        url = f"{self.endpoint}/api/{path}"
        pages = []
        while url:
            try:
                with urllib.request.urlopen(self.request(url), timeout=60) as r:
                    data = json.load(r)
                    link = r.headers.get("Link", "")
            except urllib.error.HTTPError as e:
                if e.code == HTTPStatus.FORBIDDEN or e.code == HTTPStatus.UNAUTHORIZED:
                    hint = "Ensure HF_TOKEN is set, and accept the terms and conditions"
                elif e.code == HTTPStatus.NOT_FOUND:
                    hint = "No such repository, revision or path"
                else:
                    hint = "An unknown error occurred"
                raise SetupError(f"{url}: HTTP {e.code} - {hint}")
            except urllib.error.URLError as e:
                raise SetupError(f"{url}: {e.reason} (no network? A start with every file present does not need it)")

            if not isinstance(data, list):
                return data
            pages += data

            # Link: <https://...?cursor=...>; rel="next"
            url = ""
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip(" <>")
        return pages

    def resolve_commit(self, repo: str, revision: str) -> str:
        """A branch, tag or commit -> the commit it is now."""
        answer = self.api(f"models/{repo}/revision/{urllib.parse.quote(revision, safe='')}")
        return answer.get("sha") or revision

    def list_files(self, repo: str, commit: str) -> list[RemoteFile]:
        files = []
        for f in self.api(f"models/{repo}/tree/{urllib.parse.quote(commit, safe='')}?recursive=true"):
            if f.get("type") != "file":
                continue
            lfs = f.get("lfs") or {}
            files.append(RemoteFile(f["path"], int(lfs.get("size") or f["size"]), lfs.get("oid")))
        return files

    def download(self, repo: str, commit: str, f: RemoteFile, dst: Path) -> None:
        """One file: resumable (<dst>.part with a Range request), its size and SHA-256 checked, then marked done."""
        if is_done(dst, f):
            return
        if dst.exists():
            raise SetupError(f"{dst}: there without a .done mark, so it was not checked. Move it away to download it.")

        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(f"{dst.name}.part")
        free = shutil.disk_usage(dst.parent).free
        needed = f.size - (part.stat().st_size if part.exists() else 0)
        if free < needed:
            raise SetupError(f"{dst.parent}: {size_text(free)} free, {dst.name} needs {size_text(needed)} more")

        url = f"{self.endpoint}/{repo}/resolve/{urllib.parse.quote(commit, safe='')}/{urllib.parse.quote(f.path)}"
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                self.fetch_rest(url, part, f, dst.name)
                break
            except OSError as e:          # URLError, TimeoutError and ConnectionError are all OSErrors
                if attempt == max_attempts:
                    raise SetupError(f"{dst.name}: the download keeps stopping ({e}). "
                                     "Start again: it continues where it stopped.")
                log.warning(f"{dst.name}: stopped ({e}); again in 10 s ({attempt + 1} of {max_attempts})")
                time.sleep(10)

        if part.stat().st_size != f.size:
            raise SetupError(f"{dst.name}: got {part.stat().st_size} bytes, expected {f.size}")
        got = sha256_of(part, dst.name)
        if f.sha256 and got != f.sha256:
            part.unlink()
            raise SetupError(f"{dst.name}: SHA-256 mismatch ({got[:16]}.. against {f.sha256[:16]}..); "
                             "removed, start again to retry")
        part.replace(dst)
        done_mark(dst).write_text(f"sha256 {got}")
        log.info(f"{dst.name}: done")

    def fetch_rest(self, url: str, part: Path, f: RemoteFile, label: str) -> None:
        """Append what is missing of the file to <part>, with a progress line."""
        have = part.stat().st_size if part.exists() else 0
        if have >= f.size:
            return

        headers = {"Range": f"bytes={have}-"} if have else None
        with urllib.request.urlopen(self.request(url, headers), timeout=120) as r:
            if have and r.status != HTTPStatus.PARTIAL_CONTENT:  # the server ignored the range: start over rather than corrupt it
                part.unlink()
                have = 0
            if have:
                log.info(f"{label}: resuming at {size_text(have)} of {size_text(f.size)}")

            progress = Progress(label, f.size, done=have)
            written = have
            with part.open("ab") as out:
                chunk_size = 8 << 20
                while True:
                    block = r.read(chunk_size)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
                    progress.update(written)
            progress.finish("downloaded")
