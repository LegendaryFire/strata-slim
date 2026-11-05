"""Hugging Face: list a repository's files at a commit, and download one file with its size and SHA-256 checked.

Uses the public API only (no huggingface_hub). HF_TOKEN for gated or private repositories, HF_ENDPOINT for a mirror.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from common import Progress, env, fail, say, size_text

CHUNK = 8 << 20
DOWNLOAD_TRIES = 5


@dataclass
class RemoteFile:
    path: str              # path in the repository, e.g. UD-Q4_K_XL/x-00001-of-00004.gguf
    size: int
    sha256: str | None     # None for small files that Hugging Face does not keep in LFS

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


# ---- the API ----------------------------------------------------------------------------------------------------

def endpoint() -> str:
    return (env("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def request(url: str, headers: dict | None = None) -> urllib.request.Request:
    all_headers = {"User-Agent": "strata-slim", **(headers or {})}
    if (token := env("HF_TOKEN")):
        all_headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=all_headers)


def api(path: str):
    """GET {endpoint}/api/{path} as JSON. A list answer is followed through its "next" pages and joined."""
    url, pages = f"{endpoint()}/api/{path}", []
    while url:
        try:
            with urllib.request.urlopen(request(url), timeout=60) as r:
                data, link = json.load(r), r.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            hint = {401: " (private or gated: set HF_TOKEN)", 403: " (gated: accept its terms, set HF_TOKEN)",
                    404: " (no such repository, revision or path)"}.get(e.code, "")
            fail(f"{url}: HTTP {e.code}{hint}")
        except urllib.error.URLError as e:
            fail(f"{url}: {e.reason} (no network? A start with every file present does not need it)")
        if not isinstance(data, list):
            return data
        pages += data
        next_links = [part.split(";")[0].strip(" <>") for part in link.split(",") if 'rel="next"' in part]
        url = next_links[0] if next_links else ""
    return pages


def _quote(s: str) -> str:
    return urllib.parse.quote(s, safe="")


def resolve_commit(repo: str, revision: str) -> str:
    """A branch, tag or commit -> the commit it is now."""
    return api(f"models/{repo}/revision/{_quote(revision)}").get("sha") or revision


def list_files(repo: str, commit: str) -> list[RemoteFile]:
    files = []
    for f in api(f"models/{repo}/tree/{_quote(commit)}?recursive=true"):
        if f.get("type") == "file":
            lfs = f.get("lfs") or {}
            files.append(RemoteFile(f["path"], int(lfs.get("size") or f["size"]), lfs.get("oid")))
    return files


def recent_commits(repo: str, limit: int) -> list[str]:
    """The newest commits on main, newest first."""
    return [c["id"] for c in api(f"models/{repo}/commits/main")[:limit]]


# ---- the .done mark ---------------------------------------------------------------------------------------------
# A finished, checked file has <file>.done beside it holding "sha256 <hex>" (the mark upstream's setup writes too).

def done_mark(path: Path) -> Path:
    return path.with_name(path.name + ".done")


def marked_sha(path: Path) -> str | None:
    """The SHA-256 in a file's .done mark; None when there is no mark or it holds no hash."""
    mark = done_mark(path)
    if not mark.is_file():
        return None
    m = re.search(r"\b([0-9a-f]{64})\b", mark.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


def is_done(path: Path, want: RemoteFile) -> bool:
    """True when the file is there, marked done and is the one wanted. A marked file that is another one stops."""
    if not (path.exists() and done_mark(path).exists()):
        return False
    if path.stat().st_size != want.size:
        fail(f"{path}: {path.stat().st_size} bytes, but the recorded version has {want.size}")
    have = marked_sha(path)
    if have and want.sha256 and have != want.sha256:
        fail(f"{path}: its .done mark says sha256 {have[:16]}.., the recorded version {want.sha256[:16]}..")
    return True


def sha256_of(path: Path, label: str | None = None) -> str:
    """The file's SHA-256; with a label, a progress line while it reads (a 50 GB shard takes minutes)."""
    digest = hashlib.sha256()
    progress = Progress(f"{label}: checking SHA-256", path.stat().st_size) if label else None
    with path.open("rb") as f:
        while (block := f.read(CHUNK)):
            digest.update(block)
            if progress:
                progress.update(f.tell())
    if progress:
        progress.finish("checked")
    return digest.hexdigest()


# ---- downloading ------------------------------------------------------------------------------------------------

def download(repo: str, commit: str, f: RemoteFile, dst: Path) -> None:
    """One file: resumable (<dst>.part with a Range request), its size and SHA-256 checked, then marked done."""
    if is_done(dst, f):
        return
    if dst.exists():
        adopt_unmarked(repo, commit, f, dst)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    free = shutil.disk_usage(dst.parent).free
    needed = f.size - (part.stat().st_size if part.exists() else 0)
    if free < needed:
        fail(f"{dst.parent}: {size_text(free)} free, {dst.name} needs {size_text(needed)} more")

    url = f"{endpoint()}/{repo}/resolve/{_quote(commit)}/{urllib.parse.quote(f.path)}"
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        try:
            _fetch_rest(url, part, f, dst.name)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == DOWNLOAD_TRIES:
                fail(f"{dst.name}: the download keeps stopping ({e}). Start again: it continues where it stopped.")
            say(f"{dst.name}: stopped ({e}); again in 10 s ({attempt + 1} of {DOWNLOAD_TRIES})")
            time.sleep(10)

    if part.stat().st_size != f.size:
        fail(f"{dst.name}: got {part.stat().st_size} bytes, expected {f.size}")
    got = sha256_of(part, dst.name)
    if f.sha256 and got != f.sha256:
        part.unlink()
        fail(f"{dst.name}: SHA-256 mismatch ({got[:16]}.. against {f.sha256[:16]}..); removed, start again to retry")
    part.replace(dst)
    done_mark(dst).write_text(f"sha256 {got}", encoding="utf-8")
    say(f"{dst.name}: done")


def _fetch_rest(url: str, part: Path, f: RemoteFile, label: str) -> None:
    """Append what is missing of the file to <part>, with a progress line."""
    have = part.stat().st_size if part.exists() else 0
    if have >= f.size:
        return
    with urllib.request.urlopen(request(url, {"Range": f"bytes={have}-"} if have else None), timeout=120) as r:
        if have and r.status != 206:           # the server ignored the range: start over rather than corrupt it
            part.unlink()
            have = 0
        if have:
            say(f"{label}: resuming at {size_text(have)} of {size_text(f.size)}")
        progress = Progress(label, f.size, done=have)
        written = have
        with part.open("ab") as out:
            while (block := r.read(CHUNK)):
                out.write(block)
                written += len(block)
                progress.update(written)
        progress.finish("downloaded")


def adopt_unmarked(repo: str, commit: str, f: RemoteFile, dst: Path) -> None:
    """A file already there without a .done mark (copied by hand): keep it if it is the right one."""
    say(f"{dst.name}: present without a .done mark")
    got = sha256_of(dst, dst.name)
    if f.sha256 and got != f.sha256:
        fail(f"{dst}: a different file than {repo}@{commit[:12]} has under this name. Move it away to download it.")
    done_mark(dst).write_text(f"sha256 {got}", encoding="utf-8")
