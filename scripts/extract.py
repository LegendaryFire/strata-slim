#!/usr/bin/env python3
"""Take one upstream Strata release into vendor/: only the paths in manifest.txt.

    ./scripts/extract.py v0.2.11                                     # from the cached mirror in .upstream/ (cloned on first use)
    UPSTREAM_GIT=/srv/llm/strata/repo ./scripts/extract.py v0.2.11   # from a clone you already have (no download)

vendor/ is replaced as a whole, so `git diff vendor/` afterwards shows exactly what upstream changed in the parts this
image uses. patches/*.patch are checked against it but not applied: vendor/ stays upstream's. Nothing is built here;
docker build is next.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the project root: manifest.txt, vendor/, .upstream/
UPSTREAM_URL = os.environ.get("UPSTREAM_URL") or "https://github.com/Niko1221/Strata.git"

log = logging.getLogger(__name__)


class ExtractError(Exception):
    """A problem that stops the extract (a tag, the manifest, a patch). Logged and exits 1."""


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def upstream_repo() -> Path:
    """The repo to read from: UPSTREAM_GIT as it is, or .upstream/ (a bare clone, made once, fetched every time)."""
    if os.environ.get("UPSTREAM_GIT"):
        return Path(os.environ["UPSTREAM_GIT"])
    src = ROOT / ".upstream"
    if not src.is_dir():
        log.info(f"cloning {UPSTREAM_URL} into .upstream/ (once)")
        subprocess.run(["git", "clone", "--quiet", "--bare", UPSTREAM_URL, str(src)], check=True)
    log.info(f"fetching tags from {UPSTREAM_URL}")
    subprocess.run(["git", "-C", str(src), "fetch", "--quiet", "--tags", "--force", UPSTREAM_URL,
                    "+refs/heads/main:refs/heads/main"], check=True)
    return src


def manifest_specs() -> list[str]:
    """manifest.txt as git pathspecs: comments and blank lines dropped, "!x" becomes ":(exclude)x"."""
    specs = []
    for line in (ROOT / "manifest.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            specs.append(":(exclude)" + line[1:] if line.startswith("!") else line)
    return specs


def size_text(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def extract(tag: str) -> None:
    os.chdir(ROOT)
    src = upstream_repo()

    try:
        commit = git("-C", str(src), "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        raise ExtractError(f"{tag} is not a tag or commit in {src}") from None

    # every included entry must exist upstream: a renamed folder should fail here, not in docker build
    specs = manifest_specs()
    missing = [s for s in specs if not s.startswith(":(exclude)")
               and not git("-C", str(src), "ls-tree", "-r", "--name-only", commit, "--", s)]
    if missing:
        raise ExtractError(f"manifest entries not in {tag}: {', '.join(missing)} (upstream moved or removed these: "
                           "fix manifest.txt)")

    new = ROOT / "vendor.new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir()
    archive = subprocess.Popen(["git", "-C", str(src), "archive", "--format=tar", commit, "--", *specs],
                               stdout=subprocess.PIPE)
    with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
        tar.extractall(new, filter="tar")
    if archive.wait() != 0:
        raise ExtractError("git archive failed")

    # the config keys the server reads (cfg.get("key") / cfg["key"] in serve/): launch.py warns about a CONFIG_<KEY>
    # not among them
    keys = set()
    for py in (new / "serve").glob("*.py"):
        keys.update(re.findall(r'cfg(?:\.get\(|\[)"([a-z_][a-z0-9_]*)"', py.read_text(encoding="utf-8")))
    (new / ".config-keys").write_text("".join(f"{k}\n" for k in sorted(keys)))

    version = re.search(r"^\s*project\([^)]*VERSION\s*([0-9][0-9.]*)", (new / "CMakeLists.txt").read_text(), re.M)
    engine_ver = version.group(1) if version else ""
    llama = (new / "third_party/ggml/VERSION.txt").read_text().split()[0]
    date = git("-C", str(src), "show", "-s", "--format=%cI", commit)
    (new / "UPSTREAM").write_text(f"tag={tag}\ncommit={commit}\ndate={date}\nengine_version={engine_ver}\n"
                                  f"llama_cpp_commit={llama}\n")

    shutil.rmtree(ROOT / "vendor", ignore_errors=True)
    new.rename(ROOT / "vendor")

    # local fixes (patches/*.patch) stay out of vendor/: the build applies them only when asked (the PATCHES build
    # arg). Checked here so a release that breaks one shows now, not in docker build.
    for patch in sorted((ROOT / "patches").glob("*.patch")):
        name = patch.relative_to(ROOT)
        if subprocess.run(["git", "apply", "--check", str(patch)], capture_output=True).returncode == 0:
            log.info(f"patch applies: {name}")
        elif subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
            log.warning(f"patch already in {tag} (remove it): {name}")
        else:
            raise ExtractError(f"patch does not apply to {tag}: {name} (rebase it on the new vendor/)")

    files = [p for p in (ROOT / "vendor").rglob("*") if p.is_file()]
    log.info(f"vendor/ = {tag} ({commit}), engine {engine_ver}, {len(files)} files, "
        f"{size_text(sum(p.stat().st_size for p in files))}")
    log.info("next: git diff --stat vendor/, then docker build")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [extract] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level="INFO")
    if len(sys.argv) != 2:
        sys.exit("usage: extract.py <upstream tag, e.g. v0.2.11>")
    try:
        extract(sys.argv[1])
    except ExtractError as e:
        log.error(e)
        sys.exit(1)
