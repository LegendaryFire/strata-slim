#!/usr/bin/env python3
"""Apply local fixes from patches/ to the engine sources in a build stage (both Dockerfiles run it).

    apply_patches.py <patches dir> <source root> "<PATCHES>"

PATCHES: empty or "none" (the default): nothing, the build is upstream's own code.
         "all": every *.patch, in name order.
         a comma list: just those, by file name with or without .patch, or by number ("0001,0002").
The patch files name paths as vendor/...; the source root is where vendor/'s contents are (/src in the build stage).
Writes the applied names, comma-separated, to <source root>/PATCHES (empty for none) for BUILD.json.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


class PatchError(Exception):
    """PATCHES names a patch that isn't there, or a patch doesn't apply. Logged and fails the build."""


def pick(folder: Path, want: str) -> list[Path]:
    """The patch files PATCHES names, in the order given ("all": name order)."""
    want = "".join(want.split())
    if want in ("", "none"):
        return []
    if want == "all":
        return sorted(folder.glob("*.patch"))
    picked = []
    for name in filter(None, want.split(",")):
        stem = name.removesuffix(".patch")
        matches = [Path(folder, f"{stem}.patch")] if Path(folder, f"{stem}.patch").is_file() else \
            sorted(folder.glob(f"{stem}-*.patch"))                      # a number: 0001 -> 0001-*.patch
        if not matches:
            raise PatchError(f"PATCHES: no patch named {name} in patches/")
        if len(matches) > 1:
            raise PatchError(f"PATCHES: {name} matches more than one patch")
        picked.append(matches[0])
    return picked


def apply(folder: Path, root: Path, want: str) -> None:
    applied = []
    for patch in pick(folder, want):
        with patch.open("rb") as f:
            done = subprocess.run(["patch", "-d", str(root), "-p2", "--forward", "--batch", "--no-backup-if-mismatch"],
                                  stdin=f)
        if done.returncode != 0:
            raise PatchError(f"{patch.name} does not apply (see patch's output above)")
        applied.append(patch.stem)
        log.info(f"patch applied: {patch.name}")
    if not applied:
        log.info("no patches: upstream's code as released")
    Path(root, "PATCHES").write_text(f"{','.join(applied)}\n")


if __name__ == "__main__":
    logging.basicConfig(format="[patches] %(levelname)s %(message)s", level="INFO")
    if len(sys.argv) not in (3, 4):
        sys.exit('usage: apply_patches.py <patches dir> <source root> "<PATCHES>"')
    try:
        apply(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else "")
    except PatchError as e:
        log.error(e)
        sys.exit(1)
