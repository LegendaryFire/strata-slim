#!/usr/bin/env python3
"""Write the engine's BUILD.json in a build stage (both Dockerfiles run it): what was built, from which upstream
release, for which GPUs, with which patches.

    build_info.py cuda "$CUDA_ARCHITECTURES" /src /out/engine
    build_info.py hip  "$HIP_ARCHITECTURES"  /src /out/engine

The source root holds UPSTREAM (from extract.py) and PATCHES (from apply_patches.py).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def read_upstream(root: Path) -> dict[str, str]:
    """vendor/UPSTREAM: key=value lines."""
    pairs = (line.split("=", 1) for line in Path(root, "UPSTREAM").read_text().splitlines() if "=" in line)
    return {k.strip(): v.strip() for k, v in pairs}


def cuda_version() -> str:
    found = re.search(r"release (\d+\.\d+)", subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout)
    return found.group(1) if found else "unknown"


def rocm_version() -> str:
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    try:
        return Path(rocm, ".info", "version").read_text().strip()
    except OSError:
        done = subprocess.run([str(Path(rocm, "bin", "hipconfig")), "--version"], capture_output=True, text=True)
        return done.stdout.strip() or "unknown"


def build_info(backend: str, archs_arg: str, root: Path, engine: Path) -> dict:
    upstream = read_upstream(root)
    archs = [a for a in re.split(r"[;,\s]+", archs_arg) if a]
    patches = [p for p in Path(root, "PATCHES").read_text().strip().split(",") if p]
    info = {
        "version": upstream["engine_version"],
        "source": f"strata-slim {upstream['tag']}",
        "commit": upstream["commit"],
    }
    if backend == "cuda":
        info |= {"cuda": cuda_version(), "portable": True, "archs": archs, "vision_archs": archs, "ptx": True}
    else:
        info |= {"backend": "hip", "rocm": rocm_version(), "portable": True, "archs": archs,
                 "vision": "gpu" if Path(engine, "strata-vision-hip").is_file() else "none",
                 "vision_cpu": "strata-vision", "vision_gpu": "strata-vision-hip", "vision_archs": archs}
    info["patches"] = patches
    return info


if __name__ == "__main__":
    logging.basicConfig(format="[build-info] %(levelname)s %(message)s", level="INFO")
    if len(sys.argv) != 5 or sys.argv[1] not in ("cuda", "hip"):
        sys.exit("usage: build_info.py cuda|hip <architectures> <source root> <engine folder>")
    engine = Path(sys.argv[4])
    info = build_info(sys.argv[1], sys.argv[2], Path(sys.argv[3]), engine)
    build_json = Path(engine, "BUILD.json")
    with build_json.open("w") as f:
        json.dump(info, f)
    log.info(json.dumps(info))
