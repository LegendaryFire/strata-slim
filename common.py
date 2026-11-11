"""Shared by launch.py, prepare.py, hf.py and settings.py: the setup error, progress lines, finding model files."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

ROOT = Path("/opt/strata")          # where the image keeps the engine, the server and the tools
ON_WINDOWS = os.name == "nt"

log = logging.getLogger(__name__)


class SetupError(Exception):
    """A problem with the setup (a variable, a file, the network) that stops the start.
    launch.py logs it and exits 2."""


# ---- progress lines ---------------------------------------------------------------------------------------------

def size_text(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def duration_text(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class Progress:
    """A progress line for a long download or check, printed every 5 seconds: how much is done, the speed, the time left."""
    INTERVAL = 5.0

    def __init__(self, label: str, total: int, done: int = 0):
        self.label, self.total, self.done, self.first = label, total, done, done
        self.started = self.last_print = time.monotonic()

    def update(self, done: int) -> None:
        self.done = done
        now = time.monotonic()
        if now - self.last_print >= self.INTERVAL:
            self.last_print = now
            speed = (done - self.first) / (now - self.started)
            left = f", {duration_text((self.total - done) / speed)} left" if speed > 0 else ""
            percent = 100 * done / self.total if self.total else 0
            log.info(f"{self.label}  {percent:3.0f}%  {size_text(done)} / {size_text(self.total)}  "
                     f"{speed / 1e6:.0f} MB/s{left}")

    def finish(self, what: str = "done") -> None:
        elapsed = time.monotonic() - self.started
        moved = self.done - self.first
        rate = f", {moved / elapsed / 1e6:.0f} MB/s" if elapsed > 1 and moved > 0 else ""
        log.info(f"{self.label}  {what}: {size_text(self.done)} in {duration_text(elapsed)}{rate}")


def engine_program(name: str) -> Path:
    """A program in engine/: strata.exe on Windows, strata in the image."""
    if ON_WINDOWS:
        return Path(ROOT, "engine", f"{name}.exe")
    else:
        return Path(ROOT, "engine", f"{name}")


# ---- model files ------------------------------------------------------------------------------------------------

def first_shard(model: Path) -> Path:
    """The GGUF the engine is given: the file itself, or in a folder its *-00001-of-*.gguf (or its only .gguf)."""
    if model.is_file():
        return model
    if not model.is_dir():
        raise SetupError(f"model {model} does not exist: set MODEL_GGUF to a .gguf file or a folder of shards, "
                         "or MODEL_REPO to download one")
    shards = sorted(model.glob("*-00001-of-*.gguf"))
    if shards:
        return shards[0]
    singles = [g for g in sorted(model.glob("*.gguf")) if not g.name.lower().startswith("mmproj")]
    if len(singles) == 1:
        return singles[0]
    raise SetupError(f"MODEL_GGUF={model}: {len(singles)} .gguf files and no *-00001-of-*.gguf among them; "
                     "name the file")


def mmproj_near(model_folder: Path) -> Path | None:
    """The mmproj*.gguf in the model folder or beside it, when there is exactly one."""
    found = sorted(model_folder.glob("mmproj*.gguf")) + sorted(model_folder.parent.glob("mmproj*.gguf"))
    return found[0] if len(found) == 1 else None
