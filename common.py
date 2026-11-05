"""Helpers shared by launch.py and prepare.py: reading variables, stopping with a message, finding model files."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

ROOT = Path("/opt/strata")          # where the image keeps the engine, the server and the tools


def env(name: str, default: str | None = None) -> str | None:
    """A variable's value without surrounding spaces; unset and empty both give the default."""
    value = os.environ.get(name, "").strip()
    return value or default


def say(msg: str) -> None:
    print(f"[strata-slim] {msg}", flush=True)


def fail(msg: str) -> NoReturn:
    print(f"[strata-slim] {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


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
            say(f"{self.label}  {percent:3.0f}%  {size_text(done)} / {size_text(self.total)}  {speed / 1e6:.0f} MB/s{left}")

    def finish(self, what: str = "done") -> None:
        elapsed = time.monotonic() - self.started
        moved = self.done - self.first
        rate = f", {moved / elapsed / 1e6:.0f} MB/s" if elapsed > 1 and moved > 0 else ""
        say(f"{self.label}  {what}: {size_text(self.done)} in {duration_text(elapsed)}{rate}")


DRAFTERS = ("mtp", "dflash2", "auto", "off")


def drafter() -> str:
    """DRAFTER: which draft model guesses ahead. mtp (default): the model's MTP layer. dflash2: the DFlash2 block drafter
    (Qwen3.8-Flash-Next only, one request at a time). auto: upstream's rule (DFlash2 where it runs, else MTP). off: none."""
    choice = (env("DRAFTER") or "mtp").lower()
    if choice not in DRAFTERS:
        fail(f"DRAFTER={choice}: one of {', '.join(DRAFTERS)}")
    return choice


@dataclass
class ModelPaths:
    model: Path            # the model's GGUF file, or the folder that holds its shards
    pack: Path
    mtp: Path | None       # the MTP draft layer's runtime folder; None when it is not used

    @property
    def data_dir(self) -> Path:
        """The models root as the server finds it: <root>/packs/<tag> -> <root>. The DFlash2 drafter is in <root>/dflash2/."""
        return self.pack.parent.parent


def model_paths() -> ModelPaths:
    """Where the model files go, derived from MODEL_REPO + MODEL_QUANT under MODELS_DIR (default /models):

        <MODELS_DIR>/models/<owner>-<quant>/     the GGUF files (owner: the repository's owner, e.g. unsloth)
        <MODELS_DIR>/packs/<owner>-<quant>/      the pack (lowercase)
        <MODELS_DIR>/mtp/rt/                     the MTP draft layer, one for every Qwen3.8-Flash-Next model
        <MODELS_DIR>/dflash2/                    the DFlash2 drafter (with DRAFTER=dflash2 or auto)

    which is upstream's layout and folder naming. MODEL_GGUF and PACK override the first two.
    The MTP layer is used only with DRAFTER=mtp or auto."""
    root = Path(env("MODELS_DIR") or "/models")

    if env("MODEL_GGUF"):
        model = Path(env("MODEL_GGUF"))
    elif env("MODEL_REPO"):
        owner, _, name = env("MODEL_REPO").partition("/")
        model = root / "models" / f"{owner}-{env('MODEL_QUANT') or name}"
    else:
        fail("set MODEL_REPO (and MODEL_QUANT) to download a model, or MODEL_GGUF to a model already there")

    folder = model if model.suffix != ".gguf" else model.parent
    pack = Path(env("PACK")) if env("PACK") else root / "packs" / folder.name.lower()

    mtp = root / "mtp" / "rt" if drafter() in ("mtp", "auto") else None
    return ModelPaths(model, pack, mtp)


def first_shard(model: Path) -> Path:
    """The GGUF the engine is given: the file itself, or in a folder its *-00001-of-*.gguf (or its only .gguf)."""
    if model.is_file():
        return model
    if not model.is_dir():
        fail(f"MODEL_GGUF={model}: no such file or folder inside the container (check the volume mounts)")
    shards = sorted(model.glob("*-00001-of-*.gguf"))
    if shards:
        return shards[0]
    singles = [g for g in sorted(model.glob("*.gguf")) if not g.name.lower().startswith("mmproj")]
    if len(singles) == 1:
        return singles[0]
    fail(f"MODEL_GGUF={model}: {len(singles)} .gguf files and no *-00001-of-*.gguf among them; name the file")


def mmproj_near(model_folder: Path) -> Path | None:
    """The mmproj*.gguf in the model folder or beside it, when there is exactly one."""
    found = sorted(model_folder.glob("mmproj*.gguf")) + sorted(model_folder.parent.glob("mmproj*.gguf"))
    return found[0] if len(found) == 1 else None


def vision_on() -> bool:
    mode = (env("VISION") or "off").lower()
    if mode in ("cpu", "gpu", "on"):
        return True
    if mode in ("off", "no", "false", "0"):
        return False
    fail(f"VISION={mode}: off, cpu or gpu")
