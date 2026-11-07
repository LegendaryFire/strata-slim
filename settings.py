"""strata-slim's settings: every variable it reads, with its default and type, and where they put the model files.

Settings.load() reads them once at start (launch.py), from the environment and an optional env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import get_args, get_type_hints

from dotenv import load_dotenv

from common import ROOT, SetupError


# Settings that take one of a few values. The variable is matched in any case: VISION=GPU is Vision.GPU.

class Drafter(StrEnum):
    """Which draft model guesses ahead."""
    MTP = "mtp"            # the model's MTP layer
    DFLASH2 = "dflash2"    # the DFlash2 block drafter (Qwen3.8-Flash-Next only, one request at a time)
    AUTO = "auto"          # upstream's rule: DFlash2 where it runs, else MTP
    OFF = "off"


class Vision(StrEnum):
    OFF = "off"
    CPU = "cpu"
    GPU = "gpu"


class Prepare(StrEnum):
    AUTO = "auto"          # download and build what is missing before the server starts
    OFF = "off"            # use the files as they are


class DraftVocab(StrEnum):
    """The draft token subset (the files upstream ships in data/)."""
    CJK = "cjk"
    EN = "en"
    CYRILLIC = "cyrillic"
    FR = "fr"


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Settings:
    """Every variable strata-slim reads, with its default: the field name in uppercase is the variable
    (model_repo <- MODEL_REPO). Unset and empty both keep the default. example.env explains each one.

    Not here, because they pass through as they are: STRATA_* / FNX_* / DS4_* (engine switches, read by the engine)
    and CONFIG_<KEY> (any other config key, see launch.py)."""

    # the model
    model_repo: str | None = None
    model_quant: str | None = None
    model_revision: str | None = None       # None: main on the first download, then the recorded commit
    mmproj_repo: str | None = None          # None: model_repo
    hf_token: str | None = None
    hf_endpoint: str = "https://huggingface.co"
    pack_args: str = ""
    drafter: Drafter = Drafter.MTP
    mtp_experts: str = "q2_0"
    draft_vocab: DraftVocab | None = None   # None: cjk for MTP, upstream's default for DFlash2
    model_name: str | None = None           # None: from the GGUF file name

    # the engine
    context: str | None = None              # tokens, or with k (x 1024): 256k
    kv: str | None = None
    vram_reserve_mib: int | None = None
    gpus: str | None = None                 # card numbers: 0,1
    engine_args: str = ""

    # pictures
    vision: Vision = Vision.OFF
    mmproj: Path | None = None              # None: the one mmproj*.gguf in or beside the model folder
    vision_threads: int | None = None
    vision_max_tokens: int | None = None

    # the server
    api_key: str | None = None
    api_key_file: Path | None = None
    server_args: str = ""
    host: str = "0.0.0.0"
    port: int = 8080

    # where things are (the defaults match the folders compose.yaml mounts)
    prepare: Prepare = Prepare.AUTO
    models_dir: Path = Path("/models")
    model_gguf: Path | None = None          # None: derived from model_repo and model_quant
    pack: Path | None = None                # None: derived from the model folder
    tokenizer: Path | None = None           # None: <pack>/tokenizer
    expert_profile: str = str(ROOT / "data" / "expert-profile.bin")    # or "none"
    config_out: Path = Path("/data/run/config.json")
    log_file: Path = Path("/data/logs/engine.log")
    log_level: LogLevel = LogLevel.INFO

    @classmethod
    def load(cls) -> Settings:
        """The settings from the environment, after loading the env file ENV_FILE (default /data/.env) when it
        is there. A variable already set (compose's env_file, docker run -e) wins over the file."""
        load_dotenv(os.environ.get("ENV_FILE") or "/data/.env")

        types = get_type_hints(cls)
        values = {}
        for field in fields(cls):
            name = field.name.upper()
            raw = os.environ.get(name, "").strip()
            if not raw:
                continue

            kind = types[field.name]
            if get_args(kind):              # X | None -> X
                kind = get_args(kind)[0]

            if kind is int:
                if not raw.isdigit():
                    raise SetupError(f"{name}={raw}: a whole number")
                values[field.name] = int(raw)
            elif kind is Path:
                values[field.name] = Path(raw)
            elif issubclass(kind, StrEnum):
                try:
                    values[field.name] = kind(raw.lower())
                except ValueError:
                    raise SetupError(f"{name}={raw}: one of {', '.join(kind)}")
            else:
                values[field.name] = raw

        return cls(**values)

    @property
    def vision_on(self) -> bool:
        return self.vision != Vision.OFF


# ---- where the model files go ---------------------------------------------------------------------------------

@dataclass
class ModelPaths:
    model: Path            # the model's GGUF file, or the folder that holds its shards
    pack: Path
    mtp: Path | None       # the MTP draft layer's runtime folder; None when it is not used

    @property
    def data_dir(self) -> Path:
        """The models root as the server finds it: <root>/packs/<tag> -> <root>. The DFlash2 drafter is in <root>/dflash2/."""
        return self.pack.parent.parent

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelPaths:
        """Where the model files go, derived from MODEL_REPO + MODEL_QUANT under MODELS_DIR:

            <MODELS_DIR>/models/<owner>-<quant>/     the GGUF files (owner: the repository's owner, e.g. unsloth)
            <MODELS_DIR>/packs/<owner>-<quant>/      the pack (lowercase)
            <MODELS_DIR>/mtp/rt/                     the MTP draft layer, one for every Qwen3.8-Flash-Next model
            <MODELS_DIR>/dflash2/                    the DFlash2 drafter (with DRAFTER=dflash2 or auto)

        which is upstream's layout and folder naming. MODEL_GGUF and PACK override the first two.
        The MTP layer is used only with DRAFTER=mtp or auto."""
        root = settings.models_dir

        if settings.model_gguf:
            model = settings.model_gguf
        elif settings.model_repo:
            owner, _, name = settings.model_repo.partition("/")
            model = root / "models" / f"{owner}-{settings.model_quant or name}"
        else:
            raise SetupError("set MODEL_REPO (and MODEL_QUANT) to download a model, "
                             "or MODEL_GGUF to a model already there")

        folder = model if model.suffix != ".gguf" else model.parent
        pack = settings.pack or root / "packs" / folder.name.lower()

        mtp = None
        if settings.drafter in (Drafter.MTP, Drafter.AUTO):
            mtp = root / "mtp" / "rt"
        return cls(model, pack, mtp)
