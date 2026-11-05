#!/usr/bin/env python3
"""strata-slim entrypoint: prepare the model, build the run config from environment variables, start the server.

Every start writes the config fresh from the environment, so the env file is the whole truth about how the engine
runs: nothing is read from an earlier config, there are no presets.

    docker run ... strata-slim                 # prepare what is missing (prepare.py), write the config, start the server
    docker run ... strata-slim prepare         # only download / pack / MTP, then exit (no GPU needed)
    docker run ... strata-slim print-config    # write and print the config, do not start (no GPU needed)

Every variable, with its default, is in example.env.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

import prepare
from common import ROOT, ModelPaths, drafter, env, fail, first_shard, mmproj_near, model_paths, say, vision_on

ENGINE = ROOT / "engine"


def engine_backend() -> str:
    """The engine's GPU backend, from the build record the Dockerfile writes beside it: "cuda" (./Dockerfile; also when
    there is no record) or "hip" (Dockerfile.rocm)."""
    try:
        return str(json.loads((ENGINE / "BUILD.json").read_text(encoding="utf-8")).get("backend") or "cuda")
    except (OSError, ValueError, AttributeError):
        return "cuda"


BACKEND = engine_backend()
SECRET_KEYS = {"api_key"}
# engine flags that have a variable of their own, so ENGINE_ARGS must not set them a second time
FLAGS_WITH_A_VARIABLE = ("--pack", "--native", "--expert-profile", "--mtp", "--max-context", "--kv",
                         "--vram-reserve-mib")


def must_exist(name: str, path: Path, kind: str) -> Path:
    exists = path.is_dir() if kind == "folder" else path.is_file()
    if not exists:
        fail(f"{name}={path}: no such {kind} inside the container (check the volume mounts)")
    return path


def context_tokens(raw: str) -> str:
    m = re.fullmatch(r"(\d+)\s*([kK]?)", raw)
    if not m:
        fail(f"CONTEXT={raw}: a number of tokens, or e.g. 32k / 256k")
    return str(int(m.group(1)) * (1024 if m.group(2) else 1))


def default_model_name(gguf: Path) -> str:
    name = re.sub(r"-\d{5}-of-\d{5}$", "", gguf.stem)
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")


def json_or_text(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


# ---- the config -------------------------------------------------------------------------------------------------

def engine_args(model: Path, paths: ModelPaths) -> list[str]:
    args = ["--pack", str(paths.pack), "--native", str(model)]
    profile = env("EXPERT_PROFILE") or str(ROOT / "data" / "expert-profile.bin")
    if profile.lower() != "none":
        args += ["--expert-profile", str(must_exist("EXPERT_PROFILE", Path(profile), "file"))]
    if paths.mtp:
        args += ["--mtp", str(must_exist("MTP", paths.mtp, "folder"))]
    if (ctx := env("CONTEXT")):
        args += ["--max-context", context_tokens(ctx)]
    if (kv := env("KV")):
        args += ["--kv", kv]
    if (reserve := env("VRAM_RESERVE_MIB")):
        if not reserve.isdigit():
            fail(f"VRAM_RESERVE_MIB={reserve}: a number of MiB, e.g. 550")
        args += ["--vram-reserve-mib", reserve]

    extra = shlex.split(env("ENGINE_ARGS") or "")
    for flag in FLAGS_WITH_A_VARIABLE:
        if flag in extra:
            fail(f"ENGINE_ARGS has {flag}: set it with its own variable instead")
    return args + extra


def vision_section(model: Path, mmproj: Path) -> dict:
    gpu = env("VISION", "").lower() == "gpu"
    cpu_reader = must_exist("vision reader", ENGINE / "strata-vision", "file")
    # NVIDIA: one program reads on the GPU or, without --gpu, the CPU. AMD: the GPU reader is its own program
    # (strata-vision-hip), and the server falls back to the CPU one (cpu_exe) when it does not start.
    exe = must_exist("vision reader", ENGINE / "strata-vision-hip", "file") if gpu and BACKEND == "hip" else cpu_reader
    section = {"exe": str(exe),
               "mmproj": str(must_exist("MMPROJ", mmproj, "file")),
               "model": str(model),
               "gpu": gpu}
    if exe != cpu_reader:
        section["cpu_exe"] = str(cpu_reader)
    if (threads := env("VISION_THREADS")):
        section["threads"] = int(threads)
    if (tokens := env("VISION_MAX_TOKENS")):
        section["max_tokens"] = int(tokens)
    return section


def other_config_keys() -> dict:
    """CONFIG_<KEY>=value -> {"key": value}, warning about keys the server does not read."""
    keys_file = ROOT / ".config-keys"
    known = set(keys_file.read_text().split()) if keys_file.is_file() else set()
    out = {}
    for name in sorted(os.environ):
        if not name.startswith("CONFIG_") or name == "CONFIG_OUT" or not env(name):
            continue
        key = name.removeprefix("CONFIG_").lower()
        if key in ("exe", "args", "tokenizer", "vision", "drafting"):
            fail(f"{name}: {key} is built from its own variables; set those instead")
        if known and key not in known:
            say(f"warning: {name}: the server does not read a config key {key!r} (typo?)")
        out[key] = json_or_text(env(name))
    return out


def build_config(paths: ModelPaths, model: Path, mmproj: Path | None) -> dict:
    must_exist("PACK", paths.pack, "folder")
    tokenizer = Path(env("TOKENIZER") or paths.pack / "tokenizer")
    must_exist("TOKENIZER", tokenizer / "vocab.json", "file")

    cfg = {
        "exe": str(ENGINE / "strata"),
        "args": engine_args(model, paths),
        "cwd": str(ROOT),
        "tokenizer": str(tokenizer),
        "model_name": env("MODEL_NAME") or default_model_name(model),
        "log": env("LOG_FILE") or "/data/logs/engine.log",
        "lib_dirs": [],
    }
    if BACKEND != "cuda":
        cfg["backend"] = BACKEND     # the server's default is NVIDIA: "hip" makes it number cards for HIP (HIP_VISIBLE_DEVICES)
    if (gpus := env("GPUS")):
        try:
            cfg["gpu"] = [int(g) for g in gpus.replace(" ", "").split(",")]
        except ValueError:
            fail(f"GPUS={gpus}: card numbers, e.g. 0 or 0,1")
        cfg["gpus_asked"] = True
    if mmproj:
        cfg["args"].append("--vision")
        cfg["vision"] = vision_section(model, mmproj)
    # Always said, never left to the server's default ("auto" switches to DFlash2 as soon as its files are there).
    # The server turns it into the engine flags at each start: for dflash2 it drops --mtp and adds DFlash2's own.
    cfg["drafting"] = drafter()
    cfg.update(other_config_keys())
    return cfg


# ---- starting ---------------------------------------------------------------------------------------------------

def model_files(command: str, paths: ModelPaths) -> tuple[Path, Path | None]:
    """(the GGUF for the engine, the mmproj or None): prepared by prepare.py, or as they are when PREPARE=off."""
    if command == "prepare" or (command == "serve" and (env("PREPARE") or "auto").lower() != "off"):
        prepared = prepare.run(paths, vision_on())
        return prepared.model, prepared.mmproj

    model = first_shard(paths.model)
    if not vision_on():
        return model, None
    mmproj = env("MMPROJ") or mmproj_near(model.parent)
    if not mmproj:
        fail(f"VISION is on and there is not exactly one mmproj*.gguf in or beside {model.parent}: set MMPROJ")
    return model, Path(mmproj)


def write_config(cfg: dict) -> Path:
    out = Path(env("CONFIG_OUT") or "/data/run/config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    shown = {k: ("***" if k in SECRET_KEYS else v) for k, v in cfg.items()}
    say(f"config ({out}):\n" + json.dumps(shown, indent=1))
    return out


def start_server(config: Path) -> None:
    key = env("API_KEY")
    if not key and (key_file := env("API_KEY_FILE")):
        key = must_exist("API_KEY_FILE", Path(key_file), "file").read_text().strip()
    if key:
        os.environ["STRATA_API_KEY"] = key
    else:
        os.environ.pop("STRATA_API_KEY", None)   # set but empty would stop the server; unset means no key

    argv = [sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(config),
            "--host", env("HOST") or "0.0.0.0", "--port", env("PORT") or "8080",
            *shlex.split(env("SERVER_ARGS") or "")]
    say(" ".join(shlex.quote(a) for a in argv))
    os.chdir(ROOT)
    os.execv(argv[0], argv)


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command not in ("serve", "prepare", "print-config"):
        fail(f"unknown command {command}: serve (default), prepare or print-config")

    paths = model_paths()
    model, mmproj = model_files(command, paths)
    if command == "prepare":
        return 0
    config = write_config(build_config(paths, model, mmproj))
    if command == "serve":
        start_server(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
