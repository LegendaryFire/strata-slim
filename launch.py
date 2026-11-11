#!/usr/bin/env python3
"""strata-slim entrypoint: prepare the model, build the run config from the settings, start the server.

The config is written fresh from the settings on every start (settings.Settings: the environment, plus an optional
env file), so they are the whole truth about how the engine runs. Every variable, with its default, is in example.env.

    docker run ... strata-slim                 # prepare what is missing, write the config, start the server
    docker run ... strata-slim prepare         # only download / pack / MTP, then exit (no GPU needed)
    docker run ... strata-slim print-config    # write and print the config, do not start (no GPU needed)

On Windows, start.bat runs it the same way: start.bat, start.bat prepare, start.bat print-config.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import prepare
from common import ON_WINDOWS, ROOT, SetupError, engine_program, first_shard, mmproj_near
from settings import ModelPaths, Prepare, Settings, Vision

log = logging.getLogger(__name__)


class Config:
    """The server's run config (config.json), built from the settings."""

    def __init__(self, settings: Settings, paths: ModelPaths, model: Path, mmproj: Path | None):
        self.settings = settings
        self.paths = paths
        self.model = model
        self.mmproj = mmproj

        # the Dockerfile writes BUILD.json beside the engine; only Dockerfile.rocm sets a backend ("hip")
        self.backend = "cuda"
        build_record = Path(ROOT, "engine", "BUILD.json")
        if build_record.is_file():
            self.backend = json.loads(build_record.read_text()).get("backend", "cuda")

    def build(self) -> dict:
        s = self.settings
        if not self.paths.pack.is_dir():
            raise SetupError(f"no pack at {self.paths.pack}: build it with PREPARE=auto or set PACK to an existing one")

        tokenizer = s.tokenizer or Path(self.paths.pack, "tokenizer")
        if not Path(tokenizer, "vocab.json").is_file():
            raise SetupError(f"TOKENIZER={tokenizer}: no vocab.json in it")

        # Model-Q4-00001-of-00002.gguf -> model-q4
        model_name = re.sub(r"-\d{5}-of-\d{5}$", "", self.model.stem).lower()
        model_name = re.sub(r"[^a-z0-9._-]+", "-", model_name).strip("-")

        cfg = {
            "exe": str(engine_program("strata")),
            "args": self.engine_args(),
            "cwd": str(ROOT),
            "tokenizer": str(tokenizer),
            "model_name": s.model_name or model_name,
            "log": str(s.log_file),
            "lib_dirs": [],
        }

        # the server assumes NVIDIA; "hip" makes it number the cards for HIP_VISIBLE_DEVICES
        if self.backend != "cuda":
            cfg["backend"] = self.backend

        if s.gpus:
            cfg["gpu"] = []
            for gpu in s.gpus.split(","):
                cfg["gpu"].append(int(gpu))
            cfg["gpus_asked"] = True

        if self.mmproj:
            cfg["args"].append("--vision")
            cfg["vision"] = self.vision()

        # always set it: the server's default ("auto") switches to DFlash2 as soon as its files are there
        cfg["drafting"] = str(s.drafter)

        cfg.update(self.extra_keys())
        return cfg

    def engine_args(self) -> list[str]:
        s = self.settings
        args = ["--pack", str(self.paths.pack), "--native", str(self.model)]

        if s.expert_profile.lower() != "none":
            if not Path(s.expert_profile).is_file():
                raise SetupError(f"expert profile {s.expert_profile} does not exist: set EXPERT_PROFILE to the "
                                 "profile file, or EXPERT_PROFILE=none to run without one")
            args += ["--expert-profile", s.expert_profile]

        if self.paths.mtp:
            if not self.paths.mtp.is_dir():
                raise SetupError(f"no MTP draft layer at {self.paths.mtp}: start with PREPARE=auto to build it, "
                                 "or set DRAFTER=off to run without a drafter")
            args += ["--mtp", str(self.paths.mtp)]

        if s.context:
            context = s.context
            if context.lower().endswith("k"):
                context = str(int(context[:-1]) * 1024)
            args += ["--max-context", context]

        if s.kv:
            args += ["--kv", s.kv]

        if s.vram_reserve_mib is not None:
            args += ["--vram-reserve-mib", str(s.vram_reserve_mib)]

        extra = shlex.split(s.engine_args)
        missing_required = "--prefill" not in extra or "--spec" not in extra
        if missing_required:
            raise SetupError("Engine arguments is missing required parameters. --prefill and --spec must be set")
        # flags that have their own setting, so ENGINE_ARGS must not set them again
        flags_with_a_setting = ["--pack", "--native", "--expert-profile", "--mtp", "--max-context", "--kv",
                                "--vram-reserve-mib"]
        for flag in flags_with_a_setting:
            if flag in extra:
                raise SetupError(f"ENGINE_ARGS has {flag}: set it with its own variable instead")
        return args + extra

    def vision(self) -> dict:
        s = self.settings
        if not self.mmproj.is_file():
            raise SetupError(f"picture reader {self.mmproj} does not exist: set MMPROJ to the mmproj .gguf file, "
                             "or VISION=off")

        gpu = s.vision == Vision.GPU
        section = {
            "exe": str(engine_program("strata-vision")),
            "mmproj": str(self.mmproj),
            "model": str(self.model),
            "gpu": gpu,
        }

        # NVIDIA: strata-vision reads on the GPU too. AMD: the GPU reader is its own program, and the server
        # falls back to the CPU one when it does not start.
        if gpu and self.backend == "hip":
            section["exe"] = str(engine_program("strata-vision-hip"))
            section["cpu_exe"] = str(engine_program("strata-vision"))

        if s.vision_threads is not None:
            section["threads"] = s.vision_threads
        if s.vision_max_tokens is not None:
            section["max_tokens"] = s.vision_max_tokens
        return section

    def extra_keys(self) -> dict:
        """CONFIG_<KEY>=value becomes {"key": value}. The value is parsed as JSON when it is JSON.

        These are not in Settings: any key the server reads can be set, and .config-keys (made by
        scripts/extract.py from upstream's server) is the list to check a typo against."""
        known_keys = Path(ROOT, ".config-keys").read_text().split()
        # keys built from their own settings, so CONFIG_<KEY> must not set them
        built_keys = ["exe", "args", "tokenizer", "vision", "drafting"]

        keys = {}
        for name in sorted(os.environ):
            value = os.environ[name].strip()
            # CONFIG_OUT is a setting of its own (where config.json goes), not a config key
            if not name.startswith("CONFIG_") or name == "CONFIG_OUT" or not value:
                continue

            key = name.removeprefix("CONFIG_").lower()
            if key in built_keys:
                raise SetupError(f"{name}: {key} is built from its own variables; set those instead")
            if key not in known_keys:
                log.warning(f"{name}: the server does not read a config key {key!r} (typo?)")

            try:
                keys[key] = json.loads(value)
            except ValueError:
                keys[key] = value
        return keys


class Launcher:
    """One run of the entrypoint: serve (default), prepare or print-config."""

    def __init__(self, settings: Settings, command: str):
        if command not in ("serve", "prepare", "print-config"):
            raise SetupError(f"unknown command {command}: serve (default), prepare or print-config")
        self.settings = settings
        self.command = command

    def run(self) -> None:
        paths = ModelPaths.from_settings(self.settings)
        model, mmproj = self.model_files(paths)
        if self.command == "prepare":
            return

        config_path = self.write_config(Config(self.settings, paths, model, mmproj).build())
        if self.command == "serve":
            self.start_server(config_path)

    def model_files(self, paths: ModelPaths) -> tuple[Path, Path | None]:
        """The GGUF for the engine and the mmproj (None without vision): from prepare.py, or as-is with PREPARE=off."""
        s = self.settings
        if self.command == "prepare" or (self.command == "serve" and s.prepare == Prepare.AUTO):
            return prepare.Preparer(s, paths).run()

        model = first_shard(paths.model)
        if not s.vision_on:
            return model, None

        mmproj = s.mmproj or mmproj_near(model.parent)
        if not mmproj:
            raise SetupError(f"VISION is on and there is not exactly one mmproj*.gguf in or beside {model.parent}: "
                             "set MMPROJ")
        return model, mmproj

    def write_config(self, cfg: dict) -> Path:
        path = self.settings.config_out
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(cfg, f, indent=1)

        shown = dict(cfg)
        if "api_key" in shown:
            shown["api_key"] = "***"
        log.info(f"config ({path}):\n{json.dumps(shown, indent=1)}")
        return path

    def start_server(self, config_path: Path) -> None:
        s = self.settings
        api_key = s.api_key
        if not api_key and s.api_key_file:
            if not s.api_key_file.is_file():
                raise SetupError(f"API key file {s.api_key_file} does not exist: set API_KEY_FILE to the file "
                                 "that holds the key, or API_KEY to the key itself")
            api_key = s.api_key_file.read_text().strip()

        # an empty STRATA_API_KEY stops the server; unset means no key
        if api_key:
            os.environ["STRATA_API_KEY"] = api_key
        else:
            os.environ.pop("STRATA_API_KEY", None)

        # Restore the defaults in which maintain model quality and integrity,
        os.environ.setdefault("STRATA_ROUTE_TAIL_SKIP", "0")
        os.environ.setdefault("STRATA_HC_Q8", "1")

        argv = [sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(config_path),
                "--host", s.host, "--port", str(s.port)]
        argv += shlex.split(s.server_args)

        log.info(shlex.join(argv))
        if ON_WINDOWS:
            # execv on Windows starts the server as a new process and ends this one, so the console comes back while
            # the server runs and Ctrl+C no longer reaches it; wait for it instead
            server = subprocess.Popen(argv, cwd=ROOT)
            while True:
                try:
                    sys.exit(server.wait())
                except KeyboardInterrupt:
                    pass        # the server got the Ctrl+C too, and is stopping

        logging.shutdown()   # flush before the server replaces this process
        os.chdir(ROOT)
        os.execv(argv[0], argv)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [strata-slim] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO"
    )
    try:
        settings = Settings.load()
        logging.getLogger().setLevel(settings.log_level.upper())
        Launcher(settings, sys.argv[1] if len(sys.argv) > 1 else "serve").run()
    except SetupError as e:
        log.error(e)
        sys.exit(2)
