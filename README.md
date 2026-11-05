# strata-slim

The Strata engine and its API server, built from upstream [Niko1221/Strata](https://github.com/Niko1221/Strata) with
everything else left out: no desktop app, Terminal/TUI, harness, setup, benchmarks, docs or tests. The run config is
built from environment variables on every start.

| | upstream repo | vendor/ |
|---|---|---|
| files | ~6,000 | ~880 |
| size | 100 MB | 23 MB |

`vendor/UPSTREAM` says which tag and commit `vendor/` is.

## Layout

| file | what it is |
|---|---|
| `manifest.txt` | the upstream paths taken (everything else stays out) |
| `runtime-packages.txt` | the Python packages installed; versions come from upstream's pins |
| `extract.sh` | upstream tag → `vendor/`, then `patches/` on top |
| `check.py` | reads `vendor/` and fails when the server would miss a module or package |
| `patches/` | your own changes to upstream code, reapplied on every extract |
| `launch.py` | the entrypoint: env vars → run config → `serve.server` |
| `prepare.py` | download the model, build the pack and the MTP layer (whatever is missing) |
| `hf.py` | Hugging Face: list files at a commit, verified resumable downloads |
| `common.py` | what launch.py and prepare.py share (variables, messages, finding model files) |
| `Dockerfile` | NVIDIA: stage 1 compiles the engine (upstream's flags); stage 2 is CUDA runtime + Python + server |
| `Dockerfile.rocm` | AMD Radeon (RDNA): the same with the HIP backend; stage 2 is plain Ubuntu + the ROCm libraries the engine loads |
| `example.env` | a full env file |
| `vendor/` | the extracted source (committed, so a diff between versions is a `git diff`) |

## A new upstream release

    ./extract.sh v0.2.11          # vendor/ replaced; fails if a manifest entry or a patch no longer fits
    ./check.py                    # fails if the server now imports something vendor/ lacks
    git diff --stat vendor/       # what changed in the parts you use
    docker build -t strata-slim:0.2.11 .
    docker build -f Dockerfile.rocm -t strata-slim:0.2.11-rocm .     # if you build the AMD image
    git add -A && git commit -m "Upstream v0.2.11"

Roll back: start the previous image tag (images stay until you remove them), or `git checkout` the earlier commit and
build it again.

`./extract.sh` keeps a bare clone of upstream in `.upstream/` (git-ignored). `UPSTREAM_GIT=/path/to/clone` uses a clone
you already have instead.

The compile flags in `Dockerfile` and `Dockerfile.rocm` are copied by hand from upstream's own build files (its root
`Dockerfile` and `packaging/build_hip_linux.sh`), so nothing checks them: look at upstream's diff of those two files
on each release.

## AMD Radeon (RDNA)

`Dockerfile.rocm` builds the HIP engine, upstream's backend for wave32 RDNA cards:

    docker build -f Dockerfile.rocm -t strata-slim:0.2.10-rocm .
    docker build -f Dockerfile.rocm -t strata-slim:0.2.10-rocm --build-arg HIP_ARCHITECTURES=gfx1201 .   # one card: faster

| `HIP_ARCHITECTURES` | cards | upstream's status |
|---|---|---|
| gfx1100 | RX 7900 XT / XTX | tested by the maintainers |
| gfx1201 | RX 9070 / 9070 XT, Radeon AI PRO R9700 | tested by the maintainers |
| gfx1101 | RX 7800 XT / 7700 XT | validated by owners |
| gfx1200 | RX 9060 XT | validated by owners |
| gfx1102 | RX 7600 | reported working |
| gfx1030, gfx1031 | RX 6800 / 6900, RX 6700 XT (RDNA2) | builds, not run on a card by upstream |

The default is the first five (RDNA3 and RDNA4). Every architecture is compiled into every kernel, so naming only
yours cuts the build time. An engine on a card it has no code for stops at start with a message naming what it has.
The older Instinct MI50 / Radeon VII (gfx906) is a separate upstream backend and is not built here.

The build image is `rocm/dev-ubuntu-24.04:7.14.1-full` (`--build-arg ROCM_IMAGE=` for another). The runtime image is
plain Ubuntu: upstream's `packaging/collect_libs.py` copies every ROCm library the engine and picture readers load,
and the rocBLAS / hipBLASLt kernel data for the built architectures only, into `/opt/strata/engine`. The host needs
the kernel's amdgpu driver and nothing from ROCm.

Run it with the AMD devices instead of `--gpus` (Compose: `devices:`):

    docker run --device /dev/kfd --device /dev/dri --env-file x.env -v ... strata-slim:0.2.10-rocm

What differs from the NVIDIA image (`launch.py` reads `engine/BUILD.json` and does it):

- the config gets `"backend": "hip"`, so the server sets `HIP_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`
- `GPUS` are HIP's numbers (the KFD order). Unset, the server asks `strata-device --list-devices` and takes a card the
  engine has code for, skipping integrated chips
- `VISION=gpu` runs `strata-vision-hip`, and the server falls back to the CPU reader when it does not start
- the Monitor's card readings come from the amdgpu sysfs files (no ROCm tool needed)
- hipBLASLt tuning tables (`STRATA_HIPBLASLT_TABLE`) are not shipped; upstream leaves them off on these cards anyway

## Model prep: download, pack, MTP

On a start, `prepare.py` makes whatever is missing, then the server starts (`PREPARE=off` skips it; the `prepare`
command does only this and exits, no GPU needed):

1. **Download:** `MODEL_REPO` + `MODEL_QUANT` (a missing or wrong one lists the repository's quantizations with
   their sizes) go into the `MODEL_GGUF` folder. With `VISION` on, the mmproj comes from `MMPROJ_REPO` (default
   `MODEL_REPO`), chosen by itself when the repository has one (`MMPROJ_FILE` otherwise), and is saved beside the
   model folder. Each file is checked against the size and SHA-256 Hugging Face publishes, resumes where it stopped,
   and gets a `.done` mark (the mark upstream's setup writes, so its downloads count as done).

   The first download writes `.source.json` into the model folder: the commit and every file's SHA-256. From then on
   that folder is that version, even when the repository replaces its files (Unsloth did for Qwen3.8-Flash-Next), and
   a start with all files present needs no network. A newer version goes into a new folder with a new pack. A folder
   downloaded earlier without a record is adopted: the commit whose files match its `.done` marks is found and recorded.
2. **Pack:** `tools/iq_pack.py` with `PACK_ARGS` (`--compat-bf16` for Unsloth's UD-* files).
3. **MTP draft layer** (when `MTP` is set and `$MTP/experts.bin` is missing): `tools/mtp_fetch.py` fetches the 31 MTP
   tensors (~5 GB) of the Qwen3.8-Flash-Next checkpoint, `mtp_pack.py` (`MTP_EXPERTS`, default q2_0) and `mtp_rt.py`
   build the runtime files, and `DRAFT_VOCAB` (cjk, en, cyrillic, fr) picks the draft token subset.

4. **DFlash2 drafter** (`DRAFTER=dflash2` or `auto`): upstream's own `strata_control/drafting_fetch.py` downloads the
   pinned `incoai/Qwen3.8-Flash-Next-DFlash2` files (1.2 GB), checks them, packs them to Q4_0 (0.5 GB, checked against
   the exact bytes upstream measured) into `<models>/dflash2/`, with the draft vocabulary beside it.

The MTP fetch and DFlash2 are specific to Qwen3.8-Flash-Next (upstream pins both).

### Which drafter

`DRAFTER` (`mtp` by default, `dflash2`, `auto`, `off`) is written to the config as `drafting`, and the server turns it
into engine flags at each start: for DFlash2 it drops `--mtp` and adds `--draft dflash2 --spec 6 --draft-conf ...`
(upstream's tuned values; a `--spec` in `ENGINE_ARGS` is replaced by DFlash2's window). It is always written: left
unset, the server's default `auto` would switch to DFlash2 by itself as soon as its files exist. DFlash2 runs one
request at a time (a start with `--parallel` falls back to MTP) and, per upstream's tests, is slower than MTP on RTX 30
cards (sm_86). With `DRAFTER=dflash2` the MTP layer is not built or used.

## Configuration

Every start writes `/data/run/config.json` fresh from the environment; nothing from an earlier start is reused.
`docker run --rm --env-file x.env -v ... strata-slim:0.2.10 print-config` shows the config without starting
anything (no GPU needed). The variables are listed at the top of `launch.py`; in short:

- **Model files:** derived from `MODEL_REPO` + `MODEL_QUANT` under `MODELS_DIR` (default `/models`):
  `models/<owner>-<quant>/`, `packs/<owner>-<quant>/`, `mtp/rt/`, `dflash2/` (upstream's layout and naming).
  `MODEL_GGUF` and `PACK` override the first two; `TOKENIZER`, `EXPERT_PROFILE`
- **Engine:** `CONTEXT`, `KV`, `VRAM_RESERVE_MIB`, and `ENGINE_ARGS` for every other engine flag
- **Engine switches:** `STRATA_*` / `FNX_*` / `DS4_*` reach the engine unchanged
- **GPUs:** `GPUS=0,1`
- **Pictures:** `VISION=off|cpu|gpu`, `MMPROJ`, `VISION_THREADS`, `VISION_MAX_TOKENS`
- **Server:** `MODEL_NAME`, `HOST`, `PORT`, `API_KEY` / `API_KEY_FILE`, `SERVER_ARGS`
- **Any other config key:** `CONFIG_<KEY>=<json or string>`, e.g. `CONFIG_IDLE_UNLOAD_S=600`. A key the server does
  not read is a warning (the list is `vendor/.config-keys`, made by `check.py`).

## Volumes

The image reads model files from wherever you mount them, and writes only to `/data` (the generated config,
logs). For example, with `example.env`:

```yaml
volumes:
  - ./strata/data-slim:/data
  - /mnt/models/strata/models:/models/models:ro
  - /mnt/models/strata-slim/packs:/models/packs
  - /mnt/models/strata-slim/mtp:/models/mtp
```

## What it does not do

- Pick a model for your hardware, or know model families: you name the repository, the files and the pack flags.
- Other pack paths of upstream's setup: the Q2_0 AVX-512 repack (`strata_pack.py`), NVFP4 and DeepSeek packs. Add the
  tool to `manifest.txt` and a step to `prepare.py` if you need one.
- Tuning (`TUNE` / calibrate), which is part of upstream's setup.
