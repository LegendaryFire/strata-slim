<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/strata-lockup-dark.svg">
    <img src="docs/media/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

**strata-slim** is [Strata](https://github.com/Niko1221/Strata), the real one, built straight from its releases, with
everything a server doesn't need left on the cutting room floor. No desktop app, no installer, no terminal UI, no
benchmark suites, no test farm. Just the engine, the API server and the tools that get a model ready, packed into a
Docker image and nothing else.

It's made for containers. Every setting lives in one `.env` file, and every start builds the config fresh from it, so
what you see in that file is exactly what runs: no presets, no leftover state, no clicking through setup. Pick NVIDIA
or AMD, name a model, run `docker compose up -d`, and it downloads, checks, packs and serves it on its own.

## Docker Compose

Images are published for NVIDIA (`cuda`: RTX 30, 40 and 50) and AMD (`rocm`: RX 7000 and 9000). Each release is
also tagged with its version, such as `0.2.12-1-cuda`, for pinning.

`docker-compose.yml`:

```yaml
services:
  strata-slim:
    image: ghcr.io/legendaryfire/strata-slim:cuda
    env_file: strata-slim.env
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./models:/models
      - ./data:/data
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

On AMD, use the `rocm` image and pass the GPU devices instead of the `deploy` section:

```yaml
    image: ghcr.io/legendaryfire/strata-slim:rocm
    devices:
      - /dev/kfd
      - /dev/dri
```

`strata-slim.env`:

```ini
# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

# Hugging Face repository. Files are downloaded on the first start, verified by SHA-256 and reused.
MODEL_REPO=unsloth/Qwen3.8-Flash-Next-GGUF

# Quantization to use, for example UD-IQ4_XS, UD-Q4_K_XL or UD-Q5_K_XL.
MODEL_QUANT=UD-Q4_K_XL

# Commit or branch to download. Default: main.
# MODEL_REVISION=

# Access token for gated or private repositories.
# HF_TOKEN=

# Model name reported by /v1/models. Default: derived from the file name.
# MODEL_NAME=

# Pack options. Use --compat-bf16 for Unsloth UD-* files.
PACK_ARGS=--compat-bf16

# ----------------------------------------------------------------------
# Speculative decoding
# ----------------------------------------------------------------------

# Drafter: mtp, dflash2, auto or off. Default: mtp.
DRAFTER=mtp

# Draft vocabulary: cjk, en, cyrillic or fr. Default: cjk.
# DRAFT_VOCAB=cjk

# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

# Context length in tokens, for example 32768, 32k or 256k. Default: set by the engine.
CONTEXT=64k

# KV cache type: fp16, int8, q4 or k8v4. Default: fp16. int8 uses half the VRAM.
KV=int8

# VRAM to keep free on each card, in MiB. Default: set by the engine.
# VRAM_RESERVE_MIB=550

# Additional engine flags.
# ENGINE_ARGS=--expert-cache auto --spec 4

# ----------------------------------------------------------------------
# GPUs
# ----------------------------------------------------------------------

# GPUs to use, for example 0 or 0,1. Default: chosen by the server.
GPUS=0

# Layer split across several GPUs: auto, or the first layer of each additional GPU (for example 24). Default: auto.
# CONFIG_LAYER_SPLIT=auto

# ----------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------

# Image input: off, cpu or gpu. Default: off.
VISION=off

# Repository providing the mmproj file. Default: MODEL_REPO.
MMPROJ_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF

# CPU threads for VISION=cpu.
# VISION_THREADS=8

# Maximum tokens per image, 0 for no limit. Default: 300 on CPU, 1024 on GPU.
# VISION_MAX_TOKENS=300

# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------

# API key, sent by clients as "Authorization: Bearer <key>". Default: none (keep the port private).
API_KEY=change-me

# Host names allowed behind a reverse proxy when no API key is set.
# STRATA_ALLOWED_HOSTS=llm.example.com
```

Then:

```sh
docker compose up -d
docker compose logs -f
```

The first start downloads and prepares the model (well over 100 GB), then the API is at `http://localhost:8080/v1`.
Every setting is in [`example.env`](example.env); network access, troubleshooting and building the images yourself are
in [docs/docker-compose.md](docs/docker-compose.md).

## Common settings

| Variable | Valid values | Default |
|---|---|---|
| `MODEL_REPO` | a Hugging Face repository with GGUF files, e.g. `unsloth/Qwen3.8-Flash-Next-GGUF` | none |
| `MODEL_QUANT` | a quantization that repository has, e.g. `UD-Q4_K_XL`, `UD-IQ4_XS` (a wrong one lists them) | its only one |
| `PACK_ARGS` | `--compat-bf16` for Unsloth's `UD-*` files | none |
| `DRAFTER` | `mtp`, `dflash2`, `auto`, `off` | `mtp` |
| `DRAFT_VOCAB` | `cjk`, `en`, `cyrillic`, `fr` | `cjk` |
| `MTP_EXPERTS` | `q2_0`, `q4_0`, `q8_0` | `q2_0` |
| `CONTEXT` | tokens: `32768`, or with `k`: `32k`, `64k`, `256k` | the engine's |
| `KV` | `fp16`, `int8` (half the VRAM), `q4`, `k8v4` | `fp16` |
| `VRAM_RESERVE_MIB` | MiB of VRAM kept free, e.g. `550` | the engine's |
| `GPUS` | card numbers: `0`, `0,1` | the server's choice |
| `CONFIG_LAYER_SPLIT` | `auto`, or the first layer of each later GPU: `24` (2 GPUs), `24,36,42` (4 GPUs) | `auto` |
| `VISION` | `off`, `cpu`, `gpu` | `off` |
| `VISION_MAX_TOKENS` | tokens per picture, `0` for no limit | `300` on CPU, `1024` on GPU |
| `API_KEY` | any string; clients send it as `Authorization: Bearer ...` | none (no key) |
| `MODEL_NAME` | the name `/v1/models` lists | from the file name |
| `PREPARE` | `auto`, `off` | `auto` |
| `ENGINE_ARGS` | any other engine flags, e.g. `--spec 4 --expert-cache auto` | none |

## License

[MIT](LICENSE). The upstream Strata source in `vendor/` keeps its own [MIT license](vendor/LICENSE).
