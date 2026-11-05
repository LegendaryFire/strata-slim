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

`docker-compose.yml`:

```yaml
services:
  strata-slim:
    image: strata-slim:0.2.10
    build:
      context: https://github.com/LegendaryFire/strata-slim.git
      args:
        CUDA_ARCHITECTURES: "86"        # 86 = RTX 30, 89 = RTX 40, 120 = RTX 50
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

`strata-slim.env`:

```ini
MODEL_REPO=unsloth/Qwen3.8-Flash-Next-GGUF
MODEL_QUANT=UD-Q4_K_XL
PACK_ARGS=--compat-bf16
DRAFTER=mtp

CONTEXT=64k
KV=int8
GPUS=0

VISION=off
API_KEY=change-me
```

Then:

```sh
docker compose up -d --build
docker compose logs -f
```

The first start downloads and prepares the model (well over 100 GB), then the API is at `http://localhost:8080/v1`.
Every setting is in [`example.env`](example.env); AMD cards, network access and troubleshooting are in
[docs/docker-compose.md](docs/docker-compose.md).
