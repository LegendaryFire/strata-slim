# Running strata-slim with Docker Compose

The server's settings live in one file, `.env`; `compose.yaml` holds the build, the port and the folders. Start it
with `docker compose up -d`. On the first start the
model is downloaded and prepared; after that, it starts in a few minutes.

## What you need

- Docker with Compose (`docker compose version`).
- **NVIDIA:** the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check it with `docker run --rm --gpus all ubuntu nvidia-smi`.
- **AMD:** an RDNA3 or RDNA4 card (RX 7000 / 9000) and the Linux amdgpu driver, which ships with the kernel. You don't need
  to install ROCm.
- Disk: well over 100 GB for the default model (the download plus the pack built from it).

## 1. Settings

```sh
cp example.env .env
```

In `compose.yaml`, check:

- `CUDA_ARCHITECTURES`: your card, `86` RTX 30, `89` RTX 40, `120` RTX 50.
- On AMD, swap the commented lines: the `-rocm` image, `Dockerfile.rocm`, `HIP_ARCHITECTURES` (your card: `gfx1100`
  RX 7900, `gfx1101` RX 7800 / 7700, `gfx1201` RX 9070, `gfx1200` RX 9060), and `devices` in place of `deploy`.
- The model goes in `./models` next to `compose.yaml` (100+ GB); change that volume line to put it on another disk.

Then open `.env` and check these lines:

| Setting | What to put |
|---|---|
| `GPUS` | `0` for one card, `0,1` for two |
| `ENGINE_ARGS` | the example is tuned for two RTX 3060s; if your setup is different, use `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5` |

Everything else can stay as it is.

## 2. Build the image

```sh
docker compose build
```

This compiles the engine, which takes a while (20 to 60 minutes, depending on the machine and how many architectures
you build for). You only build again after an update. If the build runs out of memory, lower `BUILD_JOBS`.

## 3. Start it

```sh
docker compose up -d
docker compose logs -f
```

The first start downloads the model, builds its pack and the MTP layer, and then starts the server. The log shows each
step with progress. A start that was interrupted continues where it stopped.

When the log says the server is listening, try it:

```sh
curl http://localhost:8080/v1/models
```

The API is OpenAI- and Anthropic-compatible: point a client at `http://localhost:8080/v1`.

## Everyday commands

| | |
|---|---|
| `docker compose up -d` | start (and apply changes in `.env`) |
| `docker compose down` | stop |
| `docker compose restart` | restart |
| `docker compose logs -f` | follow the log (`Ctrl+C` stops following, not the server) |
| `docker compose ps` | is it running and healthy? |

Editing `.env` takes effect on the next `docker compose up -d`. No rebuild is needed, except for the build args in
`compose.yaml` (`CUDA_ARCHITECTURES`, `HIP_ARCHITECTURES`, `BUILD_JOBS`): `docker compose build` after changing those.

Two useful one-offs:

```sh
docker compose run --rm strata-slim print-config   # show the config the settings make, without starting
docker compose run --rm strata-slim prepare        # only download and prepare the model, then exit
```

## Opening it to your network

By default only this machine can reach the API (`127.0.0.1:8080:8080` in `compose.yaml`). To let other machines
connect, change that line to `8080:8080` and set a key in `.env`:

```ini
API_KEY=some-long-random-string
```

Clients then send `Authorization: Bearer some-long-random-string`. Generate a key with `openssl rand -hex 32`.

Behind a reverse proxy (Caddy, nginx, Traefik) with no API key, also name the host the proxy uses:

```ini
STRATA_ALLOWED_HOSTS=llm.example.com
```

## Where things are

| On the host | What |
|---|---|
| `models/models/` | the downloaded model files |
| `models/packs/` | the pack built from them |
| `models/mtp/` | the MTP draft layer |
| `data/run/config.json` | the config generated on each start |
| `data/logs/engine.log` | the engine's log |

Both folders are next to `compose.yaml`. If you already have the model downloaded, point the `/models` volume at the
folder that holds `models/`. strata-slim checks those
files and uses them instead of downloading again.

## Updating

```sh
git pull
docker compose build
docker compose up -d
```

## When something goes wrong

| What you see | What to do |
|---|---|
| `could not select device driver "nvidia"` | install the NVIDIA Container Toolkit and restart Docker |
| `/dev/kfd: no such file` (AMD) | the amdgpu driver is not loaded, or this is not an AMD machine |
| the engine stops saying the card has no code | put your card in `CUDA_ARCHITECTURES` / `HIP_ARCHITECTURES` in `compose.yaml` and build again |
| out of memory on the GPU | lower `CONTEXT` (e.g. `64k`), or raise `VRAM_RESERVE_MIB` if other programs use the card |
| the build is killed | lower `BUILD_JOBS` in `compose.yaml` |
| a setting seems ignored | `print-config` (above) shows what the server gets; the log warns about `CONFIG_*` names it does not know |
