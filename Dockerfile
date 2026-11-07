# syntax=docker/dockerfile:1
# strata-slim: the Strata engine, its API server and model prep (download, pack, MTP), nothing else.
#
#   docker build -t strata-slim:0.2.11 .                                     # RTX 30 (sm_86), 4 compile jobs
#   docker build -t strata-slim:0.2.11 --build-arg CUDA_ARCHITECTURES="86;89" --build-arg BUILD_JOBS=8 .
#   docker build -t strata-slim:0.2.11-patched --build-arg PATCHES=all .    # with the local fixes (patches/README.md)
#
# CUDA_ARCHITECTURES: 86 = RTX 30, 89 = RTX 40, 120 = RTX 50 (the last one also gets PTX). BUILD_JOBS: each nvcc job
# can take several GB of RAM; raise it when the machine has the memory free.
#
# The compile flags are upstream's (its root Dockerfile, "engine-build" stage). vendor/ comes from scripts/extract.sh.
# The engine stage copies only the engine's sources, so a change to the server, the tools or launch.py does not
# recompile it; and ccache (a BuildKit cache mount) keeps object files between builds, so a new upstream release
# recompiles only the sources that changed.

ARG CUDA_VERSION=13.0.0
ARG UBUNTU=ubuntu24.04

# ---- 1. engine ---------------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 CCACHE_DIR=/ccache CCACHE_MAXSIZE=10G
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache curl unzip patch python3 python3-venv libgomp1 libatomic1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /tools
WORKDIR /src
COPY vendor/requirements-build.txt /src/requirements-build.txt
RUN /tools/bin/pip install --no-cache-dir -r requirements-build.txt

# llama.cpp at upstream's pinned commit: ggml for the engine, mtmd for the picture reader, gguf-py for the prep tools
COPY vendor/third_party/ggml/VERSION.txt /tmp/llama-pin
RUN set -e; commit=$(awk 'NR==1{print $1}' /tmp/llama-pin); \
    curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/${commit}.zip"; \
    unzip -q /tmp/llama.zip -d /tmp/llama; mkdir -p /opt; mv /tmp/llama/llama.cpp-${commit} /opt/llama.cpp; \
    rm -rf /tmp/llama /tmp/llama.zip

COPY vendor/CMakeLists.txt vendor/UPSTREAM /src/
COPY vendor/cmake/       /src/cmake/
COPY vendor/include/     /src/include/
COPY vendor/src/         /src/src/
COPY vendor/third_party/ /src/third_party/
COPY vendor/bench/       /src/bench/
COPY vendor/tests/       /src/tests/
COPY vendor/tools/vision/  /src/tools/vision/
COPY vendor/tools/trellis/ /src/tools/trellis/
# local fixes from patches/, only when asked: PATCHES=all, or some by number (0001,0002). Default: none, upstream's code.
COPY patches/ /patches/
COPY scripts/apply-patches.sh /usr/local/bin/apply-patches
ARG PATCHES=""
RUN apply-patches /patches /src "$PATCHES"
ARG CUDA_ARCHITECTURES=86
ARG BUILD_JOBS=4
ARG BUILD_VISION=1
RUN --mount=type=cache,target=/ccache set -e; export PATH=/tools/bin:$PATH; \
    real=""; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}; \
    last=$(echo "$CUDA_ARCHITECTURES" | tr ';,' '  ' | awk '{print $NF}'); \
    sm60=OFF; for a in $(echo "$CUDA_ARCHITECTURES" | tr ';,' '  '); do [ "$a" -lt 75 ] && sm60=ON || true; done; \
    cc="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"; \
    cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF $cc \
        -DSTRATA_EXPERIMENTAL_SM60=$sm60 \
        -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR=/opt/llama.cpp; \
    cmake --build build --target strata strata_kq -j "$BUILD_JOBS"; \
    if [ "$BUILD_VISION" = "1" ]; then \
      cmake -G Ninja -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/opt/llama.cpp $cc \
          -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"; \
      cmake --build build-vision --target strata-vision -j "$BUILD_JOBS"; \
    fi; \
    ccache --show-stats --verbose | grep -E 'Hits|Misses' | head -4 || true
RUN set -e; mkdir -p /out/engine; \
    cp build/strata /out/engine/; \
    [ -f build/libstrata_kq.so ] && cp build/libstrata_kq.so /out/engine/ || true; \
    [ -f build-vision/bin/strata-vision ] && cp build-vision/bin/strata-vision /out/engine/ || true; \
    . ./UPSTREAM; cuda=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1); \
    ENGINE_VERSION="$engine_version" TAG="$tag" COMMIT="$commit" CUDA="$cuda" ARCHS="$CUDA_ARCHITECTURES" \
    APPLIED="$(cat PATCHES)" python3 -c '\
import json, os, re; \
archs = [a for a in re.split(r"[;,\s]+", os.environ["ARCHS"]) if a]; \
print(json.dumps({"version": os.environ["ENGINE_VERSION"], "source": "strata-slim " + os.environ["TAG"], \
                  "commit": os.environ["COMMIT"], "cuda": os.environ["CUDA"], "portable": True, \
                  "archs": archs, "vision_archs": archs, "ptx": True, \
                  "patches": [p for p in os.environ["APPLIED"].split(",") if p]}))' > /out/engine/BUILD.json; \
    cat /out/engine/BUILD.json

# ---- 2. runtime --------------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/venv/bin:$PATH STRATA_GGUF_PY=/opt/strata/third_party/llama.cpp/gguf-py
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv libgomp1 libatomic1 tini curl \
    && rm -rf /var/lib/apt/lists/*
COPY vendor/requirements.txt /tmp/requirements.txt
COPY requirements.txt /tmp/requirements-slim.txt
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-slim.txt

WORKDIR /opt/strata
COPY --from=build /out/engine/ /opt/strata/engine/
COPY --from=build /opt/llama.cpp/gguf-py/ /opt/strata/third_party/llama.cpp/gguf-py/
COPY vendor/serve/          /opt/strata/serve/
COPY vendor/strata_control/ /opt/strata/strata_control/
COPY vendor/strata_cli/     /opt/strata/strata_cli/
COPY vendor/data/           /opt/strata/data/
COPY vendor/tools/*.py      /opt/strata/tools/
COPY vendor/UPSTREAM vendor/.config-keys vendor/LICENSE /opt/strata/
COPY launch.py prepare.py hf.py common.py settings.py /opt/strata/

# build-time checks (no GPU here): every library the engine links resolves except the driver's own (libcuda, from the
# host at run time); and the server starts
RUN set -e; for b in /opt/strata/engine/strata /opt/strata/engine/strata-vision; do \
      [ -f "$b" ] || continue; \
      miss=$(LD_LIBRARY_PATH=/opt/strata/engine ldd "$b" | grep 'not found' | grep -v -E 'libcuda\.so|libnvidia-ml' || true); \
      [ -z "$miss" ] || { echo "$b: missing libraries:"; echo "$miss"; exit 1; }; done; \
    cd /opt/strata && python -m serve.server --help >/dev/null && echo "serve.server starts"

ENV LD_LIBRARY_PATH=/opt/strata/engine
VOLUME ["/data"]
EXPOSE 8080
# start-period: a first start may download the model (100+ GB) and build the pack and MTP layer before it listens
HEALTHCHECK --interval=30s --timeout=5s --start-period=6h --retries=3 \
    CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/opt/strata/launch.py"]
CMD ["serve"]
