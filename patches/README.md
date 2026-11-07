# patches

Local fixes on top of upstream Strata. They are **optional and off by default**: `vendor/` is always upstream's code
exactly, and a build applies patches only when you ask for them with the `PATCHES` build arg.

| `PATCHES` | Build |
|---|---|
| unset, empty or `none` (default) | upstream's code as released |
| `all` | every `*.patch` here, in name order |
| `0001,0002` | just those, by number or file name |

```sh
docker build -t strata-slim:0.2.11 .                                         # stock
docker build -t strata-slim:0.2.11-patched --build-arg PATCHES=all .         # with the fixes
```

`compose.yaml` builds the stock image; build a patched one with `docker build` as above. The engine's `BUILD.json` lists the patches a build has (`"patches": [...]`).

`scripts/apply-patches.sh` applies them in the build stage. `scripts/extract.sh` checks each against a new release:

- A patch upstream already has gets a note: delete it.
- A patch that no longer applies stops the extract: rebase it on the new `vendor/`.

New patch: make the change in `vendor/`, commit it, `git format-patch -1 <commit> --stdout > patches/NNNN-name.patch`,
then revert the commit so `vendor/` stays upstream's.

| Patch | What |
|---|---|
| `0001-hc-q8-honor-stage-trim.patch` | `STRATA_HC_Q8` + `STRATA_STAGE_TRIM`: each layer-split card loads only its own layers' Q8_0 hyper-connection copies, not all 48 (~0.3 GiB per card on 2x RTX 3060, +357 resident experts) |
| `0002-hc-q8-skip-bf16-multi-gpu.patch` | `STRATA_HC_Q8` on a layer split: don't load the BF16 copy of the HC down/up weights, as single GPU already does (~600 MiB per card) |
