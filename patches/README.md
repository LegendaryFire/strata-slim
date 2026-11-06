# patches

Local fixes on top of upstream Strata. `scripts/extract.sh` applies every `*.patch` here, in name order, after it
replaces `vendor/`, and lists them in `vendor/UPSTREAM`. Paths are relative to the project root (`vendor/...`).

- A patch upstream already has is skipped with a note: delete it.
- A patch that no longer applies stops the extract: rebase it on the new `vendor/`.

New patch: commit the change to `vendor/`, then `git format-patch -1 <commit> --stdout > patches/NNNN-name.patch`.

| Patch | What |
|---|---|
| `0001-hc-q8-honor-stage-trim.patch` | `STRATA_HC_Q8` + `STRATA_STAGE_TRIM`: each layer-split card loads only its own layers' Q8_0 hyper-connection copies, not all 48 (~0.3 GiB per card on 2x RTX 3060, +357 resident experts) |
| `0002-hc-q8-skip-bf16-multi-gpu.patch` | `STRATA_HC_Q8` on a layer split: don't load the BF16 copy of the HC down/up weights, as single GPU already does (~600 MiB per card) |
