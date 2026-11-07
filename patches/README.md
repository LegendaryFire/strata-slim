# patches

Small fixes we carry on top of upstream Strata. They're off by default, so a normal build is upstream's code as
released. Turn them on with the `PATCHES` build arg:

```sh
docker build -t strata-slim:0.2.11-patched --build-arg PATCHES=all .
```

`PATCHES` takes `all`, or a list of patch numbers like `0001,0002`.

To add one, make the change in `vendor/`, commit it, save it with
`git format-patch -1 <commit> --stdout > patches/NNNN-name.patch`, then revert the commit.

When upstream updates, `scripts/extract.py` checks that each patch still applies. If upstream already has the fix,
delete the patch.

| Patch | What it does |
|---|---|
| `0001-hc-q8-honor-stage-trim.patch` | When the model is split across GPUs, each GPU only loads the HC Q8 weights for its own layers instead of all of them. |
| `0002-hc-q8-skip-bf16-multi-gpu.patch` | When the model is split across GPUs, skip loading the BF16 copy of the HC weights, the same as a single GPU already does. |
