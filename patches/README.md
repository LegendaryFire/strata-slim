# patches

Small fixes we carry on top of upstream Strata. They're off by default, so a normal build is upstream's code as
released. Turn them on with the `PATCHES` build arg:

```sh
docker build -t strata-slim:0.2.13-patched --build-arg PATCHES=all .
```

`PATCHES` takes `all`, or a list of patch numbers like `0001,0002`.

To add one, make the change in `vendor/`, commit it, save it with
`git format-patch -1 <commit> --stdout > patches/NNNN-name.patch`, then revert the commit.

When upstream updates, `scripts/extract.py` checks that each patch still applies. If upstream already has the fix,
delete the patch.

| Patch | What it does |
|---|---|
| `0001-hc-q8-skip-bf16-multi-gpu.patch` | When the model is split across GPUs, skip loading the BF16 copy of the HC weights, the same as a single GPU already does. |
| `0002-dense-arena.patch` | Pack the dense projection matrices into one VMM range per GPU instead of one `cudaMalloc` each, so they stop losing VRAM to 2 MiB rounding; the expert cache gets it. Same output. CUDA only; `STRATA_DENSE_ARENA=0` turns it off. |
| `0003-verify-no-pcie-staging.patch` | With `--pcie-frac 0`, the verify window keeps no PCIe staging slots (about 61 MiB per GPU on UD-Q4_K_XL), which were never used; the expert cache gets it. `STRATA_PCIE_STAGING_KEEP=1` keeps them. |
