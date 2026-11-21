# patches

Small fixes we carry on top of upstream Strata. They're off by default, so a normal build is upstream's code as
released. Turn them on with the `PATCHES` build arg:

```sh
docker build -t strata-slim:0.2.17-patched --build-arg PATCHES=all .
```

`PATCHES` takes `all`, or a list of patch numbers like `0001,0002`.

To add one, make the change in `vendor/`, commit it, save it with
`git format-patch -1 <commit> --stdout > patches/NNNN-name.patch`, then revert the commit. A patch changes either the
engine or the server's files (`serve/`, `strata_control/`, `strata_cli/`, `data/`, `tools/*.py`), not both: the build
applies each where those files are.

When upstream updates, `scripts/extract.py` checks that each patch still applies. If upstream already has the fix,
delete the patch.

| Patch | What it does |
|---|---|
| `0001-hc-q8-skip-bf16-multi-gpu.patch` | When the model is split across GPUs, skip loading the BF16 copy of the HC weights, the same as a single GPU already does. |
| `0002-dense-arena.patch` | Pack the dense projection matrices into one VMM range per GPU instead of one `cudaMalloc` each, so they stop losing VRAM to 2 MiB rounding; the expert cache gets it. Same output. CUDA only; `STRATA_DENSE_ARENA=0` turns it off. |
| `0003-verify-no-pcie-staging.patch` | With `--pcie-frac 0`, the verify window keeps no PCIe staging slots (about 61 MiB per GPU on UD-Q4_K_XL), which were never used; the expert cache gets it. `STRATA_PCIE_STAGING_KEEP=1` keeps them. |
| `0004-inkling2-engine.patch` | Runs Inkling-2-Mini (`unsloth/Inkling-2-Mini-GGUF`, the looped `inkling2` GGUFs of llama.cpp PR #32695) on an engine path of its own, split by layers over the GPUs. CUDA only. Needs `0005`. |
| `0005-inkling2-serve.patch` | The server side of Inkling 2: its tokenizer, chat template, reply format (thinking, text, tool calls) and end-of-turn token. Needs `0004`. |
| `0006-inkling2-page-select.patch` | Opt-in page selection for Inkling 2's tiered decode (`--kv-select N`, `--kv-page P`): a decode reads only the RAM-part pages with the best bound on attention instead of all of them. Not exact. Needs `0004`; leave it out to drop the feature. |
| `0007-inkling2-draft.patch` | Inkling 2 drafts its next tokens from the conversation itself (prompt lookup) and checks up to 8 in one forward; the output is unchanged, and the draft length follows measured acceptance and cost (`--lookup-max-t 0` turns it off). Pays most on long contexts and on text that copies from the context. Needs `0004`. |
