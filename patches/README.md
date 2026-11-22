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
| `0006-inkling2-page-select.patch` | Opt-in page selection for Inkling 2's tiered decode (`--kv-select N`, `--kv-page P`): a decode reads only the RAM-part pages with the best bound on attention instead of all of them. Not exact. Needs `0004`; leave it out to drop the feature. Doesn't combine with `0008`. |
| `0007-inkling2-draft.patch` | Inkling 2 drafts its next tokens from the conversation itself (prompt lookup) and checks up to 8 in one forward; the output is unchanged, and the draft length follows measured acceptance and cost (`--lookup-max-t 0` turns it off). Pays most on long contexts and on text that copies from the context. Needs `0004`. |
| `0008-inkling2-vram-pool.patch` | Inkling 2's experts and its cache windows share VRAM (`--kv-resident auto`): every expert and cache cell is also in RAM, a session's window grows as it does and the least-used experts give it their room (computed on CPU threads, or copied in for prompt chunks), and the room goes back to experts when a session starts over. Same output as with every expert in VRAM. Needs `0004` and `0007`; doesn't combine with `0006`. |
| `0009-inkling2-prompt-fa2.patch` | Inkling 2 reads prompts faster: one-pass, FlashAttention-2 style tensor-core attention (the relative bias only near the diagonal, the next key tile copied in while this one runs), prompt chunks on both cards at once (`--prefill-pipe`, on with two cards), and a prompt's cache window kept out of the experts' VRAM. About 1.7x at 150K, 2.5x at 49K. Needs `0004`, `0007`, `0008`. |
| `0010-inkling2-lanes-share.patch` | Inkling 2's two conversations stay out of each other's way: decoding keeps GPU priority over the other's prompt reading (and the pool's adapt round never makes it wait), and a conversation sent again while its answer is still being written starts from the busy lane's state instead of being read again. Conversations are parked in RAM (`--conversation-cache-mib` / `-slots` / `-min-tokens`), and checkpoints every 8K-16K tokens let a request that shares a long start with any lane or parked conversation continue from there. Reports its experts in VRAM to the Monitor. Needs `0004`, `0007`-`0009`. |
| `0011-inkling2-decode-speed.patch` | Inkling 2 decodes faster: tensor-core decode attention that reads the cache once for all rows (a drafted verify no longer re-reads it per token), each full-attention window split across the two cards and read by both at once, an NVFP4 down kernel for Inkling's shape, fused small steps, an expert pool that ranks by routings and never waits on the CPU for a layer with no expert in RAM. 180K: 28.4 -> 35.6 tok/s, 1K: 43.6 -> 50.0. Needs `0004`, `0007`-`0010`. |
| `0012-inkling2-dspark.patch` | Inkling 2 drafts with its own DSpark block drafter (`--dspark FILE`, Unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf`): the target's hidden states at four visits go into the drafter's cache as tokens are committed, a block is drafted in one pass and checked by the same exact verify step as lookup drafting (which still goes first when the conversation repeats itself). Draft length adapts to the measured acceptance and times (`--dspark-n`, at most 3 by default). 180K: 35.6 -> 57.3 tok/s. Needs `0004`, `0007`-`0011`. |
| `0013-inkling2-vram-trim.patch` | Inkling 2 decode takes less VRAM (attention partials sized for what decode uses, a RAM part read in larger partials, the drafter's features in FP16: about 100 more experts in VRAM at 2x512K) DSpark's adaptive length keeps measuring longer drafts (180K: 49.5 -> 56.1 tok/s greedy, 43.4 at temperature 1), and a draft reads the output head once for all its rows, scoring only each row's top candidates (+5-9% where it shows). Needs `0004`, `0007`-`0012`. |
| `0014-inkling2-expert-gemm.patch` | Inkling 2 reads prompts faster: its NVFP4 expert GEMM is warp-specialized (half the warps copy rows and convert weights while the other half multiply, the raw weights copied four steps at a time), and a hot expert's rows spread over many blocks instead of one. Not bit-exact with `0013` (the tensor cores sum each 16 in another order; rounding-level). 7K: 1679 -> 1818 tok/s, 60K: 1739 -> 1865, 180K: 1428 -> 1508. Needs `0004`, `0007`-`0013`. |
| `0015-inkling2-split-by-pipeline.patch` | With two cards, Inkling 2's layers are split for its prompt pipeline instead of by bytes (Inkling-2-Mini: 0-5 and 12-18 on GPU 0, 6-11 and 19-23 on GPU 1, so the looped core no longer leaves one card waiting), the head and DSpark drafter go wherever they even the free VRAM, and each card's layers start with the experts that fit there. Same output. 7K: 1825 -> 1971 tok/s, 60K: 1867 -> 2222, 180K: 1509 -> 1782. Needs `0004`, `0007`-`0014`. |
| `0016-inkling2-decode-kernels.patch` | Inkling 2 decodes faster: a visit's q, rel, gate, k and v projections run in one launch (the small ones no longer underfill the cards), and the looped layers' control projections split over K. Rounding-level change. Decode 1K: 48.9 -> 51.6 tok/s, 60K: 41.8 -> 44.1, 180K: 30.3 -> 31.3. Needs `0004`, `0007`-`0015`. |
| `0017-inkling2-tensor-core-bias.patch` | Inkling 2's prompt attention takes its relative bias from the tensor cores (FP32 accuracy, with split FP16 operands) instead of a 16-term dot product per score, which was 65% of the sliding-window layers' attention time. Prompt 7K: 2038 -> 2117 tok/s, 60K: 2208 -> 2383, 180K: 1780 -> 1900. Needs `0004`, `0007`-`0016`. |
