"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the end tokens part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations


from serve.server_parts.base import IM_END


def end_token_ids(tokenizer) -> set[int]:
    """The ids that end a reply for a tokenizer whose family was not named.

    Qwen's <|im_end|> and <|endoftext|> end a reply only where each is ONE token.  A tokenizer without them (one that
    splits "<|im_end|>" into '<', '|', 'im', '_end', '|', '>') would make every piece of them a stop id: the reply ended in front
    of the first "|", ">", "end", "of", "text" or "im" it wrote, so "Nerazim" came out as "Neraz" (#4576, and a table's first
    "|" in #4137).  Those ids are left out, and the end token the tokenizer names (special_ids "eos") ends the reply instead."""
    ids: set[int] = set()
    for marker in (IM_END, "<|endoftext|>"):
        got = tokenizer.encode(marker, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    specials = getattr(tokenizer, "special_ids", None) or {}
    if not ids:
        eos = specials.get("eos")
        if eos is not None:
            ids.add(int(eos))
    # Inkling 2's end of turn was not <|im_end|>, so now the GGUF's eos / eot ids end a reply too
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if getattr(tokenizer, "pre", None) == "inkling2" and specials.get(key) is not None:
            ids.add(int(specials[key]))
    return ids
