"""Tensor-presence witness for the σ-expansion (closes Q8). A produced vindex's
weight-file listing → a presence vector over weight classes. The presence of
attention vs FFN weight files at a level is the latent variable explaining why
some partial/hollow vindexes panic on INFER (guard passes, FFN absent) while
others refuse gracefully (no attention weights → guard refuses)."""
import glob
import json
import os
import re
import sys
from pathlib import Path

# vindex weight file → the class it witnesses
_CLASS = {
    "attn_weights.bin": "attn",
    "up_weights.bin": "ffn_up",
    "down_weights.bin": "ffn_down",
    "gate_vectors.bin": "gate",
    "embeddings.bin": "embed",
    "norms.bin": "norm",
    "down_meta.bin": "down_meta",
    "interleaved_kquant.bin": "kquant",
    "lm_head.bin": "lm_head",
}


def presence(listing):
    cls = {}
    for fn, sz in (listing or {}).items():
        c = _CLASS.get(fn)
        if c and isinstance(sz, int) and sz > 0:
            cls[c] = sz
    has_attn = "attn" in cls
    has_ffn = ("ffn_up" in cls) and ("ffn_down" in cls)
    return {
        "n_files": len(listing or {}),
        "classes": cls,
        "has_attn": has_attn,
        "has_ffn": has_ffn,
        "has_gate": "gate" in cls,
        "ffn_unwrap_risk": has_attn and not has_ffn,
    }


def listing_of(vindex_dir):
    try:
        return {f: os.path.getsize(os.path.join(vindex_dir, f))
                for f in os.listdir(vindex_dir)}
    except OSError:
        return {}
