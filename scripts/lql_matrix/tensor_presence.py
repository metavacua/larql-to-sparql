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


def load_listing(path):
    """Load a listing.json, or raise.

    This used to `except Exception: return {}`, which made an unreadable or
    malformed listing indistinguishable from a vindex that genuinely contains
    no files — the caller then reported "0 files" as a finding about larql when
    it was a fact about the harness. A missing input is the caller's problem to
    surface, not this function's to paper over."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(d).__name__}")
    return d


_LEG_RE = re.compile(r"manifest-(.+?)/listing\.json$")


def collect(results_glob):
    rows = {}
    seen = set()
    pats = sorted(glob.glob(results_glob)) + sorted(
        glob.glob("**/" + results_glob, recursive=True))
    for p in pats:
        if p in seen:
            continue
        seen.add(p)
        m = _LEG_RE.search(p)
        if m:
            try:
                rows[m.group(1)] = presence(load_listing(p))
            except Exception as e:
                # Recorded per leg, never blanked and never dropped. A leg whose
                # listing cannot be read is a DIFFERENT fact from a leg with no
                # files, and one unreadable listing must not cost the other legs
                # their data.
                rows[m.group(1)] = {"error": f"{type(e).__name__}: {e}", "path": p}
    return rows


def main():
    args = sys.argv[1:]
    results_glob = args[0] if args else "artifacts/results-*/manifest-*/listing.json"
    out_json = args[1] if len(args) > 1 else "presence.json"
    rows = collect(results_glob)
    Path(out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"tensor-presence: {len(rows)} legs")


if __name__ == "__main__":
    main()
