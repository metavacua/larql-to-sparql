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
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


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
            rows[m.group(1)] = presence(load_listing(p))
    return rows


def resolve_q8(presence_by_leg, panic_by_leg):
    rows, counter = [], []
    for leg in sorted(set(presence_by_leg) & set(panic_by_leg)):
        risk = bool(presence_by_leg[leg]["ffn_unwrap_risk"])
        pan = bool(panic_by_leg[leg])
        agree = (risk == pan)
        rows.append({"leg": leg, "ffn_unwrap_risk": risk, "panic": pan, "agree": agree})
        if not agree:
            counter.append({"leg": leg, "ffn_unwrap_risk": risk, "panic": pan})
    return {"rows": rows, "biconditional_holds": not counter, "counterexamples": counter}


def render_q8(presence_by_leg, violations, native_only=True):
    """Join presence with observed crash outcomes → Q8 resolution.
    violations: the conformance.json "violations" list, or None if the artifact
    was unavailable. Returns a dict with keys: status ∈ {"holds","refuted",
    "inconclusive"}, biconditional_holds (bool|None), rows, counterexamples,
    n_native (int), n_risk (int), reason (str)."""
    def _incon(reason):
        return {"status": "inconclusive", "biconditional_holds": None, "rows": [],
                "counterexamples": [], "n_native": 0, "n_risk": 0, "reason": reason}
    if violations is None:
        return _incon("conformance data unavailable")
    # panic := a no-crash violation on a CORPUS cell (produce-time crashes excluded)
    panicked = {v.get("leg") for v in violations
                if v.get("invariant") == "no-crash" and v.get("cell") != "produce"}
    legs = {k: p for k, p in presence_by_leg.items()
            if (not native_only) or (".native." in k)}
    if not legs:
        return _incon("no native legs with a produced listing")
    n_risk = sum(1 for p in legs.values() if p.get("ffn_unwrap_risk"))
    n_panic = sum(1 for k in legs if k in panicked)
    if n_risk == 0 and n_panic == 0:
        return _incon("no risk legs and no panics observed — nothing to test")
    panic = {k: (k in panicked) for k in legs}
    res = resolve_q8(legs, panic)
    res["status"] = "holds" if res["biconditional_holds"] else "refuted"
    res["n_native"] = len(legs)
    res["n_risk"] = n_risk
    res["reason"] = ""
    return res


def q8_markdown(result):
    """Render a render_q8() result as the q8.md report."""
    head = {"holds": "biconditional holds: **True**",
            "refuted": "biconditional holds: **False**",
            "inconclusive": "**inconclusive**"}[result["status"]]
    L = ["# Q8 — panic ⇔ (has_attn ∧ ¬has_ffn)?", "",
         f"{head} · counterexamples: {len(result['counterexamples'])} "
         f"· native legs: {result.get('n_native', 0)} · risk legs: {result.get('n_risk', 0)}"]
    if result.get("reason"):
        L.append(f"reason: {result['reason']}")
    L += ["", "| leg | ffn_unwrap_risk | panic | agree |", "|---|---|---|---|"]
    for r in result["rows"]:
        mark = "✅" if r["agree"] else "❌"
        L.append(f"| `{r['leg']}` | {r['ffn_unwrap_risk']} | {r['panic']} | {mark} |")
    return "\n".join(L) + "\n"


def main():
    args = sys.argv[1:]
    results_glob = args[0] if args else "artifacts/results-*/manifest-*/listing.json"
    out_json = args[1] if len(args) > 1 else "presence.json"
    rows = collect(results_glob)
    Path(out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"tensor-presence: {len(rows)} legs, "
          f"{sum(1 for r in rows.values() if r['ffn_unwrap_risk'])} at ffn_unwrap_risk")


if __name__ == "__main__":
    main()
