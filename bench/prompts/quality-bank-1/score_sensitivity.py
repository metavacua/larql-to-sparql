#!/usr/bin/env python3
"""Score the frozen candidates with SENSITIVITY-1A alone, then compare
that ranking against the Q-BANK verdicts.

    python3 score_sensitivity.py <sens.json> <sweep.json>

The candidate definitions below are the protections each Q-BANK arm was
compiled with. Nothing here reads the verdict before ranking.
"""
import json, sys
import numpy as np

# (projection or None, (lo,hi) or None) rules; a tensor is protected if any matches.
CANDIDATES = {
    "attn-protected":  [("q_proj",None),("k_proj",None),("v_proj",None),("o_proj",None)],
    "ffn-protected":   [("gate_proj",None),("up_proj",None),("down_proj",None)],
    "v-protected":     [("v_proj",None)],
    "o-protected":     [("o_proj",None)],
    "down-protected":  [("down_proj",None)],
    "k-protected":     [("k_proj",None)],
    "early10":         [(None,(0,9))],
    "late10":          [(None,(30,39))],
    "late5-ffn":       [("gate_proj",(35,39)),("up_proj",(35,39)),("down_proj",(35,39))],
    "late10-ffn":      [("gate_proj",(30,39)),("up_proj",(30,39)),("down_proj",(30,39))],
    "late15-ffn":      [("gate_proj",(25,39)),("up_proj",(25,39)),("down_proj",(25,39))],
    "late10-ffn-o":    [("gate_proj",(30,39)),("up_proj",(30,39)),("down_proj",(30,39)),("o_proj",None)],
    "late10-ffn-v":    [("gate_proj",(30,39)),("up_proj",(30,39)),("down_proj",(30,39)),("v_proj",None)],
}

def proj_of(t):
    p = t.split(".")
    return p[-2] if len(p) >= 2 else t

def layer_of(t):
    try: return int(t.split(".")[0])
    except ValueError: return None

def protected(rules, tensor):
    for proj, rng in rules:
        if proj is not None and proj_of(tensor) != proj: continue
        if rng is not None:
            l = layer_of(tensor)
            if l is None or not (rng[0] <= l <= rng[1]): continue
        return True
    return False

def main(sens_path, sweep_path):
    sens = json.load(open(sens_path))
    sweep = {c["label"]: c for c in json.load(open(sweep_path))["candidates"]}
    base = sweep["R0-recheck"]

    rows = []
    for name, rules in CANDIDATES.items():
        sel = [t for t in sens if protected(rules, t["tensor"])]
        if not sel or name not in sweep:
            continue
        extra = sum(t["source_bytes"] - t["compiled_bytes"] for t in sel) / 2**20
        # 1A: relative error removed, and the same weighted by weight energy
        # (absolute error removed) — both stated before any comparison.
        rel = sum(t["rel_error"] for t in sel)
        abs_ = sum(t["rel_error"] * t["energy"] for t in sel)
        v = sweep[name]
        rows.append({
            "name": name, "tensors": len(sel), "extra_mib": extra,
            "s_rel": rel, "s_rel_per_mib": rel / extra if extra else 0,
            "s_abs": abs_, "s_abs_per_mib": abs_ / extra if extra else 0,
            "kl_p99_rec_per_mib": (base["kl"]["p99"] - v["kl"]["p99"]) / extra if extra else 0,
            "hi_rec_per_mib": (base["flips_high_margin"] - v["flips_high_margin"]) / extra if extra else 0,
        })

    def rank(key, reverse=True):
        return [r["name"] for r in sorted(rows, key=lambda r: r[key], reverse=reverse)]

    def spearman(a, b):
        ra = {n: i for i, n in enumerate(a)}
        rb = {n: i for i, n in enumerate(b)}
        x = np.array([ra[n] for n in ra]); y = np.array([rb[n] for n in ra])
        return float(np.corrcoef(x, y)[0, 1])

    truth = rank("kl_p99_rec_per_mib")
    print("SENSITIVITY-1A vs Q-BANK — 13 candidates\n")
    print(f"  {'candidate':<16}{'+MiB':>7}{'1A rel/MiB':>12}{'1A abs/MiB':>12}"
          f"{'p99rec/MiB':>12}{'hi/MiB':>9}")
    print("  " + "-"*68)
    for r in sorted(rows, key=lambda r: -r["kl_p99_rec_per_mib"]):
        print(f"  {r['name']:<16}{r['extra_mib']:>7.0f}{r['s_rel_per_mib']:>12.5f}"
              f"{r['s_abs_per_mib']:>12.3e}{r['kl_p99_rec_per_mib']:>12.5f}{r['hi_rec_per_mib']:>9.3f}")

    print("\n  rankings (best first)")
    for key, label in [("s_rel_per_mib","1A rel/MiB"),("s_abs_per_mib","1A abs/MiB"),
                       ("kl_p99_rec_per_mib","Q-BANK p99/MiB")]:
        print(f"    {label:<16} {' > '.join(rank(key)[:6])}")

    print("\n  Spearman vs Q-BANK p99/MiB")
    for key, label in [("s_rel_per_mib","1A rel/MiB"),("s_abs_per_mib","1A abs/MiB")]:
        print(f"    {label:<16} {spearman(rank(key), truth):+.3f}")

    print("\n  THE BAR (both halves required)")
    NEG = {"v-protected","k-protected","down-protected"}
    for key, label in [("s_rel_per_mib","1A rel/MiB"),("s_abs_per_mib","1A abs/MiB")]:
        r = rank(key)
        top = r[0]
        half1 = "ffn" in top or top.startswith("late")
        neg_pos = sorted(r.index(n) for n in NEG if n in r)
        half2 = min(neg_pos) >= len(r) - len(NEG) - 2   # negatives clustered at the bottom
        print(f"    {label}")
        print(f"      1. identifies late-FFN highest-return : {'PASS' if half1 else 'FAIL'}  (top = {top})")
        print(f"      2. rejects v/k/down as low-value      : {'PASS' if half2 else 'FAIL'}"
              f"  (ranks {[r.index(n)+1 for n in NEG if n in r]} of {len(r)})")
        print(f"      => {'PASS' if (half1 and half2) else 'FAIL'}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
