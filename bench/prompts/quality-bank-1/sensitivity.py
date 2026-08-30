#!/usr/bin/env python3
"""Rank precision-map candidates by fidelity recovered per extra byte.

    python3 sensitivity.py <bank-dir> <baseline-label> <candidate-label>...

Every candidate is compared against the same frozen BF16 reference and
against the R0 baseline, so the question answered is not "is this good"
but "what did these extra bytes buy".
"""
import json, os, sys
import numpy as np

def load(d, label):
    p = os.path.join(d, f"compare-{label}.json")
    return json.load(open(p))

def stats(rows):
    kl = np.array([r["kl"] for r in rows])
    dn = np.array([r["dnll"] for r in rows if r["dnll"] is not None])
    flip = np.array([r["flip"] for r in rows])
    marg = np.array([r["margin"] for r in rows])
    hi = int((flip & (marg >= 0.01)).sum())
    return {
        "kl_mean": kl.mean(), "kl_p95": np.percentile(kl, 95),
        "kl_p99": np.percentile(kl, 99),
        "dnll_mean": dn.mean(), "flips": int(flip.sum()), "hi_flips": hi,
    }

def by_category(rows):
    out = {}
    for c in sorted({r["category"] for r in rows}):
        sel = [r for r in rows if r["category"] == c]
        k = np.array([r["kl"] for r in sel])
        out[c] = (k.mean(), int(sum(r["flip"] for r in sel)))
    return out

def main(d, base_label, labels):
    base = load(d, base_label)
    b = stats(base["rows"])
    bbytes = base["container"].get("compiled_bytes")
    print(f"\nQ-BANK-1 sensitivity — baseline {base_label}")
    print(f"  {base['positions'] if 'positions' in base else len(base['rows']):,} positions")
    print(f"  KL mean {b['kl_mean']:.4f}  p95 {b['kl_p95']:.4f}  p99 {b['kl_p99']:.4f}"
          f"   flips {b['flips']} (high-margin {b['hi_flips']})")
    print()
    # Several objectives on purpose. R0's damage has a fat tail — median
    # KL 0.061 against p99 4.62 — so a policy can recover many flips while
    # leaving the ugly tail untouched. The candidate that collapses the
    # tail may be worth more than the one that optimises the average.
    hdr = (f"  {'candidate':<18}{'+MiB':>8}{'KLmean':>9}{'KLp95':>9}{'KLp99':>9}"
           f"{'flips':>7}{'hi':>5}{'dNLL':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows_out = []
    for lab in labels:
        try:
            c = load(d, lab)
        except FileNotFoundError:
            print(f"  {lab:<18}  (not run yet)")
            continue
        s = stats(c["rows"])
        extra = max(0, c.get("payload_bytes", 0) - base.get("payload_bytes", 0)) / 2**20
        rec = b["hi_flips"] - s["hi_flips"]
        per = (rec / extra) if extra > 0 else float("nan")
        rows_out.append((lab, extra, s, rec, per))
        print(f"  {lab:<18}{extra:>8.0f}{s['kl_mean']:>9.4f}{s['kl_p95']:>9.4f}"
              f"{s['kl_p99']:>9.4f}{s['flips']:>7}{s['hi_flips']:>5}{s['dnll_mean']:>9.4f}")
    if rows_out:
        print("\n  recovered per extra MiB (higher is better; a single ranking")
        print("  would hide that these objectives disagree)")
        print(f"    {'candidate':<18}{'hi-flip':>9}{'all flip':>10}{'KL p95':>10}{'KL p99':>10}{'dNLL':>10}")
        print("    " + "-" * 67)
        def per(v, e):
            return v / e if e > 0 else float("nan")
        table = []
        for lab, extra, st, rec, _ in rows_out:
            table.append((
                lab, extra,
                per(rec, extra),
                per(b["flips"] - st["flips"], extra),
                per(b["kl_p95"] - st["kl_p95"], extra),
                per(b["kl_p99"] - st["kl_p99"], extra),
                per(b["dnll_mean"] - st["dnll_mean"], extra),
            ))
        for row in sorted(table, key=lambda r: -(r[2] if r[2] == r[2] else -1e9)):
            print(f"    {row[0]:<18}{row[2]:>9.3f}{row[3]:>10.3f}"
                  f"{row[4]:>10.5f}{row[5]:>10.5f}{row[6]:>10.5f}")
        print("\n  best by each objective:")
        for i, name in [(2, "high-margin flips"), (3, "all flips"),
                        (4, "KL p95 (tail)"), (5, "KL p99 (tail)"), (6, "mean dNLL")]:
            valid = [r for r in table if r[i] == r[i]]
            if valid:
                w = max(valid, key=lambda r: r[i])
                print(f"    {name:<20} {w[0]}  ({w[i]:.4f} per MiB)")
    print("\n  per category (KL mean / flips)")
    cats = by_category(base["rows"])
    names = list(cats)
    print("    " + "category".ljust(13) + base_label.rjust(16)
          + "".join(l.rjust(16) for l in labels if os.path.exists(os.path.join(d, f"compare-{l}.json"))))
    for c in names:
        line = f"    {c:<13}" + f"{cats[c][0]:>9.4f}/{cats[c][1]:<6}"
        for lab in labels:
            p = os.path.join(d, f"compare-{lab}.json")
            if not os.path.exists(p):
                continue
            cc = by_category(json.load(open(p))["rows"])[c]
            line += f"{cc[0]:>9.4f}/{cc[1]:<6}"
        print(line)
    print("\n  A candidate that improves aggregate KL by helping one category")
    print("  while another gets worse is not obviously preferable — which is")
    print("  why the category split stays in the report.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
