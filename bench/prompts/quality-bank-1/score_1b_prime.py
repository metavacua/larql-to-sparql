#!/usr/bin/env python3
"""SENSITIVITY-1B' — aggregate per-tensor consequence and apply the frozen bar.

    python3 score_1b_prime.py <consequence.json> <sweep.json> [--expect-token-digest HEX]

Rust owns numerics and execution semantics; this owns aggregation and
judgment. It reads the per-tensor records `vindex3 consequence` emitted and
does exactly what the pre-registration says, once:

    candidate_num = sum of tensor num over the candidate's tensors
    score         = candidate_num / extra physical MiB

No alternative normalisation. No coefficient fitting. No exploratory pass
before the bar — the aggregation was fixed before any number existed, so
there is nothing to decide after seeing them.

Read `SENSITIVITY-1B-PRIME.md` before changing anything here. This is a
one-shot rung: the point of the ladder is that a failure escalates to 1C
rather than getting tuned into a pass.
"""
import json
import sys

import candidates as C

MIB = 2 ** 20


def load(path):
    return json.load(open(path))


def check_candidate_coverage(records):
    """Every (projection, layer) a candidate's rules name must carry a
    measurement.

    Region congruence proved the screen's regions match the Q-BANK arms'.
    This closes the other side: that the measurement covers each region
    *fully*. A candidate scored over a subset of its tensors would carry
    the full byte cost in the denominator with only part of the
    consequence in the numerator — which is precisely how a frozen
    negative could climb the ranking for a reason that is not about
    sensitivity at all.
    """
    by_proj = {}
    for r in records:
        by_proj.setdefault(r["projection"], set()).add(r["layer"])
    layers = sorted({r["layer"] for r in records})
    failures = []
    for label, rules in C.POOL.items():
        for proj, rng in rules:
            want = [l for l in layers if rng is None or rng[0] <= l <= rng[1]]
            have = by_proj.get(proj, set())
            missing = [l for l in want if l not in have]
            if missing:
                failures.append((label, proj, missing))
    return failures


def aggregate(records):
    inventory = [
        {
            "tensor": r["tensor"],
            "source_bytes": r["source_bytes"],
            "compiled_bytes": r["compiled_bytes"],
            "num": r["num"],
        }
        for r in records
    ]
    out = {}
    for label in C.POOL:
        sel = C.tensors_for(label, inventory)
        num = sum(t["num"] for t in sel)
        extra = sum(t["source_bytes"] - t["compiled_bytes"] for t in sel) / MIB
        out[label] = {
            "num": num,
            "extra_mib": extra,
            "score": num / extra if extra else 0.0,
            "tensors": len(sel),
        }
    return out


def rho(agg):
    """Marginal collapse between the first and second five protected FFN
    layers. Dimensionless, so it compares shape across models without
    asserting anything about absolute scale."""
    l5, l10 = agg.get("late5-ffn"), agg.get("late10-ffn")
    if not l5 or not l10:
        return None
    m5 = l5["num"] / l5["extra_mib"]
    d_num = l10["num"] - l5["num"]
    d_mib = l10["extra_mib"] - l5["extra_mib"]
    if d_mib <= 0 or d_num <= 0:
        # A non-positive increment means no peak by construction.
        return 0.0
    return m5 / (d_num / d_mib)


def truth_rho(sweep):
    arms = {c["label"]: c for c in sweep["candidates"]}
    base = arms.get(C.BASE_LABEL) or arms.get("R0-deploy")
    if not base:
        return None
    try:
        l5, l10 = arms["late5-ffn"], arms["late10-ffn"]
    except KeyError:
        return None
    b = (base["payload_bytes"] / MIB, base["kl"]["p99"])
    a = (l5["payload_bytes"] / MIB, l5["kl"]["p99"])
    c = (l10["payload_bytes"] / MIB, l10["kl"]["p99"])
    m5 = (b[1] - a[1]) / (a[0] - b[0])
    m10 = (a[1] - c[1]) / (c[0] - a[0])
    return m5 / m10 if m10 else None


def spearman(a, b):
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main(consequence_path, sweep_path, expect_digest=None):
    records = load(consequence_path)
    sweep = load(sweep_path)

    if not records:
        raise SystemExit("REFUSED: no consequence records")

    # ---- provenance ----------------------------------------------------
    digests = {r["calibration_token_digest"] for r in records}
    if len(digests) != 1:
        raise SystemExit(f"REFUSED: records mix calibration banks: {sorted(digests)}")
    digest = digests.pop()
    if expect_digest and digest != expect_digest:
        raise SystemExit(
            f"REFUSED: consequence was computed against a different calibration bank.\n"
            f"  expected {expect_digest}\n  records  {digest}"
        )
    moment_digests = {r["moment_artifact_digest"] for r in records}
    if len(moment_digests) != 1:
        raise SystemExit("REFUSED: records mix moment artifacts")
    print(f"calibration {digest[:16]}…  moments {moment_digests.pop()[:16]}…")

    if any(r["projection"] == "o_proj" for r in records):
        raise SystemExit(
            "REFUSED: an o_proj consequence was emitted. There is no attention-output\n"
            "site, so no honest number exists for it and the pool excludes it."
        )

    # ---- completeness --------------------------------------------------
    gaps = check_candidate_coverage(records)
    if gaps:
        print("REFUSED: candidates name tensors with no measurement:")
        for label, proj, missing in gaps:
            print(f"  {label}: {proj} missing layers {missing}")
        return 1
    print(f"pool completeness OK  {len(records)} tensors measured")

    # ---- the one aggregation -------------------------------------------
    agg = aggregate(records)
    arms = {c["label"]: c for c in sweep["candidates"]}
    base = arms.get(C.BASE_LABEL) or arms.get("R0-deploy")
    truth = {}
    for label in C.POOL:
        if label in arms and base:
            extra = (arms[label]["payload_bytes"] - base["payload_bytes"]) / MIB
            truth[label] = (base["kl"]["p99"] - arms[label]["kl"]["p99"]) / extra

    order = sorted(C.POOL, key=lambda l: -agg[l]["score"])
    print()
    print(f"{'rank':>4} {'candidate':16s} {'+MiB':>9s} {'1B′ score':>13s} {'Q-BANK p99/MiB':>15s}")
    for i, label in enumerate(order, 1):
        neg = "  (negative)" if label in C.NEGATIVES else ""
        t = truth.get(label)
        print(
            f"{i:4d} {label:16s} {agg[label]['extra_mib']:9.1f} "
            f"{agg[label]['score']:13.6g} {t if t is None else round(t, 6):>15}{neg}"
        )

    # ---- the frozen bar -------------------------------------------------
    print()
    ranks = {l: i for i, l in enumerate(order, 1)}
    n = len(order)
    late5_rank = ranks["late5-ffn"]
    neg_ranks = {g: ranks[g] for g in C.NEGATIVES}
    r = rho(agg)
    tr = truth_rho(sweep)

    c1 = all(late5_rank < ranks[g] for g in C.NEGATIVES)
    c2 = all(v > n / 2 for v in neg_ranks.values())
    c3 = (
        agg["late5-ffn"]["score"] > agg["late10-ffn"]["score"]
        and agg["late5-ffn"]["score"] > agg["late15-ffn"]["score"]
        and r is not None
        and r > 1.0
    )
    print("THE BAR (Granite)")
    print(f"  1. late5-ffn above all three negatives : {'PASS' if c1 else 'FAIL'}"
          f"   (late5 rank {late5_rank}, negatives {neg_ranks})")
    print(f"  2. v/k/down all in the bottom half     : {'PASS' if c2 else 'FAIL'}"
          f"   (rank > {n/2:.1f} required)")
    print(f"  3. knee survives, rho > 1              : {'PASS' if c3 else 'FAIL'}"
          f"   (rho_1B' {r if r is None else round(r, 3)}, truth rho {tr and round(tr, 2)})")

    if truth:
        common = [l for l in order if l in truth]
        s = spearman([agg[l]["score"] for l in common], [truth[l] for l in common])
        print(f"\n  Spearman vs Q-BANK p99/MiB {s:+.3f}   (reported; cannot rescue a failed condition)")

    verdict = c1 and c2 and c3
    print(f"\n=> Granite 1B′ {'PASS' if verdict else 'FAIL'}")
    if not verdict:
        print("   1C is earned. Do not tune this rung.")
    else:
        print("   Next: capture Glimmer moments and check rho_1B'(Glimmer) < 1.")
    return 0 if verdict else 2


if __name__ == "__main__":
    a = sys.argv[1:]
    expect = None
    if "--expect-token-digest" in a:
        i = a.index("--expect-token-digest")
        expect = a[i + 1]
        a = a[:i] + a[i + 2:]
    if len(a) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(a[0], a[1], expect))
