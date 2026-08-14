#!/usr/bin/env python3
"""Diff two engines' MoE expert selections, and test the tie-break hypothesis.

`docs/k3-funnel.md` §4.9.2 records that GPT-OSS's 511-token layer diff passes
(cos >= 0.99997) with a residual much larger than the 85-token run, and
*attributes* that to expert-selection disagreement at near-tie router scores.
That attribution was reasoned, not measured. This measures it.

    LARQL_MOE_ROUTE_TRACE=rust.jsonl larql shannon layer-dump ... --out dumps/rust
    python scripts/dump_layers_hf.py MODEL --ref dumps/rust --out dumps/hf \\
        --router-trace hf.jsonl
    python scripts/diff_moe_routing.py --rust rust.jsonl --hf hf.jsonl

**The hypothesis is falsifiable and this is how it fails.** If the two engines
select identical experts everywhere, the residual comes from somewhere else
entirely and §4.9.2's explanation is wrong. If they flip at wide router margins
*while the residual entering that layer is still tiny*, the router is wrong
rather than tie-breaking.

**Judge the cascade, not a flat rate.** A first flip perturbs its token's
residual, which shifts that token's router logits downstream, which can flip it
again at a margin far above a tie. Those later flips are *consequences*, so
scoring all disagreements against one margin threshold treats a causal chain as
independent draws and reports a router defect that is not there. The
discriminating question is whether a wide-margin flip ever happens *before* the
residual has diverged — pass `--layer-diff` to test it.

Rust record:  {"layer": L, "seq": S, "experts": [[e,...], ...]}
HF record:    {"layer": L, "experts": [[e,...], ...], "margin": [m, ...]}
"""

import argparse
import json
from pathlib import Path

# A margin below this counts as "effectively tied" — the two engines' router
# logits differ by more than this only through accumulated f32-vs-bf16
# arithmetic, so a swap here is expected and a swap above it is not.
TIE_MARGIN = 1e-2


def load(path: Path) -> dict[int, dict]:
    """Layer -> record. Rust writes one record per (layer, call); a single
    forward makes one call per layer, so a duplicate layer means the trace
    spans more than one forward and the comparison is not well defined."""
    out: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        layer = rec["layer"]
        if layer in out:
            raise SystemExit(
                f"{path}: layer {layer} appears twice — the trace spans multiple "
                "forward passes; clear the file and run one pass"
            )
        out[layer] = rec
    if not out:
        raise SystemExit(f"{path}: empty trace")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--tie-margin", type=float, default=TIE_MARGIN)
    ap.add_argument("--layer-diff", type=Path, default=None, metavar="FILE",
                    help="a `larql shannon layer-diff --json` RESULT line; enables the "
                         "cascade test, which is the one that discriminates")
    args = ap.parse_args()

    entering: dict[int, float] = {}
    if args.layer_diff:
        text = args.layer_diff.read_text().strip()
        for line in text.splitlines():
            if line.startswith("RESULT "):
                text = line[len("RESULT "):]
        payload = json.loads(text)
        # capture i+1 is layer i's output, so layer L is *entered* with L-1's.
        by_layer = {c["capture"] - 1: c["rel_rms"] for c in payload["captures"] if c["capture"] > 0}
        entering = {L: by_layer.get(L - 1, 0.0) for L in by_layer}

    rust, hf = load(args.rust), load(args.hf)
    shared = sorted(set(rust) & set(hf))
    if not shared:
        raise SystemExit(f"no layers in common: rust {sorted(rust)} vs hf {sorted(hf)}")
    if set(rust) != set(hf):
        print(f"warning: layer sets differ — rust {sorted(rust)}, hf {sorted(hf)}")

    print(f"{'layer':>5} {'tokens':>7} {'differ':>7} {'%':>7} "
          f"{'tied':>6} {'wide':>6} {'max_wide_margin':>16}")
    print("-" * 60)

    total_tokens = total_differ = total_tied = total_wide = 0
    wide_margins: list[float] = []
    all_margins_at_diff: list[float] = []
    # (layer, token, margin, rel_rms entering the layer, is_tied)
    events: list[tuple[int, int, float, float, bool]] = []

    for layer in shared:
        r_sel = rust[layer]["experts"]
        h_sel = hf[layer]["experts"]
        margins = hf[layer].get("margin") or [float("nan")] * len(h_sel)
        n = min(len(r_sel), len(h_sel))
        if len(r_sel) != len(h_sel):
            print(f"warning: layer {layer} token counts differ "
                  f"({len(r_sel)} vs {len(h_sel)}); comparing first {n}")

        differ = tied = wide = 0
        layer_wide_max = 0.0
        for t in range(n):
            # Selection is a *set*: the same experts in a different order is
            # the same computation, so only membership counts.
            if set(r_sel[t]) == set(h_sel[t]):
                continue
            differ += 1
            m = float(margins[t])
            all_margins_at_diff.append(m)
            if m <= args.tie_margin:
                tied += 1
                events.append((layer, t, m, entering.get(layer, 0.0), True))
            else:
                wide += 1
                wide_margins.append(m)
                layer_wide_max = max(layer_wide_max, m)
                events.append((layer, t, m, entering.get(layer, 0.0), False))

        pct = 100.0 * differ / n if n else 0.0
        print(f"{layer:>5} {n:>7} {differ:>7} {pct:>6.2f}% "
              f"{tied:>6} {wide:>6} {layer_wide_max:>16.4f}")
        total_tokens += n
        total_differ += differ
        total_tied += tied
        total_wide += wide

    print("-" * 60)
    pct = 100.0 * total_differ / total_tokens if total_tokens else 0.0
    print(f"{'ALL':>5} {total_tokens:>7} {total_differ:>7} {pct:>6.2f}% "
          f"{total_tied:>6} {total_wide:>6}")
    print()

    # ── verdict ──────────────────────────────────────────────────────
    if total_differ == 0:
        print("VERDICT: REFUTED — expert selections are IDENTICAL on every token "
              "of every layer.")
        print("  The residual growth is NOT expert-selection disagreement. "
              "§4.9.2's attribution is wrong and the real cause is unmeasured.")
        return

    tied_frac = total_tied / total_differ
    print(f"disagreements: {total_differ} / {total_tokens} tokens ({pct:.2f}%)")
    print(f"  at margin <= {args.tie_margin}: {total_tied} ({100 * tied_frac:.1f}%)")
    print(f"  at wider margin:          {total_wide} ({100 * (1 - tied_frac):.1f}%)")
    if wide_margins:
        print(f"  widest disagreeing margin: {max(wide_margins):.4f}")

    affected = sorted({t for _, t, _, _, _ in events})
    print(f"  distinct tokens involved:  {len(affected)} of {total_tokens // len(shared)} "
          f"({', '.join(str(t) for t in affected[:12])}"
          f"{', ...' if len(affected) > 12 else ''})")
    print()

    if not entering:
        print("VERDICT: INCONCLUSIVE — pass --layer-diff to run the cascade test.")
        print("  Without it, a flat margin threshold cannot separate a seeded")
        print("  cascade from a router defect.")
        return

    events.sort(key=lambda e: (e[0], e[1]))
    seed = events[0]
    print(f"{'layer':>5} {'token':>6} {'margin':>10} {'rel_rms entering':>18} {'tied':>6}")
    for layer, tok, m, ent, is_tied in events:
        print(f"{layer:>5} {tok:>6} {m:>10.5f} {ent:>18.6f} {str(is_tied):>6}")
    print()
    print(f"seed (first flip): layer {seed[0]}, token {seed[1]}, "
          f"margin {seed[2]:.6f}, entering rel_rms {seed[3]:.3e}")

    wide_events = [e for e in events if not e[4]]
    tied_events = [e for e in events if e[4]]
    if not wide_events:
        print()
        print("VERDICT: CONFIRMED — every flip is a near-tie.")
        return

    min_wide_entering = min(e[3] for e in wide_events)
    earlier_tied = [e for e in tied_events if e[3] < min_wide_entering]
    print(f"every wide-margin flip enters with rel_rms >= {min_wide_entering:.6f}")
    print(f"flips below that divergence: {len(earlier_tied)}, all tied: "
          f"{all(e[4] for e in earlier_tied)}")
    print()

    # The discriminating condition: a defective router flips at wide margins
    # regardless of how far the residual has drifted, so it would do so at the
    # shallowest layers too. A cascade cannot — it has nothing to cascade from.
    seed_is_tie = seed[2] <= args.tie_margin
    no_early_wide = all(e[4] for e in earlier_tied)
    if seed_is_tie and no_early_wide:
        print("VERDICT: CONFIRMED — a tie-break cascade, not a router defect.")
        print(f"  The first flip is a near-tie ({seed[2]:.2e}) at negligible divergence,")
        print("  and no wide-margin flip occurs before the residual has drifted.")
        print("  A defective router would flip at wide margins from the first layer.")
    else:
        print("VERDICT: REFUTED — wide-margin flips occur before the residual")
        print("  diverges, so they cannot be downstream consequences. The router")
        print("  itself disagrees with the reference.")


if __name__ == "__main__":
    main()
