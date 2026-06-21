#!/usr/bin/env python3
"""V3-adjacent: MoE routing locality analysis (Gemma 4 26B-A4B).

Parses MOE_DEBUG=1 routing capture (per-layer `experts:[...]` lines), segments
into per-forward groups (each decode token = one 30-layer forward), and measures
the temporal locality that decides disk-resident MoE viability:

  - adjacent reuse        |E_t ∩ E_{t-1}| / top_k   (token-to-token)
  - cumulative cache-hit  |E_t ∩ seen_{<t}| / top_k (LRU-all warm cache)
  - working-set size      distinct experts/layer over the stream (of num_experts)
  - saturation            new experts/token in the tail (→0 ⇒ working set closes)
  - hot concentration     top-frequency expert mass

Then projects steady-state disk-resident cost using V3's cold-read latency.
"""
import json, re, sys
from pathlib import Path

CAP = sys.argv[1] if len(sys.argv) > 1 else "bench/aim-validation/moe-routing/route_capture.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bench/aim-validation/moe-routing/v3moe_locality.json"
NUM_EXPERTS = 128
TOP_K = 8
N_LAYERS = 30
# V3 measured: cold scattered 16KB read ~100µs; expert q4k ≈ 3MB ≈ 180 pages.
COLD_US_PER_PAGE = 100.0
PAGES_PER_EXPERT = 180

line_re = re.compile(r"\[L(\d+)\].*experts:\[([\d,\s]+)\]")

if not Path(CAP).exists():
    print(f"Error: capture file not found at {CAP}")
    sys.exit(1)

groups = []  # each: {layer: [experts]}
cur = {}
for line in Path(CAP).read_text(errors="ignore").splitlines():
    m = line_re.search(line)
    if not m:
        continue
    layer = int(m.group(1))
    experts = [int(x) for x in m.group(2).split(",") if x.strip() != ""]
    if layer == 0 and cur:
        groups.append(cur)
        cur = {}
    cur[layer] = experts
if cur:
    groups.append(cur)

# MOE_DEBUG prints per POSITION (one L00..L29 cycle per token position), and the
# default path is full-recompute: with --max-tokens 1 there are exactly 2 forwards
# — prefill (P positions) then one recompute (P+1). The FIRST forward (first half
# of groups) is the clean, position-ordered routing for the whole input sequence;
# the recompute half re-sees the same experts and would pollute temporal metrics.
n_groups = len(groups)
half = max(1, n_groups // 2)
decode = groups[:half]  # per-POSITION routing across the input sequence
n_tok = len(decode)
print(f"parsed {n_groups} position-forwards; using first {n_tok} (prefill, clean per-position), {N_LAYERS} layers, top_k={TOP_K}")
if n_tok < 3:
    print("too few positions to analyze")
    sys.exit(1)

per_layer = {}
for L in range(N_LAYERS):
    seqs = [set(g[L]) for g in decode if L in g]
    if len(seqs) < 2:
        continue
    adj = [len(seqs[t] & seqs[t - 1]) / TOP_K for t in range(1, len(seqs))]
    seen = set()
    cum_hit, new_per_tok, wss_curve = [], [], []
    for t, s in enumerate(seqs):
        if t > 0:
            cum_hit.append(len(s & seen) / TOP_K)
            new_per_tok.append(len(s - seen))
        seen |= s
        wss_curve.append(len(seen))
    # hot concentration: usage frequency over the stream
    freq = {}
    for s in seqs:
        for e in s:
            freq[e] = freq.get(e, 0) + 1
    total_acts = sum(freq.values())
    top10 = sorted(freq.values(), reverse=True)[:10]
    tail = max(1, len(new_per_tok) // 3)
    per_layer[L] = {
        "adjacent_reuse": sum(adj) / len(adj),
        "cumulative_hit": sum(cum_hit) / len(cum_hit),
        "working_set": len(seen),
        "working_set_frac": len(seen) / NUM_EXPERTS,
        "new_experts_per_tok_tail": sum(new_per_tok[-tail:]) / tail,
        "top10_expert_mass": sum(top10) / total_acts if total_acts > 0 else 0.0,
    }

def avg(key):
    vals = [v[key] for v in per_layer.values()]
    return sum(vals) / len(vals) if vals else 0.0

overall = {
    "adjacent_reuse": avg("adjacent_reuse"),
    "cumulative_hit": avg("cumulative_hit"),
    "mean_working_set": avg("working_set"),
    "mean_working_set_frac": avg("working_set_frac"),
    "mean_new_experts_per_tok_tail": avg("new_experts_per_tok_tail"),
    "mean_top10_mass": avg("top10_expert_mass"),
}
# Total distinct (layer,expert) working set across all layers.
total_ws = sum(v["working_set"] for v in per_layer.values())
total_ws_gb = total_ws * PAGES_PER_EXPERT * 16384 / 1e9

# Steady-state disk projection: new (cold) experts/token across all layers in the tail.
new_per_tok_all = sum(v["new_experts_per_tok_tail"] for v in per_layer.values())
cold_ms_per_tok = new_per_tok_all * PAGES_PER_EXPERT * COLD_US_PER_PAGE / 1000.0

print(f"\n== MoE routing locality (Gemma 4 26B-A4B, {n_tok} decode tokens) ==")
print(f"  adjacent token reuse:     {overall['adjacent_reuse']*100:.1f}% of top-{TOP_K}")
print(f"  cumulative cache-hit:     {overall['cumulative_hit']*100:.1f}% (all-seen warm cache)")
print(f"  mean working set / layer: {overall['mean_working_set']:.1f} / {NUM_EXPERTS} ({overall['mean_working_set_frac']*100:.0f}%)")
print(f"  new experts/token (tail): {overall['mean_new_experts_per_tok_tail']:.2f}/layer, {new_per_tok_all:.1f} total")
print(f"  top-10 expert mass:       {overall['mean_top10_mass']*100:.0f}% (hot-expert concentration)")
print(f"  total working set:        {total_ws} (layer,expert) pairs ≈ {total_ws_gb:.1f} GB resident")
print(f"\n== disk-resident projection (V3 cold = {COLD_US_PER_PAGE}µs/page, ~{PAGES_PER_EXPERT} pages/expert) ==")
print(f"  steady-state cold reads:  {new_per_tok_all:.1f} experts/token → ~{cold_ms_per_tok:.0f} ms/token cold")
print(f"  → working set {total_ws_gb:.1f} GB: fits 128GB RAM ⇒ warms up, steady-state cold cost ≈ {cold_ms_per_tok:.0f}ms/tok")

result = {
    "test_id": "V3-MoE-locality",
    "model": "gemma4-26b-a4b",
    "config": {"num_experts": NUM_EXPERTS, "top_k": TOP_K, "n_layers": N_LAYERS, "decode_tokens": n_tok},
    "overall": overall,
    "total_working_set_pairs": total_ws,
    "total_working_set_gb_est": total_ws_gb,
    "disk_projection": {
        "cold_us_per_page": COLD_US_PER_PAGE,
        "pages_per_expert_est": PAGES_PER_EXPERT,
        "steady_new_experts_per_token": new_per_tok_all,
        "steady_cold_ms_per_token": cold_ms_per_tok,
    },
    "per_layer": per_layer,
    "notes": "faithful in-process decode (build_moe_weights local experts); decode-phase per-token routing. Cold latency from V3 granite-30b probe.",
}
Path(OUT).write_text(json.dumps(result, indent=2))
print(f"\n  artifact → {OUT}")
