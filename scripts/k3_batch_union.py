#!/usr/bin/env python3
"""Does batching rescue K3-from-SSD by turning 20 tok/s into a throughput claim?

The reframe under test: a routed expert's fetch cost amortises across every
concurrent stream that routes to it in the same step, so single-stream latency
stays at 1-3 tok/s while *aggregate* throughput reaches 20. That is an
arithmetic question, not a research one, and it turns on one curve — how fast
the expert union grows with batch.

Two things decide it:

  1. **Union growth.** Over B independent tokens, top-k of E experts touches
     `E*(1-(1-k/E)^B)` distinct experts. Amortisation is `B*k / union(B)`. The
     catch is that this gets *worse* as activation gets sparser: K3 activates
     1.79% of its experts per token against Gemma-26B-A4B's 6.25%, so unions
     overlap far less at the same batch. Routing locality pulls the other way
     and is calibrated here against the DEC-0 routed arm's measured curve.

  2. **What batch costs in RAM.** Once the union saturates, step time is fixed
     and aggregate throughput is purely linear in batch — so the target batch is
     whatever hits 20 tok/s, and the question becomes whether that many
     concurrent streams' attention state fits in 128 GB alongside the resident
     dense weights.

Companion to k3_fetch_budget.py, same class of instrument: pencil, traces and
published shapes, no inference and no download.
"""

import argparse
import json
from pathlib import Path

from k3_fetch_budget import DEFAULT_CACHE, load_config, load_header

# ── Link / target premises (shared with k3_fetch_budget.py) ────────────────
LINK_GB_S = 3.5
TB_PORTS = 3
TARGET_AGG_TOK_S = 20.0

# Byte-side levers that survive DEC-8.0, applied to the fetch:
#   drop the w3 GLU branch (hard ceiling 1.5x) x cold-tail precision (1.4x).
# Row sparsity is deliberately NOT banked here — DEC-8.2's request-rate ceiling
# binds it independently, so the reframe has to stand up without it.
BYTE_LEVERS = 1.5 * 1.4

# ── DEC-0 routed arm, measured (bench/dec0/dec0_routed_replay_20260723-232826) ──
# gemma4-26b-a4b-q4k, E=128, top_k=8. experts_union_frac = union/(B*k).
DEC0 = {"n_experts": 128, "top_k": 8,
        "measured": {1: 1.0, 8: 0.5109, 16: 0.3463, 32: 0.2253, 64: 0.1386}}

# ── Per-stream attention state ─────────────────────────────────────────────
# K3 is a 3:1 hybrid: 24 layers are full MLA attention (cache grows with
# context), 69 are KDA linear attention (fixed-size recurrent state).
#
# VERIFIED against modeling_kimi_linear.py: `head_k_dim = head_dim` and
# `num_k_heads = num_heads` (KimiDeltaAttention.__init__), so the recurrent
# state is [num_heads, head_k_dim, head_v_dim] = 96 x 128 x 128 per layer per
# stream. `kda_layers`/`full_attn_layers` in the config list the split
# explicitly. Plus three ShortConvolution states (q/k/v) of
# [projection_size, kernel_size - 1] each.
#
# NOT verified: the state's dtype. modeling_kimi_linear.py delegates to fla's
# chunk_kda / fused_recurrent_kda, which conventionally hold recurrent state in
# fp32; bf16 is the optimistic reading. Reported as a band over both.
STATE_DTYPE_BYTES = (2, 4)
DEFAULT_CONTEXT = 2048
MACHINE_RAM_GB = 128.0
DENSE_RESIDENT_4BIT_GB = 29.5 + 1.2   # non-expert weights + embeddings, 4-bit


def union_uniform(n_experts, top_k, batch):
    """Expected distinct experts over `batch` independent top-k draws."""
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** batch)


def locality_calibration():
    """How much better than uniform is real routing? Measured, per batch."""
    E, k = DEC0["n_experts"], DEC0["top_k"]
    out = {}
    for b, measured_frac in DEC0["measured"].items():
        uniform_frac = union_uniform(E, k, b) / (b * k)
        out[b] = measured_frac / uniform_frac
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    ap.add_argument("--state-dtype-bytes", type=int, default=None,
                    help="pin the recurrent-state dtype width (default: band 2 and 4)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    hdr, cfg = load_header(args.cache), load_config(args.cache)

    E = cfg["num_experts"]
    k = cfg["num_experts_per_token"]
    n_moe = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
    n_layers = cfg["num_hidden_layers"]
    full_attn = set(cfg["linear_attn_config"]["full_attn_layers"])
    n_full = len(full_attn)
    n_linear = n_layers - n_full

    expert_bytes = sum(
        e["data_offsets"][1] - e["data_offsets"][0]
        for name, e in hdr.items() if ".experts.0." in name
    )
    bank_bytes = E * expert_bytes * n_moe
    link = LINK_GB_S * 1e9 * TB_PORTS

    # per-stream state, per candidate state dtype
    lac = cfg["linear_attn_config"]
    heads, head_dim = lac["num_heads"], lac["head_dim"]
    n_linear = len(lac["kda_layers"])
    conv_elems = 3 * heads * head_dim * (lac["short_conv_kernel_size"] - 1)

    def stream_state(dtype_bytes):
        kda = heads * head_dim * head_dim * dtype_bytes + conv_elems * dtype_bytes
        mla = (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * 2 * args.context
        return n_linear * kda + n_full * mla, kda

    dtypes = ([int(args.state_dtype_bytes)] if args.state_dtype_bytes
              else list(STATE_DTYPE_BYTES))
    stream_bytes, kda_state = stream_state(dtypes[0])

    loc = locality_calibration()
    loc_mean = sum(loc[b] for b in loc if b > 1) / len([b for b in loc if b > 1])

    print("=" * 78)
    print("K3 batch-union: can 20 tok/s be an aggregate claim?")
    print("=" * 78)
    print(f"K3: {k}-of-{E} ({100*k/E:.2f}% activation), {n_moe} MoE layers, "
          f"expert {expert_bytes/1e6:.2f} MB, bank {bank_bytes/1e12:.3f} TB")
    print(f"link {LINK_GB_S} GB/s x {TB_PORTS} ports = {link/1e9:.1f} GB/s, "
          f"byte levers {BYTE_LEVERS:.2f}x (w3 drop x precision tail)")
    print()
    print("--- locality calibration, from the DEC-0 routed arm (measured) ---")
    print(f"gemma4-26b-a4b-q4k, {DEC0['top_k']}-of-{DEC0['n_experts']} "
          f"({100*DEC0['top_k']/DEC0['n_experts']:.2f}% activation)")
    print(f"{'batch':>6} {'measured':>10} {'uniform':>10} {'locality':>10}")
    for b in sorted(DEC0["measured"]):
        uf = union_uniform(DEC0["n_experts"], DEC0["top_k"], b) / (b * DEC0["top_k"])
        print(f"{b:>6} {DEC0['measured'][b]:>10.4f} {uf:>10.4f} {loc[b]:>10.3f}")
    print(f"  => real routing touches ~{loc_mean:.2f}x the uniform union "
          f"(locality helps; applied to K3 as the optimistic arm)")
    print()

    print("--- K3 union growth and aggregate throughput ---")
    print(f"{'batch':>6} {'union':>8} {'of bank':>8} {'amort':>7} "
          f"{'GB/token':>9} {'step s':>8} {'agg tok/s':>10} {'state GB':>9}")
    rows = []
    for b in (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        u_uni = union_uniform(E, k, b)
        u = min(float(E), loc_mean * u_uni)          # optimistic arm
        step_bytes = u * expert_bytes * n_moe / BYTE_LEVERS
        step_s = step_bytes / link
        agg = b / step_s
        gb_tok = step_bytes / b / 1e9
        state_gb = b * stream_bytes / 1e9
        rows.append({"batch": b, "union": u, "union_frac_of_bank": u / E,
                     "amortisation_x": (b * k) / u, "gb_per_token": gb_tok,
                     "step_seconds": step_s, "aggregate_tok_s": agg,
                     "stream_state_gb": state_gb})
        print(f"{b:>6} {u:>8.1f} {u/E:>8.1%} {(b*k)/u:>7.2f}x "
              f"{gb_tok:>9.2f} {step_s:>8.1f} {agg:>10.2f} {state_gb:>9.1f}")
    print()

    # Past saturation the step time is fixed and aggregate is linear in batch.
    # Where the union actually saturates is UNMEASURED past B=64 for any model,
    # so bound it both ways instead of picking one:
    #   pessimistic — locality washes out, every expert is eventually touched
    #   optimistic  — the measured ~0.56 locality factor persists to the ceiling
    ram_free_gb = MACHINE_RAM_GB - DENSE_RESIDENT_4BIT_GB
    b_ram_max = ram_free_gb * 1e9 / stream_bytes

    print(f"--- per-stream attention state (context {args.context}) ---")
    ram_caps = {}
    for db in dtypes:
        sb, kda = stream_state(db)
        cap = ram_free_gb * 1e9 / sb
        ram_caps[db] = (sb, cap)
        print(f"  state dtype {db}B: KDA {kda/1e6:6.2f} MB/layer x {n_linear} "
              f"+ MLA {n_full} layers x {args.context} tok  ->  "
              f"{sb/1e6:6.0f} MB/stream, RAM caps batch at {cap:.0f}")
    print()

    arms = {}
    for label, ceiling in (("pessimistic (union -> all 896)", float(E)),
                           ("optimistic (locality holds, 504)", loc_mean * E)):
        step_bytes = ceiling * expert_bytes * n_moe / BYTE_LEVERS
        step_s = step_bytes / link
        arms[label] = {
            "union_ceiling": ceiling,
            "saturated_step_bytes": step_bytes,
            "saturated_step_seconds": step_s,
            "batch_for_target": TARGET_AGG_TOK_S * step_s,
            "state_gb_at_target_batch": TARGET_AGG_TOK_S * step_s * stream_bytes / 1e9,
            "aggregate_tok_s_at_ram_ceiling": b_ram_max / step_s,
        }

    print("--- the crossing point (union ceiling past B=64 is UNMEASURED: band) ---")
    for label, a in arms.items():
        print(f"  {label}")
        print(f"    saturated step {a['saturated_step_bytes']/1e9:>6.0f} GB "
              f"= {a['saturated_step_seconds']:>5.1f} s   ->  "
              f"B for {TARGET_AGG_TOK_S:.0f} tok/s = {a['batch_for_target']:>6.0f} streams "
              f"({a['state_gb_at_target_batch']:.0f} GB of state)")
    print()
    print(f"--- does that batch fit? free RAM {ram_free_gb:.1f} GB "
          f"({MACHINE_RAM_GB:.0f} - {DENSE_RESIDENT_4BIT_GB:.1f} dense at 4-bit) ---")
    corners = []
    for db, (sb, cap) in ram_caps.items():
        for label, a in arms.items():
            agg = cap / a["saturated_step_seconds"]
            corners.append(agg)
            print(f"  state {db}B x {label:<32} batch {cap:>4.0f} "
                  f"-> {agg:>5.1f} tok/s aggregate")
    lo, hi = min(corners), max(corners)
    print(f"  => aggregate ceiling  {lo:.1f}-{hi:.1f} tok/s "
          f"(short of {TARGET_AGG_TOK_S:.0f} by {TARGET_AGG_TOK_S/hi:.1f}-"
          f"{TARGET_AGG_TOK_S/lo:.1f}x)")
    print()
    print("  The verdict does NOT rest on the union extrapolation: both arms are")
    print("  capped by the same RAM ceiling, which is what actually binds.")
    print("=" * 78)

    out = {
        "premises": {"link_gb_s": LINK_GB_S, "ports": TB_PORTS,
                     "byte_levers_x": BYTE_LEVERS, "target_agg_tok_s": TARGET_AGG_TOK_S,
                     "context": args.context, "machine_ram_gb": MACHINE_RAM_GB,
                     "dense_resident_4bit_gb": DENSE_RESIDENT_4BIT_GB},
        "k3": {"n_experts": E, "top_k": k, "activation_frac": k / E,
               "n_moe_layers": n_moe, "expert_bytes": expert_bytes,
               "bank_bytes": bank_bytes, "n_full_attn_layers": n_full,
               "n_linear_attn_layers": n_linear},
        "locality_calibration_dec0": loc,
        "locality_mean_applied": loc_mean,
        "curve": rows,
        "arms": arms,
        "stream_state_bytes": stream_bytes,
        "batch_ram_ceiling": b_ram_max,
        "aggregate_tok_s_ceiling_band": [lo, hi],
        "ram_caps_by_state_dtype": {str(k): {"stream_bytes": v[0], "batch_cap": v[1]} for k,v in ram_caps.items()},
        "shortfall_x_band": [TARGET_AGG_TOK_S / hi, TARGET_AGG_TOK_S / lo],
        "kda_state_shape": [heads, head_dim, head_dim],
        "kda_state_shape_verified": "modeling_kimi_linear.py KimiDeltaAttention",
        "state_dtype_bytes_banded": dtypes,
    }
    out_path = args.out or (args.cache / "k3_batch_union.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
