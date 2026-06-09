# `LARQL_QWEN35_KV_PRESEED` bench results

Debug knob that synthetically fills the device KV cache to N rows on
first allocation. Output is not coherent (zeros in the preseed slots)
but VRAM peak and per-token decode time are measured under the
long-context pressure they'd see with a real N-token prefill.

## Why this exists

At ~70 ms / prompt token on Qwen3.6-35B-A3B, a true 32K-token prefill
takes 37 minutes per data point; 128K takes ~2.5 hours. The
`LARQL_QWEN35_KV_PRESEED` knob lets us validate the VRAM-savings
architecture **without** paying for the prefill. It's debug-only —
production callers run real prefill.

## Captured 2026-05-21 (RTX 4090, max_seq=40000, qwen35-35B-A3B)

All runs use `LARQL_QWEN35_GPU=1 LARQL_QWEN35_KV_MAX_SEQ=40000` and
the prompt "hi" with 16-token decode. Run 1 is cold (lazy weight load
+ kernel compilation). Run 3 is steady-state.

| preseed | mode | Run 3 wall_s | Run 3 tok/s | VRAM (MiB) |
|---|---|---|---|---|
| 4096  | f16  | 1.18 | 13.53 | 10380 |
| 4096  | iso3 | 1.27 | 12.55 | **9900** |
| 16384 | iso3 | 1.33 | 12.03 | 9900 |

**Iso3 saves 480 MiB at max_seq=40000** vs f16 (consistent across
preseed values — VRAM is set by max_seq, not by current cached_seq_len).

## Throughput cost

Iso3 is ~8% slower than f16 at this effective context. The dequant
overhead (`dequantize_to_f32_device` + `f32_to_f16_device_into`) for
the full slab on each attention call grows with `cached_seq_len`. At
preseed=4096 with kv_dim=512 the dequant pass is 2M f32 ops per K and
per V per layer per token — measurable.

## Theory vs observed

Theoretical VRAM at max_seq=40000, kv_dim=512:

  f16 slabs (16 attn layers × K + V): 16 × 40000 × 512 × 4 bytes  = 1280 MiB
  iso3 codes + scratches:              256 MiB codes + 156 MiB scratch  = 412 MiB
  Expected delta:                                                          868 MiB

Observed delta: 480 MiB. Discrepancy attributable to:
- Lazy allocation: cudarc's `alloc_zeros` doesn't commit pages until
  first touch. At preseed=4096 only 4096 / 40000 = ~10 % of the slab
  pages have been read by the attention kernel — the rest are
  uncommitted.
- Per-process VRAM tracking via `nvidia-smi` reports the committed
  working set, not the virtual reservation.

The architectural delta scales with **touched** rows. Going from
preseed=4096 (480 MiB saving) to preseed=32K (theoretical ~700 MiB)
to preseed=128K (theoretical ~3.2 GiB) needs each kernel call to
actually iterate over those rows — which it does once `pos = N - 1`.

## Reproducing

```bash
# f16 baseline at effective 4K context
LARQL_QWEN35_GPU=1 \
  LARQL_QWEN35_KV_MAX_SEQ=40000 \
  LARQL_QWEN35_KV_PRESEED=4096 \
  ./target/release/larql-server <vindex> --port 8181

# Iso3 at effective 4K context
LARQL_QWEN35_GPU=1 \
  LARQL_QWEN35_KV_FORMAT=iso3 \
  LARQL_QWEN35_KV_MAX_SEQ=40000 \
  LARQL_QWEN35_KV_PRESEED=4096 \
  ./target/release/larql-server <vindex> --port 8181

# Bench a single decode (any prompt — preseed dictates the cache
# state regardless of prompt content):
curl -s --max-time 60 -X POST http://localhost:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}],
       "max_tokens":16,"temperature":0.0}'

# Compare nvidia-smi VRAM usage across the two server modes.
```

The first request triggers lazy weight load (~30-40 s on Qwen3.6-35B-A3B);
discard its timing. Runs 2+ are steady-state.
