---
name: BOUNDARY ref confidence-gate design
description: Exp 43 result — BOUNDARY refs are conditional compressed checkpoints; confidence gate determines when to compress vs fall back
type: project
---

Exp 43 (residual stream codec) established that int8-clip3σ boundary compression
works at confident positions and cascades catastrophically at low-margin positions.
The result changed the BOUNDARY ref design from "always compress" to a conditional policy.

**Why:** tech_3 (93.3% within-text top-1) had 0/40 continuation divergence while
qa_2 (95.8%) had 40/40. Per-position mean top-1 does not predict continuation quality.
What matters is the margin P(top-1) − P(top-2) at the specific boundary position.

**How to apply:** BOUNDARY ref frames now carry metadata:
- `raw_top1_prob`, `raw_margin`, `compressed_agrees`, `fallback_policy`

Runtime policy:
```python
if compressed_top1 == raw_top1 and raw_margin > MARGIN_THRESHOLD:
    send int8_clip3sigma boundary  # 2× compressed, Contract D-
else:
    send bf16 boundary or cold-replay reference
```

MARGIN_THRESHOLD is provisionally ~0.3–0.5; Exp 44 will calibrate.

**Stack position:** 2564 bytes (int8-clip3σ) vs 5120 bytes (bf16) vs ~1024-byte target.
The 1KB target needs another ~2.5× beyond int8-clip3σ — outlier-aware quant (Exp 44).

**Key insight:** A BOUNDARY ref is not just a compressed tensor.
It is a contracted model-state packet: compressed when confident, exact when fragile.
The compression scheme and its behavioural contract must be negotiated together.
