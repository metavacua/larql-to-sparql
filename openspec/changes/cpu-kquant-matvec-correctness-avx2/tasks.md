# Tasks

All implementation already shipped via PRs #102, #103, #104, #105, #106, #107, #108. This list captures the open follow-ups.

## Done (this change)

- [x] Fix Q6_K wire-format layout in CPU matvec readers (#102)
- [x] Add canonical-dequant oracle test for Q6_K (#102)
- [x] Add `walk_ffn_q8k` format dispatch for Q4_K vs Q6_K FFN_DOWN (#103)
- [x] AVX2 `q6k_q8k_matvec_avx2` bit-exact vs scalar (#104)
- [x] Disable broken aarch64 NEON Q6_K dispatch, keep kernel as reference (#104)
- [x] Add canonical-dequant oracle test for Q4_K (#105)
- [x] Add cross-path parity test (f32 trait vs Q8K AVX2 on same Q6_K weights) (#106)
- [x] Add `quantize_q6_k` round-trip oracle test (#107)
- [x] Add head-to-head AVX2 vs scalar Q6_K bench (#108)
- [x] Route `CpuBackend::q4k_matvec` / `q6k_matvec` (trait dispatch) through AVX2 Q8K kernel (#110)
- [x] x86_64 fallback for `q4k_q8k_gate_up_into` dispatches to AVX2 `q4k_q8k_matvec_into` (#112)
- [x] Bench q4k gate+up fused vs split AVX2 (#113)
- [x] Q4_KF streaming dequant+dot — eliminates 2.7 GB alloc (#114)
- [x] Q4_KF AVX2 — `q4kf_q8k_matvec_avx2` parallel to Q4_K AVX2 (#115)

## Open follow-ups

- [ ] Re-vectorise aarch64 NEON `q6k_q8k_matvec_neon` against canonical layout — task #123
- [ ] Audit Metal `q6k_matvec` shader layout vs vindex Q6_K — task #124
