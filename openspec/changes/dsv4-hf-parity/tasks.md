## 1. Reference-dump format + Python generator

- [x] 1.1 JSON dump format (`model`, `prompt`, `dtype`, `token_ids`, `top_k: [{token_id, logit}]`) — documented in `scripts/dsv4_hf_reference.py`, `tests/goldens/README.md`, and the test's doc comment.
- [x] 1.2 `scripts/dsv4_hf_reference.py` — loads HF DeepSeek-V4-Flash via `transformers`, tokenizes the prompt, runs one forward, writes the final-position top-K dump; reports the top1→top2 logit gap (warns if argmax may not be Q4_K-stable).

## 2. Rust parity harness

- [x] 2.1 Reference-dump loader — `serde_json` `Deserialize` structs (`HfReference { token_ids, top_k }`).
- [x] 2.2 Parity test `dsv4_forward_matches_hf_reference` (`#[ignore]`, `tests/test_dsv4_hf_parity.rs`): opens the GGUF + hyperparams + head, runs `dsv4_streaming_model_forward_cached` on the dump's `token_ids` across all 43 layers, extracts final-position top-K via `dsv4_topk_logits`.
- [x] 2.3 Assertions: greedy argmax == reference top-1 (load-bearing); top-K set overlap ≥ k/2; top-1 logit within `LOGIT_REL_TOL = 0.15` (generous, calibration-pending).
- [x] 2.4 Skip-clean: missing dump (`LARQL_DSV4_HF_REF` / default `tests/goldens/`) or GGUF → print skip + return success. **Verified**: runs green-by-skip with no dump present.

## 3. Verify + wire

- [x] 3.1 `cargo test -p larql-inference --test test_dsv4_hf_parity` compiles; the test skips cleanly with no dump (green-by-skip). clippy clean.
- [x] 3.2 Spec scenarios linked via `<!-- test: -->` (consume / greedy / top-K / skip → the parity test). The dump-format scenario describes the Python generator (no Rust test) and stays unbacked.
- [x] 3.3 `openspec validate dsv4-hf-parity --strict` valid; traceability regenerated.
- [ ] 3.4 **(Out-of-band, maintainer)** Generate the dump with the HF DeepSeek-V4-Flash + `transformers`, commit it (top-K JSON is small), calibrate `LOGIT_REL_TOL`, and confirm the test passes on the real GGUF. This is the only step needing the HF weights; the harness is complete and inert until then.
