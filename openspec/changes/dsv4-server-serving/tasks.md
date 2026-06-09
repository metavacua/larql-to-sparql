## 1. S0 — Groundwork

- [ ] 1.1 Confirm how a `deepseek_v4` model reaches the server today: does a DSv4 vindex even exist, or is the model GGUF-only? Decide how a DSv4 entry declares its source GGUF path (a `gguf` field in the vindex config, vs a server-config field, vs a minimal tokenizer-only vindex + gguf path). Record in design D1.
- [ ] 1.2 Confirm the tokenizer the server loads for a DSv4 entry matches the DSv4-Flash vocab (129280) and round-trips a prompt.
- [ ] 1.3 Verify host RAM headroom for resident Flash (~161 GB) on the target box; document the requirement + the OOM-failure behavior.

## 2. S1 — GGUF-backed model load + residency

- [ ] 2.1 `DsV4ServeState { layers: Vec<(DsV4LayerWeightStorage, DsV4LayerVariant)>, hp: DsV4Hyperparams, head: DsV4HeadStorage, prefix_cache: Mutex<DsV4PrefixCache> }` in larql-server; `Arc`-shared.
- [ ] 2.2 `LoadedModel.dsv4: OnceLock<Arc<DsV4ServeState>>` + an `ensure_dsv4` that opens the GGUF, runs `from_gguf` + `load_dsv4_resident_layers` + `load_dsv4_head`, opens the prefix cache, once. Errors (missing GGUF, OOM) → typed `ServerError`, no panic.
- [ ] 2.3 Plumb the GGUF path per D1 (S0.1). Unit-test `ensure_dsv4` wiring with a small synthetic stand-in where possible; real load is an `#[ignore]` integration test.

## 3. S2 — Chat-route DSv4 branch (buffered)

- [ ] 3.1 Replace the `deepseek_v4` rejection (chat.rs:~762) with a branch: `ensure_dsv4`, tokenize, `dsv4_resident_generate_with_prefix_cache`, detokenize → `GenerateResult` (tokens + timings). Mirror the qwen35 branch shape.
- [ ] 3.2 Honor `max_tokens`, sampling (temp/top-k/top-p/greedy), EOS, and stop-strings using the route's existing post-processing over the produced ids.
- [ ] 3.3 `#[ignore]` integration test: a `/v1/chat/completions` request against a DSv4 GGUF returns a coherent completion (greedy, short).

## 4. S3 — Streaming (SSE)

- [ ] 4.1 Add a per-token callback to the resident generate (`...generate_with_prefix_cache_cb(on_token: impl FnMut(u32))`); the buffered path collects, the SSE path emits a chunk per token via `Detokenizer`.
- [ ] 4.2 Wire the chat route's streaming branch (spawn_blocking + mpsc + `Sse`) to the DSv4 callback. EOS/stop handling parity with buffered.
- [ ] 4.3 `#[ignore]` test: streamed token sequence equals the buffered sequence for the same greedy request.

## 5. S4 — Prefix-cache wiring + finish

- [ ] 5.1 Open the per-model `DsV4PrefixCache` under `<server_cache>/dsv4-prefix/<model_id>/` in `DsV4ServeState`; thread it through the generate calls so shared prefixes are reused. Size cap from config.
- [ ] 5.2 Concurrency: confirm the per-model weights lock serializes DSv4 generation (single-flight); document. Resident `layers` shared read-only via `Arc`.
- [ ] 5.3 `make ci` green; traceability regenerated; openspec validate. Docs: README/served-models note on DSv4 GGUF serving + the ~161 GB RAM requirement.
