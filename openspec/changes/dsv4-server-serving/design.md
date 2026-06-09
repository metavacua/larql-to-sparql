# Design — dsv4-server-serving

## Context

`larql-server` request path (from the integration sweep):
- `routes/openai/chat.rs::run_chat_completion` → `pick_template` /
  `render` → tokenize (`model.tokenizer.encode`) → lock weights
  (`lock_weights_for_gen`) → branch on `weights.arch.family()`.
- qwen35 branches to `qwen35_generate_with_sampling` (owns its loop);
  the generic path uses `layer_graph::generate_with_sampling`.
- `arch_family == "deepseek_v4"` → hard rejection (chat.rs:~762).
- `LoadedModel` (state.rs:21) holds `id`, `path` (vindex dir),
  `tokenizer`, lazy `weights: OnceLock<RwLock<ModelWeights>>`, and the
  arch-specific `qwen35_weights: OnceLock<Arc<Qwen35Weights>>`.
- Server is **vindex-only**; no GGUF is ever opened. DSv4 has **no
  vindex extraction**; its engine reads GGUF.

DSv4 engine entry points (all public, GGUF-based): `DsV4Hyperparams::
from_gguf`, `load_dsv4_resident_layers`, `load_dsv4_head`,
`dsv4_resident_generate_with_prefix_cache`, `DsV4PrefixCache::open`.

## Goals / Non-Goals

**Goals (v1):** serve DSv4-Flash from a GGUF through the OpenAI chat route
(buffered + streaming), resident-loaded once, with prefix-cache reuse.

**Non-goals (v1):** DSv4 vindex extraction; constrained / JSON-schema
(`response_format`) generation for DSv4; `/v1/completions`; multiple
concurrent DSv4 models; quantizing the served model further.

## Decisions

### D1 — Serve from GGUF, not vindex

DSv4 is GGUF-only and has no vindex path; building one is out of scope.
A DSv4 model entry carries a GGUF path. **Alternative considered:** build
deepseek-v4 vindex extraction so DSv4 serves through the standard path —
a large separate project; deferred. The GGUF-direct path is isolated to
the `deepseek_v4` arch, so the vindex path is untouched.

How the GGUF path reaches the server: the model's vindex config (or a new
config field) records the source GGUF. The server still loads the vindex
for the tokenizer + arch detection (it already does), and additionally
opens the GGUF for DSv4 weights. (If a DSv4 entry has no vindex, a
minimal config providing tokenizer + `arch=deepseek_v4` + gguf path is
the fallback — detailed in tasks S1.)

### D2 — Resident state in a `OnceLock`, loaded once

`LoadedModel` gains `dsv4: OnceLock<Arc<DsV4ServeState>>` where
`DsV4ServeState { layers, hp, head, prefix_cache: Mutex<DsV4PrefixCache> }`.
First DSv4 request loads it (open GGUF → `from_gguf` → `load_dsv4_
resident_layers` → `load_dsv4_head`); ~161 GB RAM for Flash, held for the
process lifetime — mirrors the qwen35 `OnceLock` precedent (which notes a
30–40 s one-time vindex→weights cost). Gate behind host-RAM: if the load
fails (OOM), return a clear error, don't crash.

### D3 — Per-token streaming callback

The resident decode loop already samples + forwards one token at a time.
Add a callback variant `dsv4_resident_generate_with_prefix_cache_cb(...,
on_token: impl FnMut(u32))` (or thread an `Option<&mut dyn FnMut>`), so
the SSE path fires a chunk per token via the existing `Detokenizer`
(`seed(prompt_ids)` + `push(id)`), exactly like `generate_streaming`. The
buffered path passes a collector. EOS + stop-string handling reuse the
route's existing logic over the streamed ids.

### D4 — Prefix cache under the server cache dir

Open one `DsV4PrefixCache` per DSv4 model at
`<server_cache>/dsv4-prefix/<model_id>/`, wrapped in a `Mutex` inside
`DsV4ServeState` (the cache is `&mut` on use; requests are already
serialized by the per-model weights lock, so contention is nil). The
model id salts the hash (already supported). Size cap from config
(default a few GB).

## Risks

- **~161 GB resident** in the server process — only hosts with enough RAM
  can serve Flash. Document the requirement; fail loudly on OOM.
- **~2.5 s/tok decode** — slow for interactive serving; acceptable for
  the research/showcase use case (the whole point is *running* DSv4 on
  pre-AMX hardware, not low latency). The prefix cache mitigates *prefill*
  cost, not per-token decode.
- **GGUF-path plumbing** is the least-settled piece (D1): how a DSv4
  entry declares its GGUF. S1 nails this down before any forward wiring;
  keep it config-driven and DSv4-gated so nothing else is affected.
- **Concurrency**: the resident layers are read-only and shareable
  (`Arc`); the prefix cache + KV caches are per-request — the existing
  per-model write lock already serializes generation, so v1 is
  single-flight per model (fine; document it).
