## Why

The DSv4-Flash inference engine is now feature-complete in
`larql-inference` — resident-Q4_K weights (P1–P8), CPU-FFN/GPU-attention
hybrid placement, and the on-disk prefix cache (7.3× prefill reuse) — but
it is **unreachable**: the OpenAI chat route rejects it
(`routes/openai/chat.rs:~762`, "DeepSeek V4 Flash ... not yet available").
None of the shipped work can actually serve a request.

The blocker is a format mismatch. `larql-server` serves models **from a
vindex** (`load_single_vindex`; qwen35 reconstructs its weights from the
vindex on first use). But the entire DSv4 engine reads a **GGUF
directly** (`load_dsv4_resident_layers(gguf, ...)`), and there is **no
DSv4 vindex extraction** — `larql-vindex` has no deepseek-v4 path.
Building one would be a separate, large project.

So the pragmatic, constraint-determined route is to serve DSv4 **directly
from its GGUF**: a GGUF-backed model entry whose resident-quantized layers
are loaded once into the server process and driven by the existing
`dsv4_resident_generate_with_prefix_cache`. This is what makes the
DSv4-Flash showcase real on this box (the project's driving goal:
CPU-FFN / GPU-attention, model > VRAM, served).

## What Changes

DSv4-only, additive — no existing (vindex) serving path changes.

- **GGUF-backed model entry**: a server model can point at a DSv4-Flash
  `.gguf`. `LoadedModel` gains an `OnceLock<DsV4ServeState>` (mirroring
  the existing `qwen35_weights: OnceLock<...>`): on the first DSv4
  request it opens the GGUF, extracts `DsV4Hyperparams`, and loads the
  resident-Q4_K layers + head once (~161 GB RAM for Flash), reused across
  all later requests.
- **Chat-route DSv4 branch**: the `arch_family == "deepseek_v4"`
  rejection becomes a generation branch — the same shape as the qwen35
  branch — that tokenizes via the existing tokenizer, runs
  `dsv4_resident_generate_with_prefix_cache`, and detokenizes via the
  existing `Detokenizer`.
- **Streaming**: add a per-token callback to the resident generate (the
  decode loop already produces tokens one at a time) so the SSE path can
  emit a chunk per token; the buffered path collects them.
- **Prefix cache wiring**: the per-model on-disk `DsV4PrefixCache`
  (already built) is opened under the server's cache dir, keyed by the
  model id, so shared prompt prefixes are reused across requests.

## Capabilities

### New Capabilities
- `dsv4-server-serving`: `larql-server` serves DeepSeek-V4-Flash directly
  from a GGUF — loading its resident-quantized layers once into the
  process and driving the OpenAI chat route (buffered + SSE streaming)
  through the DSv4 resident generate + on-disk prefix cache. Covers the
  GGUF-backed load/residency, the chat-route DSv4 branch, token
  streaming, and prefix-cache reuse.

### Modified Capabilities
None. The existing `server-attention-service` (vindex serving) is
unchanged; this adds a parallel GGUF-direct path used only for the
`deepseek_v4` arch.
