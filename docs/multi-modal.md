# Multi-Modal Support in LARQL

> **Status:** Phase 0+1 shipped (PR #143, 2026-05-24). Phase 2 shipped
> (PR #144, 2026-05-25). Phases 3–6 remain design-only.

LARQL supports embed-splice vision input for Gemma 3 (SigLIP, Phase 1)
and Granite Vision (SigLIP2 + MLP GELU + AnyRes tiling, Phase 2). This
note maps the original design and what it would take to extend to
accept vision and audio inputs in a way that survives across model
families — Gemma 3/4, Llama 3.2 Vision, Qwen2-VL / Qwen2-Audio,
Granite Vision — without quietly pinning the architecture to whichever
model we happen to try first.

The load-bearing claim: **vindex is already modality-agnostic** (it
stores residual-shaped vectors keyed by `(layer, feature_index)`), so
the real seam is the **embedding boundary** in
`crates/larql-inference/src/forward/trace.rs:33` and
`crates/larql-compute/src/forward/embed.rs:11`, which today hard-assumes
`tokens → embed table → blocks`. The harder question is whether one
seam is enough — across the four model families we'd want to support,
there are at least three structurally different ways of getting a
non-text modality into a decoder.

---

## TL;DR

| Family               | Vision encoder | Connector            | Integration pattern        | Position scheme        |
| -------------------- | -------------- | -------------------- | -------------------------- | ---------------------- |
| Gemma 3 (4B/12B/27B) | SigLIP         | Linear projection    | **Embed-splice** (256/img) | Sequential, 1D RoPE    |
| Gemma 4 (audio)      | USM-like       | Linear projection    | Embed-splice               | Sequential, 1D RoPE    |
| Granite Vision 3.2   | SigLIP2        | 2-layer MLP (GELU)   | **Embed-splice** (per-tile)| Sequential, 1D RoPE    |
| Qwen2-VL / 2.5-VL    | ViT (custom)   | MLP                  | **Embed-splice** (dynamic) | **M-RoPE** (t, h, w)   |
| Qwen2-Audio          | Whisper        | Linear               | Embed-splice               | Sequential, 1D RoPE    |
| Llama 3.2 Vision     | ViT-H          | (none — cross-attn)  | **Cross-attention layers** | LM stays text-position |

Three integration patterns fall out:

1. **Embed-splice** (Gemma 3/4, Granite, Qwen-VL/Audio, LLaVA family) —
   encoder output is projected to LM hidden size and inserted into the
   input sequence at placeholder positions. The LM forward pass is
   structurally unchanged.
2. **Cross-attention** (Llama 3.2 Vision) — encoder output stays in
   encoder dim; the LM gains *new layer types* (cross-attention blocks
   inserted between self-attention layers) that read from it. The
   forward pass changes; vindex storage assumptions for "layers" change.
3. **Prefix-only** (BLIP-2-era; not strictly needed but trivially
   subsumed by embed-splice with placeholder-at-position-0).

A v1 multi-modal LARQL covering embed-splice gets us Gemma 3, Granite,
Qwen-VL, Gemma 4 audio, Qwen2-Audio — five model families across two
modalities — without touching the layer graph. **Llama 3.2 Vision is
deliberately deferred**, and the framing matters: adding cross-attention
support later is **not a trait extension, it is a forward-pass refactor**.
The current per-layer loop in `forward/trace.rs` assumes one layer type
and one KV cache; Llama 3.2 Vision interleaves cross-attention blocks
at fixed positions (every 4th LM layer) that read from a *separate* KV
cache derived from the vision encoder. v1 trait surface is sized for
embed-splice. When Phase 6 lands, the load-bearing change is making the
layer loop polymorphic over layer type and making KV cache a set, not a
single buffer — adding a new `MultiModalProtocol` variant is the easy
part. Saying this explicitly so we don't, in six months, mistake the
protocol extension for the whole job.

A second seam — **position encoding** — is forced by Qwen-VL's M-RoPE
even within the embed-splice family. We need positions to be a typed
plan, not a `Range<usize>`.

---

## Goals and non-goals

**In scope (v1):**

- Accept image and audio file paths through the CLI and Python bindings.
- Load a vision/audio encoder (initially: SigLIP for Gemma 3, ViT for
  Qwen-VL) and project its output into the LM's residual dim.
- Splice encoder embeddings into the input sequence at placeholder
  positions emitted by the tokenizer / chat template.
- Generate text conditioned on the multi-modal input. `lm_head` is
  unchanged: text out only.
- Make the trait surface broad enough that adding a third encoder
  family (Whisper, USM) is *new architecture code only*, not changes
  to the forward pass.

**Out of scope (v1):**

- Llama 3.2 Vision cross-attention integration. Tracked separately;
  v1 trait should not block it but does not implement it.
- Image / audio *output*. No diffusion head, no audio decoder.
- Compiling encoder weights into vindex. Encoders are opaque
  safetensors at boot.
- LQL surface changes (`SELECT … WHERE image = …`). Multi-modal
  enters at `larql run`, not yet at the query layer.
- Quantization of encoder weights. v1 runs encoders in f16/f32 only.
- Metal kernels for the vision tower. v1 is CPU encoder, Metal LM —
  matches how most local MM runtimes start.

---

## What "cross-architecture" actually means here

We already have `ModelArchitecture` (`crates/larql-models/src/config/architecture.rs:49`)
abstracting per-family LM details (tensor keys, norm type, activation,
RoPE scaling, per-layer geometry). That trait does its job — adding
Granite or Qwen didn't require forward-pass changes.

The multi-modal equivalent needs to abstract:

1. **Encoder family** — SigLIP vs. SigLIP2 vs. ViT vs. Whisper vs. USM.
   Each has its own tensor layout, patch/frame extraction, position
   embed scheme, and activation function. This is the direct analogue
   of `ModelArchitecture` and wants its own trait.
2. **Connector** — linear, MLP-GELU, MLP-SiLU, sometimes pixel-shuffle
   (Granite). Always small. Family-specific but trivially parameterized.
3. **Integration pattern** — embed-splice vs. cross-attention. Embed-
   splice can be one shared implementation. Cross-attention can't.
4. **Placeholder protocol** — every embed-splice model has its own
   marker scheme:
   - Gemma 3: `<start_of_image>` + 256 × `<image_soft_token>` + `<end_of_image>`.
   - Granite: `<image>` token per tile; AnyRes tiling at the host.
   - Qwen-VL: `<|vision_start|>` + N × `<|image_pad|>` + `<|vision_end|>`,
     where N depends on image resolution.
   - LLaVA: `<image>` token, expanded host-side.
   This is a property of the *model family*, not the encoder. It lives
   on the LM-side `ModelArchitecture`.
5. **Position scheme** — sequential vs. M-RoPE. This is also LM-side;
   Qwen2-VL's text-only sibling already uses M-RoPE, so the trait
   change is forced regardless of MM.
6. **Vision token budget** — fixed (Gemma 3: 256; Granite per tile: 729)
   vs. dynamic (Qwen-VL). KV-cache sizing must be computed *after* the
   encoder has run, not from token count alone.

A single set of design choices satisfies all of (1–6) for the embed-
splice family. The trait surface below is sized for that.

---

## Current state of the seams

From the survey (file:line references are authoritative — read them
before editing this section):

### `larql-models` — LM architecture abstraction

- `ModelArchitecture` trait at
  `crates/larql-models/src/config/architecture.rs:49`.
- Per-family impls in `crates/larql-models/src/architectures/`:
  `gemma3.rs`, `gemma4.rs`, `granite.rs`, `qwen.rs`, `llama.rs`,
  `mistral.rs`, `mixtral.rs`, `deepseek_v4.rs`, `gpt_oss.rs`, ...
- Gemma 4 already strips `model.language_model.` from tensor keys
  (`crates/larql-models/src/architectures/gemma4.rs:102–109`) — Google's
  wrapper prefix when a model is shipped as multimodal. This is the
  *only* multimodal hint in the codebase today; no encoder code exists.

### `larql-vindex` — storage

- `VectorIndex` at
  `crates/larql-vindex/src/index/core/mod.rs:48–76` is dimension-agnostic:
  `(num_layers, hidden_size)` parametric, with `gate` / `ffn` substores
  keyed by `(layer, feature_index)`.
- Storage handle trait `VindexStorage`
  (`crates/larql-vindex/src/index/storage/vindex_storage/mod.rs`) is
  opaque at byte level; mmap / Redis / S3 backends already coexist.
- **Nothing about vindex assumes text tokens**. Residual vectors are
  residual vectors. ✓ No changes required for v1.

### `larql-inference` — forward pass

- Embedding lookup in
  `crates/larql-compute/src/forward/embed.rs:11–24`: `tokens: &[u32]`
  → `Array2<f32>` of shape `(seq_len, hidden_size)`, scaled by
  `arch.embed_scale()`.
- Forward trace in
  `crates/larql-inference/src/forward/trace.rs:31–62` calls
  `embed_tokens()` once at line 33, then loops layers.
- No abstraction exists for "pre-built embeddings." This is the
  primary surgery site.

### `larql-cli` / `larql-python` — input

- `RunArgs` at `crates/larql-cli/src/commands/primary/run_cmd.rs:72`
  takes `prompt: Option<String>`. No file inputs.
- Python `Session` at `crates/larql-python/src/session.rs:62–76` wraps
  LQL text queries.
- ChatTemplate handles Gemma / Mistral / Llama / ChatML / Plain
  (referenced from `docs/virtual-experts-dispatch.md`). Multi-modal
  chat templates (Gemma 3's image markers, Qwen-VL's vision markers,
  Granite's `<image>`) would extend this layer.

---

## Proposed trait surface

The goal of the trait sketch is to make the *embed-splice* path one
shared implementation and the *cross-attention* path a future
extension that doesn't require ripping out v1.

### `Encoder` trait (new, in `larql-models`)

```rust
pub trait Encoder: Send + Sync {
    /// Stable family name, e.g. "siglip", "siglip2", "qwen2-vit", "whisper".
    fn family(&self) -> &str;

    /// Hidden size produced by the encoder *before* projection.
    fn encoder_hidden_size(&self) -> usize;

    /// Run the encoder on raw input bytes for one item.
    /// Returns (seq_len, encoder_hidden_size) — variable seq_len allowed.
    fn encode(&self, input: ModalInput) -> Result<Array2<f32>>;
}

pub enum ModalInput<'a> {
    Image(ImageBytes<'a>),  // RGB; encoder family decides patching.
    Audio(AudioFrames<'a>), // 16 kHz mono mel; encoder family decides framing.
}
```

Encoders are LM-agnostic. SigLIP is SigLIP whether it feeds Gemma 3
or PaliGemma.

### `Connector` trait (new, in `larql-models`)

```rust
pub trait Connector: Send + Sync {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

    /// Project encoder output into LM hidden size.
    fn project(&self, encoder_out: &Array2<f32>) -> Array2<f32>;
}
```

Concrete impls: `LinearConnector`, `MlpGeluConnector`,
`MlpSiluPixelShuffleConnector` (Granite). Always small, always part of
the LM-side weights (loaded with the LM, not the encoder).

### `ModelArchitecture` additions

```rust
trait ModelArchitecture {
    // ... existing methods ...

    /// Modality hooks. Default `None` — text-only model.
    fn multimodal(&self) -> Option<&dyn MultiModalProtocol> { None }
}

pub trait MultiModalProtocol: Send + Sync {
    /// Encoder family this LM was trained against.
    fn vision_encoder(&self) -> Option<&str>;  // e.g. Some("siglip")
    fn audio_encoder(&self) -> Option<&str>;

    /// Placeholder token id(s) that the host must replace with embeddings.
    fn image_placeholder(&self) -> Option<PlaceholderProtocol>;
    fn audio_placeholder(&self) -> Option<PlaceholderProtocol>;

    /// How many placeholder positions does one image occupy?
    fn image_token_budget(&self) -> TokenBudget;

    /// Whether `Precomputed` embeddings should pass through `arch.embed_scale()`
    /// before splicing. Two cases in practice:
    ///   - `None`: connector output is final. The connector is responsible for
    ///     any modality-specific scaling, baked into `project()`.
    ///   - `SameAsTokens`: apply the same scale used for token embeddings
    ///     (e.g. Gemma's sqrt(hidden_size)) on top of connector output.
    /// We deliberately do NOT expose a `Custom(f32)` case — bare scalars on
    /// the protocol have to be remembered by every implementer, fail silently
    /// when copied, and the right home for any model-specific scalar is
    /// inside the connector's `project()` (which already owns LM hidden size
    /// and travels with the LM weights).
    fn precomputed_scaling(&self) -> PrecomputedScaling;

    /// Discrete tile counts that this model's AnyRes accepts. Empty for
    /// non-tiling models. Granite Vision ships a fixed grid set —
    /// picking a count outside it breaks placeholder accounting.
    fn valid_tile_counts(&self) -> &[usize] { &[] }
}

pub struct PlaceholderProtocol {
    pub start: Option<u32>,   // e.g. Gemma 3's <start_of_image>
    pub fill:  u32,           // e.g. Gemma 3's <image_soft_token>
    pub end:   Option<u32>,   // e.g. Gemma 3's <end_of_image>
}

pub enum TokenBudget {
    /// One image = exactly N placeholder positions. Gemma 3 = 256.
    Fixed(usize),
    /// One tile = exactly N positions; tile count is host-side AnyRes choice.
    /// Granite Vision 3.2 = 729 per tile. Host owns the tiling grid.
    PerTile { tokens_per_tile: usize },
    /// Encoder decides at run time based on input shape. Qwen-VL.
    /// Host must run the encoder before sizing the input sequence.
    Dynamic,
}

pub enum PrecomputedScaling {
    /// `Precomputed` rows go in as-is. Connector's `project()` owns any
    /// modality-specific scaling.
    None,
    /// Apply the same `arch.embed_scale()` used for token embeddings.
    SameAsTokens,
}
```

Per-family impls — `Gemma3MultiModal` returns SigLIP + token budget
256 + Gemma's three-token sandwich; `Qwen2VlMultiModal` returns
`qwen2-vit` + dynamic budget + Qwen's two-marker scheme; `LlamaVision`
returns *cross-attention pattern* (a separate enum variant we add when
that pattern lands).

### `EmbeddingPlan` (new, in `larql-inference`)

This is the surgery in the forward pass. Replace `embed_tokens(&[u32])`
with:

```rust
pub enum EmbeddingChunk {
    /// Standard token-id lookup. Embed scaling is applied per `arch.embed_scale()`.
    Tokens(Vec<u32>),
    /// Pre-computed embeddings to splice in at this position.
    /// Contract: **ready to concatenate** — connector has been applied,
    /// any modality-specific scaling has been applied by the host according
    /// to `MultiModalProtocol::precomputed_scaling()`. The embed step does
    /// NOT re-scale these rows. This separation exists because the
    /// "do we scale vision embeddings the same as token embeddings" decision
    /// is checkpoint-specific and we don't want it implicit in two places.
    ///
    /// `modality` is load-bearing for `PositionScheme::Mrope`: M-RoPE
    /// advances different position axes (t, h, w) depending on the chunk's
    /// modality. For `PositionScheme::Sequential` it is telemetry-only.
    /// Drop it if M-RoPE wiring proves it isn't needed.
    Precomputed { rows: Array2<f32>, modality: Modality },
}

pub struct EmbeddingPlan {
    pub chunks: Vec<EmbeddingChunk>,
    pub positions: PositionScheme,
}

pub enum PositionScheme {
    Sequential,                     // Gemma 3, Granite, most.
    Mrope { axes: MropeAxes },      // Qwen-VL.
}

pub fn embed_plan(weights: &Weights, arch: &dyn ModelArchitecture,
                  plan: &EmbeddingPlan) -> Array2<f32> { ... }
```

The forward trace calls `embed_plan(...)` instead of `embed_tokens(...)`.
Text-only paths build a one-chunk plan; multi-modal paths build a
multi-chunk plan upstream from CLI / chat template.

### Host-side plumbing

- `MultiModalInput` type at the CLI/Python layer carries text + image
  paths + audio paths.
- `prepare_plan(input, tokenizer, arch, encoders, connector)`:
  - tokenize text with placeholders;
  - for each image / audio item, run `encoder.encode()` then
    `connector.project()`;
  - build the `EmbeddingPlan` by replacing placeholder spans with
    `Precomputed` chunks.

---

## Two seams, recap

| Seam                      | Where                                                                  | Why it changes                       |
| ------------------------- | ---------------------------------------------------------------------- | ------------------------------------ |
| **Embedding boundary**    | `forward/trace.rs:33`, `forward/embed.rs:11`                           | Multi-source embeddings; placeholder splicing. |
| **Position encoding**     | `forward/rope.rs` (everywhere RoPE is applied)                         | Qwen-VL M-RoPE — positions become tuples. |

vindex, layer graph, KV cache shape, lm_head, sampler — all unchanged.

---

## Phased rollout

**Phase 0 — trait + plumbing, no encoder code. SHIPPED (PR #143, 2026-05-24).** Land `Encoder`,
`Connector`, `MultiModalProtocol`, `EmbeddingPlan`, `PositionScheme::Mrope`
(behind a stub). Re-route `forward/trace.rs` through `embed_plan(...)`.
Text-only tests stay green; Shannon gate still passes. *No model
behavior changes.* This is the load-bearing PR — once it lands, every
subsequent encoder is additive.

**Phase 1 — Gemma 3 4B + SigLIP, prefix-only. SHIPPED (PR #143, 2026-05-24).** Smallest interesting
end-to-end. Load SigLIP from safetensors, project, prepend to text.
`larql run --image foo.jpg "describe"`. CPU encoder, Metal LM. Validates
the `Encoder` and `Connector` trait surface against one concrete model
*without* exercising mid-sequence splicing. Expected outcome:
PaliGemma-quality captions. Gemma 3 specifically because we already
have its zone map characterised — see Open Question #11. Time: ~1 week.

**Phase 2 — Granite Vision (SigLIP2 + MLP connector + AnyRes). SHIPPED (PR #144, 2026-05-25).**
*This is the splice stress test, not Phase 3.* Granite's per-tile
mechanism puts **N splice points per image** (one per AnyRes tile)
into the input sequence, where Gemma 3's `<start_of_image>` sandwich
puts only one. If `EmbeddingPlan`'s splice machinery has a bug,
Granite finds it; Gemma interleaving would not. Granite also forces
the `TokenBudget::PerTile` path and the host-side AnyRes tiler. Time:
~1 week incremental.

**Phase 3 — Gemma 3 native interleaving = the chat-template phase.**
Splice machinery is proven from Phase 2, but Phase 1 and Phase 2 both
sidestep tokenizer-side placeholder emission (Phase 1 prefixes
everything; Phase 2 can prefix-glue tiles per image). Phase 3 is the
first phase that genuinely interleaves text and image *mid-sequence*,
which forces the tokenizer-ownership question (Open Question:
ChatTemplate-vs-pre-pass) to be answered. That is the real work here
— not splice, not encoder, but where placeholder emission lives. Don't
under-budget this phase on the basis of "splice already works."

**Phase 4 — Qwen2-VL.** Forces M-RoPE through the pipeline and forces
`TokenBudget::Dynamic`. This is the trait-surface stress test for
positions — if Phase 0's `PositionScheme` was wrong, we find out here.
Time: ~1 week (encoder) + an unknown amount fixing M-RoPE.

**Phase 5 — Audio.** Gemma 4 audio path (USM-style encoder) and/or
Qwen2-Audio (Whisper-style). Same splice machinery, different encoder
family. Time: ~1 week per encoder family once a reference impl exists.

**Phase 6 (separate decision) — Llama 3.2 Vision.** Cross-attention
integration. New `MultiModalProtocol` variant for cross-attn pattern is
the *easy* part — the load-bearing change is the forward-pass refactor:
making the layer loop polymorphic over layer type, splitting KV-cache
into a set keyed by `(layer, cache_source)`, and routing encoder output
as a second KV stream. This is its own design doc and very likely its
own ADR; it is not a continuation of v1 in any meaningful sense beyond
sharing the encoder/connector code from earlier phases.

---

## GPU / Metal encoder acceleration

> **Status:** Not Phase 2 scope. Written to inform the Metal encoder
> follow-up PR after Phase 2 merges.

### Current state

Phase 1 and Phase 2 both use CPU-only vision encoders (SigLIP, SigLIP2).
The production inference pipeline is:

```
CPU encoder → CPU connector → embed_plan (CPU) → Metal LM prefill → Metal decode
```

This matches how most local multi-modal runtimes start. The CPU→GPU
boundary is at `prefill_from_hidden()`, which receives the fully-spliced
`Array2<f32>` initial hidden state and runs the LM layers on Metal.

### Metal encoder feasibility

The existing Metal shader inventory in `crates/larql-compute-metal/src/shaders/`
already covers most of the SigLIP/SigLIP2 encoder pipeline:

| Encoder op | Existing shader | Gap |
| --- | --- | --- |
| Patchify (Conv2D as matmul) | `sgemm` / `sgemm_transb` | None — reshape + matmul |
| Position embedding add | Element-wise add | Trivial |
| LayerNorm (scale + bias) | `layer_norm` | None |
| Biased QKV projections | `sgemm` + bias add | None |
| Bidirectional attention | `causal_attention` | **Mask removal** — existing shader applies triangular mask; encoder needs the same GEMM sequence minus the mask |
| Output projection | `sgemm` + bias add | None |
| GELU activation | `activation` (gelu_tanh) | None for SigLIP; SiLU variant may be needed for some SigLIP2 configs |
| MLP (fc1 → act → fc2) | `sgemm` + activation + `sgemm` | None |
| Post-LayerNorm | `layer_norm` | None |

**Single structural gap:** bidirectional attention. The current
`causal_attention.rs` shader applies `if col > row { -inf }` before
softmax. SigLIP's attention is identical except without that mask line.
This is a shader variant (remove the conditional), not a new kernel
architecture.

### Memory budget

| Encoder | Params | f32 | f16 |
| --- | --- | --- | --- |
| SigLIP (Gemma 3 4B) | ~400M | ~1.6 GB | ~800 MB |
| SigLIP2 (Granite Vision) | ~400M | ~1.6 GB | ~800 MB |

Both fit comfortably in Apple Silicon unified memory alongside the LM
weights. Quantization of encoder weights is deferred (v1 runs f16/f32
only) but the memory headroom is not a blocker.

### Expected speedup

The SigLIP forward pass on Gemma 3 4B-it (27 layers, 4096 patches,
hidden=1152) takes ~30s on CPU (from real-checkpoint test). Metal
should bring this to 1–3s based on the GPU matmul throughput ratio
observed in the LM path. This is the primary user-visible performance
improvement — the LM path is already Metal-accelerated.

### Zero-copy pipeline

The target architecture eliminates the GPU→CPU→GPU round-trip:

```
Metal encoder GPU buffer → Metal connector GPU buffer
  → Metal embed_plan splice → Metal LM prefill
```

This requires `embed_plan` to accept GPU-resident `Precomputed` chunks
(currently `Array2<f32>` on the host). The actual Metal encoder seam
change is making `EmbeddingChunk::Precomputed` carry either a host
array or a Metal buffer handle. The host-side `decode_and_tile` /
`decode_and_resize_square` remains CPU (image decoding is not
GPU-worthy at these resolutions).

### AnyRes + Metal interaction

For Granite Vision, each tile runs through the encoder independently.
Metal encoder enables parallel tile processing via sequential tile
dispatch on Metal's command queue — the N+1 tiles per image (base +
detail) can pipeline without CPU round-trips between tiles.

### Recommended sequencing

1. Phase 2 ships CPU encoder (unchanged).
2. Metal encoder is a follow-up PR after Phase 2 merges. Start with
   SigLIP since Gemma 3 is the established baseline with a working
   CPU reference; SigLIP2 follows trivially once the bidirectional
   attention shader variant exists.
3. The bidirectional attention shader variant is the first deliverable
   — it unblocks the entire Metal encoder pipeline.
4. Metal connector (MLP GELU, Gemma projector) is trivially composed
   from existing `sgemm` + activation shaders once the encoder
   outputs are GPU-resident.

---

## Open questions

1. **Where do encoder weights live?**
   - **Option A**: opaque safetensors, loaded into a sibling
     `EncoderWeights` struct, never compressed. Simplest. vindex stays
     LM-only.
   - **Option B**: encoder weights *also* in a vindex with a different
     prefix (e.g. `vision/`). Lets you slice / probe / compile encoder
     FFN slots later. More work, unclear payoff in v1.
   - **Recommendation**: A for v1. Revisit if there's a research
     motivation for compiling encoder FFNs (there might be — SigLIP's
     vision FFN is a candidate for the same vindex compilation
     experiments LARQL runs on LM FFNs).

2. **Connector weights — where?**
   - Connector is small and ships with the LM (Gemma 3 4B includes it
     in `model-*.safetensors`). Load it alongside LM weights. No
     vindex compilation.

3. **Tokenizer / chat template scope.**
   - Each MM model has its own image marker convention. ChatTemplate
     needs a `multimodal` extension that knows how to emit placeholders
     in the right places given a `MultiModalInput`. Cleanest place:
     extend the existing auto-detection chat-template code.

4. **Backend support.**
   - Encoders are matmul + LayerNorm + softmax. CPU works. Metal is a
     stretch goal — not v1. Worth checking whether existing Metal
     GEMM kernels in `larql-compute-metal` are reusable for a SigLIP
     forward pass (they probably are, with shape adjustments).

5. **KV cache sizing — a sequencing constraint, not an audit.**
   This one is architectural. Today: `tokenize → size cache → forward`.
   With MM: `tokenize → encode → project → splice → size cache → forward`.
   The encoder run cannot be lazy, because cache allocation depends on
   its output length (true for `TokenBudget::Dynamic`, and also for
   `PerTile` once the AnyRes choice is made). Any code path that
   currently allocates the KV cache before having seen all inputs —
   streaming prefill is the obvious candidate — has to change shape,
   not just gain an audit. Worth identifying these sites in Phase 0
   even though the encoder isn't wired up yet, because the sequencing
   contract is what the trait surface is encoding.

6. **Quantization of encoder weights.**
   - Defer. f16 fits comfortably for SigLIP-So400m at ~400M params
     (≈800 MB f16). Worth a check at Phase 2 start: confirm what
     Granite Vision 3.2 actually ships. If it's SigLIP2-giant or
     larger, the comfort margin shrinks and Phase 2 inherits a
     memory question that Phase 1 didn't have. Flag before starting
     Phase 2, not after.

7. **Llama 3.2 Vision deferral — for how long?**
   - Cross-attention is a structurally different problem. If the use
     case is "support every modality across every model," it's
     mandatory. If it's "give LARQL eyes and ears for research,"
     embed-splice covers Gemma 3 / Granite / Qwen / audio without it.
     Flag as a non-blocking gap.

8. **LQL surface.**
   - Out of scope for v1, but worth a sketch: `SELECT * FROM facts
     WHERE image LIKE '<path>'` would need a stable embedding of the
     image as the key. That's research territory — the moment LARQL
     has eyes, "compile this image as a key" becomes an experiment.

9. **Test strategy.**
   - End-to-end: a tiny golden corpus of (image, expected caption)
     and (audio, expected transcript) pairs per encoder family, run
     in CI. Slow tests, ungated by default.
   - Shannon-style: extend `larql shannon verify` with a multi-modal
     variant — same bits/char correctness instrument, but with image
     context. Probably a Phase 2 deliverable.

10. **Naming.**
    - `Encoder` is overloaded (we already have things called
      encoders). `ModalEncoder` is clearer. `Connector` is the
      LLaVA convention; some papers say "adapter" or "projector." Pick
      one before any code lands.

11. **Does the vindex choke point map survive cross-modal input?**
    This is a research question, not a v1 blocker, but it's worth
    naming so we don't lose it. The L4 commit / L26 retrieval-reopening
    structure was measured on text-only Gemma 3 4B. Adding 256 vision
    tokens to the prefix is a real perturbation.

    **Falsifiable form** (shape of the measurement, not the bounds):
    *With 256 SigLIP tokens prefixed to each of an N-prompt corpus,
    measure (a) the L4 polysemantic-routing PR shift relative to
    text-only baseline and (b) the fraction of prompts whose L4
    dispatch cell remains identical to text-only. Choke-point map
    is modality-stable if PR drops by <X% and dispatch-cell-match
    >Y%. If either bound breaks on Gemma 3 4B, the dispatch trait
    needs a modality axis and the vindex zone story is text-specific.*
    Commit to (X, Y) the day Phase 1 lands and we can actually run
    the probe; the point of writing the shape now is so the question
    is concrete enough to slot into a sprint instead of perennially
    "interesting."

---

## What this does *not* commit us to

- It does not commit to compiling vision FFNs into vindex.
- It does not commit to image / audio output.
- It does not commit to Metal encoder kernels.
- It does not commit to cross-attention integration.
- It does not commit to LQL multi-modal queries.
- It does not commit to supporting every MM model — just to making
  the trait surface broad enough that adding one is bounded work.

## What this *does* commit us to (if we proceed)

- A real change to `forward/trace.rs` and `forward/embed.rs`. The
  `embed_tokens(&[u32]) -> Array2<f32>` signature goes away in favour
  of `embed_plan(&EmbeddingPlan) -> Array2<f32>`. Every call site
  updates.
- A real change to RoPE application paths. M-RoPE means positions are
  no longer always `0..seq_len`.
- A new crate dependency (image decoding, audio resampling) — or a
  small in-tree implementation.

---

## Decisions (resolved)

The four decision points from the first revision are settled:

1. **Embed-splice-first is the v1 scope.** Llama 3.2 Vision is
   deferred — and not just at the trait level. See TL;DR framing:
   Phase 6 is a forward-pass refactor (polymorphic layer loop, KV
   cache as a set), not a protocol extension.
2. **Phase 1 model = Gemma 3 4B + SigLIP, prefix-only.** Chosen
   specifically because we already have its zone map characterised
   (Open Question #11) — the existing measurement baseline means we
   can detect when MM input perturbs the choke-point structure.
3. **Phase 0 lands as a single PR** — trait + plumbing + routed embed
   path. Splitting invites a half-landed state where the text path
   uses both old and new code paths, which is exactly the kind of
   state that breaks the Shannon gate without telling us why.
   **Acceptance criterion**: Shannon gate green *and*
   `embed_plan(EmbeddingPlan { chunks: vec![Tokens(toks)], positions: Sequential })`
   produces **bit-identical** output to today's `embed_tokens(toks)` on
   the existing text corpus. That bit-identity is the contract that
   makes the single-PR shape safe; without it we are trusting that
   nothing got perturbed in the re-route, which is exactly the kind
   of trust that decays silently over a year.
4. **CPU-only encoder for v1.** Metal port is a separate experiment
   that benefits from having a working baseline to measure against.

## Open questions for the next session

The previous "four decisions" are settled; what's actually unresolved:

- **Phase 0 PR sequencing inside the single PR.** Trait first then
  re-route, or both at once? Probably both at once given that the
  trait is the contract being tested.
- **Tokenizer ownership of placeholder emission.** ChatTemplate is
  the natural place, but tokenizers don't currently know about image
  paths. Either ChatTemplate becomes MM-aware, or there's a pre-pass
  that resolves placeholders before the chat template runs.
- **Where the AnyRes tiler lives.** Granite-specific code in
  `larql-models/architectures/granite_vision.rs`, or a shared host
  utility? Argues for shared — Qwen-VL's dynamic patching also wants
  resolution-aware tiling logic and they'll share more than they
  diverge. But the shared utility is **not pure-shared**: it has to
  consult the model's `valid_tile_counts()` to pick a count the
  placeholder protocol can actually account for. Shared mechanics,
  per-model config.
- **Resolution of Open Question #5** (KV cache sequencing). Identify
  the call sites in Phase 0 even though encoder isn't wired — the
  sequencing contract is the thing the trait encodes.
