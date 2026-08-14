//! Shared numerical defaults for model architectures.
//!
//! These are the values HuggingFace transformers uses when an HF config omits
//! the corresponding field. They live in one module so the parser
//! (`detect/parser.rs`), the trait defaults (`config.rs`), and per-architecture
//! fallbacks (`architectures/gemma{3,4}.rs`, etc.) all reference the same
//! number — preventing the kind of drift that, before 2026-05-16, had
//! `rms_norm_eps` defaulting to 1e-6 in inference even when the model's
//! config.json explicitly specified 1e-5.

/// Default RoPE theta for Gemma-family models when `rope_theta` is absent.
/// Matches HF `Gemma3TextConfig.rope_theta` class default.
pub const ROPE_BASE_GEMMA: f64 = 1_000_000.0;

/// Default RoPE theta for non-Gemma families and the per-layer
/// `rope_local_base` fallback (Gemma 3 sliding layers use this).
/// Matches HF `LlamaConfig.rope_theta` class default.
pub const ROPE_BASE_DEFAULT: f64 = 10_000.0;

/// Norm-epsilon fallback for the **majority** of families when the config
/// omits `rms_norm_eps` / `layer_norm_eps` / `layer_norm_epsilon` /
/// `norm_epsilon`. Matches the `transformers` class default for Llama,
/// Mistral, Qwen3 (+ MoE), Gemma 2/3 and GraniteMoE.
///
/// It is not universal — see [`DEFAULT_NORM_EPS_1E5`] and
/// `ModelArchitecture::default_norm_eps`, which is the accessor to read.
pub const DEFAULT_NORM_EPS: f32 = 1e-6;

/// Norm-epsilon fallback for the families whose `transformers` config class
/// defaults to 1e-5: **OLMoE, GPT-OSS, StarCoder2, Phi-3**.
///
/// An order of magnitude apart from [`DEFAULT_NORM_EPS`], and it matters: a
/// checkpoint that omits the field is served at *its own* class default, so
/// serving OLMoE at 1e-6 is a systematic error in every norm of every layer.
/// Measured cost on `allenai/OLMoE-1B-7B-0924-Instruct` (which omits it):
/// final-residual cosine 0.890 vs the HF reference, 0.991 once corrected.
pub const DEFAULT_NORM_EPS_1E5: f32 = 1e-5;

// ── Llama-3 `rope_scaling` class defaults ───────────────────────────────
// Llama-3.x configs ship the full four-field set, but synthetic / partial
// configs (test fixtures, custom checkpoints) may omit individual fields.
// These match HF's class defaults in `Llama3TextConfig` so a partial
// llama3 rope_scaling block resolves identically on both sides.

/// Default `low_freq_factor` for HF `rope_scaling = {rope_type: llama3}`.
pub const LLAMA3_LOW_FREQ_FACTOR_DEFAULT: f64 = 1.0;

/// Default `high_freq_factor` for HF `rope_scaling = {rope_type: llama3}`.
pub const LLAMA3_HIGH_FREQ_FACTOR_DEFAULT: f64 = 4.0;

/// Default `original_max_position_embeddings` for HF `rope_scaling =
/// {rope_type: llama3}`. Llama-3 was pre-trained at 8K context before
/// long-context fine-tuning.
pub const LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS_DEFAULT: f64 = 8192.0;

// ── YaRN `rope_scaling` class defaults ──────────────────────────────────
// HF's `_compute_yarn_parameters` falls back to the values proposed in the
// YaRN paper when a config omits them, so a checkpoint that ships a partial
// yarn block is still fully determined. These mirror that fallback.

/// Default `beta_fast` for HF `rope_scaling = {rope_type: yarn}` — the number
/// of rotations above which a dimension is left *unscaled* (extrapolated).
pub const YARN_BETA_FAST: f64 = 32.0;

/// Default `beta_slow` for HF `rope_scaling = {rope_type: yarn}` — the number
/// of rotations below which a dimension is fully *interpolated* (÷ factor).
pub const YARN_BETA_SLOW: f64 = 1.0;

/// Default `truncate` for HF `rope_scaling = {rope_type: yarn}`.
///
/// When true the correction range is rounded outward to whole dimensions
/// (`floor(low)` / `ceil(high)`); when false it stays fractional and the ramp
/// starts and ends part-way through a dimension. `openai/gpt-oss-20b` ships
/// `false`, so this default is *not* the value that model runs under — which
/// is exactly why it is read rather than assumed.
pub const YARN_TRUNCATE: bool = true;
