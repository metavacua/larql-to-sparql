//! Normalisation kinds and the scope their statistic reduces over.

/// Normalization type used by the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// RMSNorm (Gemma, Llama)
    RmsNorm,
    /// Standard LayerNorm (GPT-2, BERT)
    LayerNorm,
}

/// The vector over which a QK-norm's RMS statistic is reduced.
///
/// Both variants apply the *same* weight vector elementwise across the whole
/// projection. They differ only in the denominator, and that difference is not
/// visible in the tensor names — OLMoE and Qwen3 both store
/// `self_attn.{q,k}_norm.weight`, and `transformers`' own Qwen3 module marks
/// the distinction in a comment (`# unlike olmo, only on the head dim!`)
/// rather than in a type. It is not always visible in the shapes either:
/// OLMoE-1B-7B is MHA with `num_heads * head_dim == hidden_size`, so its
/// `[2048]` weight is exactly as wide as a per-head convention would imply if
/// you only counted elements.
///
/// Getting it wrong rescales every head to a common norm, which discards the
/// relative magnitude *between* heads — a structural change to attention, not
/// a rounding one. See `docs/k3-funnel.md` §4.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNormScope {
    /// RMS over each head's `head_dim` slice independently.
    /// `Qwen3RMSNorm(head_dim)` — Qwen3, Gemma 3/4.
    PerHead,
    /// RMS over the entire `num_heads * head_dim` projection, before any
    /// reshape into heads. `OlmoeRMSNorm(hidden_size)` — OLMoE.
    FullProjection,
}
