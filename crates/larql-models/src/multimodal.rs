//! Multi-modal trait surface — Phase 0 of multi-modal support.
//!
//! See `docs/multi-modal.md` for the design rationale. This module
//! defines the *shape* of multi-modal support without committing to
//! any encoder or model-family implementation. Every type here is
//! a trait or a small data type; no behaviour lives in this module.
//!
//! Phase 0 acceptance criterion is that none of this affects existing
//! text-only models — every `ModelArchitecture` impl gets a default
//! `multimodal()` that returns `None`, and the text path is structurally
//! unchanged.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use ndarray::Array2;

// ─── Modality tag ────────────────────────────────────────────────────────

/// What kind of data a chunk of embeddings was derived from.
///
/// Used in two places:
///   1. As context on a precomputed embedding chunk so the position
///      scheme (`PositionScheme::Mrope` in particular) knows which RoPE
///      axes to advance.
///   2. As telemetry on storage / routing analytics so MoE histograms
///      and vindex slot statistics can be sliced by modality.
///
/// Drop this enum if M-RoPE wiring (Phase 4) shows it isn't needed
/// — the embed-splice path doesn't strictly require it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
}

// ─── Modal encoder trait ─────────────────────────────────────────────────

/// Raw bytes for one modal item, pre-encoder.
///
/// Encoders own their own preprocessing (patching, framing, mel
/// extraction). The host hands over decoded bytes plus minimal shape
/// metadata; the encoder decides what to do with them.
pub enum ModalInput<'a> {
    /// Decoded RGB image, row-major, three channels. The encoder
    /// decides patch size and any resampling.
    Image {
        rgb: &'a [u8],
        width: usize,
        height: usize,
    },
    /// Mono PCM audio at 16 kHz, f32 samples in `[-1.0, 1.0]`.
    /// The encoder decides framing and mel-spectrogram parameters.
    Audio { samples: &'a [f32] },
}

/// A modality-specific encoder. Produces a `(seq_len, encoder_hidden_size)`
/// matrix of embeddings before the connector projects them into the LM's
/// hidden space.
///
/// Encoders are LM-agnostic: SigLIP is SigLIP whether the downstream LM
/// is Gemma 3 or PaliGemma. Per-LM behaviour belongs on `MultiModalProtocol`.
pub trait ModalEncoder: Send + Sync {
    /// Stable family name (e.g. "siglip", "siglip2", "qwen2-vit",
    /// "whisper", "usm"). Used for compatibility checks against
    /// `MultiModalProtocol::vision_encoder()` / `audio_encoder()`.
    fn family(&self) -> &str;

    /// Hidden size produced by the encoder *before* the connector
    /// projects to LM hidden size.
    fn encoder_hidden_size(&self) -> usize;

    /// Run the encoder on one modal item.
    ///
    /// Returns `(seq_len, encoder_hidden_size)`. `seq_len` is variable
    /// — see `TokenBudget::Dynamic` for the host-side accounting.
    ///
    /// Errors are encoder-defined; Phase 0 leaves the error type
    /// unspecified (`String` for now, replace with a typed enum the
    /// first time we need to discriminate).
    fn encode(&self, input: ModalInput<'_>) -> Result<Array2<f32>, String>;
}

// ─── Connector trait ─────────────────────────────────────────────────────

/// Projects encoder output into the LM's hidden size.
///
/// Implementations: linear (PaliGemma), 2-layer MLP with GELU (LLaVA,
/// Granite), MLP + pixel shuffle (Granite Vision 4.1), identity
/// (early-fusion models like Qwen3.5 where the "encoder" is already in
/// LM-space).
///
/// Any *modality-specific scaling* of the projected embeddings (e.g.
/// applying `sqrt(hidden_size)` for Gemma-style models that scale text
/// tokens) belongs *inside* `project()`, not as a bare scalar on the
/// protocol. See `docs/multi-modal.md` for why — we deliberately do
/// not expose a `PrecomputedScaling::Custom(f32)` case.
pub trait Connector: Send + Sync {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

    /// Project a `(seq_len, input_dim)` matrix to `(seq_len, output_dim)`.
    /// Output is the final, scaled embedding ready to splice into the
    /// LM input sequence.
    fn project(&self, encoder_out: &Array2<f32>) -> Array2<f32>;
}

// ─── Placeholder protocol ────────────────────────────────────────────────

/// Token IDs that signal "an encoded modal item belongs here."
///
/// Conventions vary per model family:
///   - Gemma 3: `<start_of_image>`, then N × `<image_soft_token>`, then `<end_of_image>`
///   - Granite Vision 4.1: `<image>` token per AnyRes tile
///   - Qwen3-VL: `<|vision_start|>`, then N × `<|image_pad|>`, then `<|vision_end|>`
///   - LLaVA: `<image>` token, expanded host-side
///
/// `start` and `end` are sentinel markers (often LM family-specific
/// special tokens). `fill` is the per-position marker that the host
/// replaces with one row of `Precomputed` embedding.
#[derive(Debug, Clone)]
pub struct PlaceholderProtocol {
    pub start: Option<u32>,
    pub fill: u32,
    pub end: Option<u32>,
}

// ─── Token budget ────────────────────────────────────────────────────────

/// How many placeholder positions one modal item consumes in the
/// LM input sequence.
///
/// Three distinct mechanisms in the wild:
///   - `Fixed(N)`: every image expands to exactly N placeholders.
///     Gemma 3 = 256.
///   - `PerTile { tokens_per_tile: N }`: AnyRes-style tiling. The host
///     chooses how many tiles based on input resolution; each tile
///     contributes exactly N placeholders. Granite Vision 4.1.
///   - `Dynamic`: the encoder decides at runtime based on input shape.
///     Qwen3-VL's "naive dynamic resolution" — placeholder count is
///     known only *after* encoding.
///
/// `Dynamic` and `PerTile` both have the same downstream consequence
/// (KV cache cannot be sized before encoder runs) but the upstream
/// host code differs: `PerTile` requires an AnyRes tiler, `Dynamic`
/// doesn't.
#[derive(Debug, Clone, Copy)]
pub enum TokenBudget {
    Fixed(usize),
    PerTile { tokens_per_tile: usize },
    Dynamic,
}

// ─── Precomputed scaling ─────────────────────────────────────────────────

/// Whether host-side splicing should re-scale `Precomputed` embeddings
/// by `arch.embed_scale()` before they enter the LM.
///
/// Two cases in practice; we deliberately omit `Custom(f32)` —
/// modality-specific scalars belong inside the `Connector`, not on
/// this enum. See `docs/multi-modal.md` for the rationale (a bare
/// scalar fails silently when copied across model impls).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecomputedScaling {
    /// Connector output is final; embed step does NOT re-scale.
    None,
    /// Apply the same `arch.embed_scale()` used for token embeddings.
    SameAsTokens,
}

// ─── MultiModalProtocol trait ────────────────────────────────────────────

/// Per-LM-family multi-modal contract.
///
/// Lives behind `ModelArchitecture::multimodal()`. Returns `None` for
/// text-only LMs (the default), `Some` for Gemma 3 / Granite Vision /
/// Qwen-VL / etc. The trait describes *what the LM expects from the
/// host*, not how the encoder runs — encoders are LM-agnostic and
/// referenced by family name.
pub trait MultiModalProtocol: Send + Sync {
    /// Encoder family the LM was trained against, by name. The host
    /// loads the corresponding `ModalEncoder` impl and verifies that
    /// `encoder.family() == self.vision_encoder().unwrap()` before
    /// wiring them together.
    fn vision_encoder(&self) -> Option<&str> {
        None
    }
    fn audio_encoder(&self) -> Option<&str> {
        None
    }

    /// Placeholder convention for image inputs. `None` if the LM does
    /// not accept images.
    fn image_placeholder(&self) -> Option<PlaceholderProtocol> {
        None
    }
    /// Placeholder convention for audio inputs. `None` if the LM does
    /// not accept audio.
    fn audio_placeholder(&self) -> Option<PlaceholderProtocol> {
        None
    }

    /// How many placeholder positions one image expands to. Default
    /// `Dynamic`; LMs with fixed budgets should override.
    fn image_token_budget(&self) -> TokenBudget {
        TokenBudget::Dynamic
    }

    /// How `Precomputed` embedding chunks should be scaled at the
    /// splice site. Default `None` — connector output is final.
    fn precomputed_scaling(&self) -> PrecomputedScaling {
        PrecomputedScaling::None
    }

    /// Discrete tile counts the host's AnyRes tiler may pick from.
    /// Empty for non-tiling models; non-empty for `TokenBudget::PerTile`
    /// LMs. The host MUST select a count from this list — picking
    /// outside it breaks placeholder accounting.
    fn valid_tile_counts(&self) -> &[usize] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    //! Default-impl coverage for the `MultiModalProtocol` trait. A minimal
    //! `EmptyProtocol` (overrides nothing) exercises every default body —
    //! the LM-family-specific impls (Gemma3MultiModal, etc.) live in their
    //! own architecture modules and test their *overrides*; this module
    //! tests the *defaults*.
    use super::*;

    struct EmptyProtocol;
    impl MultiModalProtocol for EmptyProtocol {}

    #[test]
    fn default_vision_and_audio_encoders_are_none() {
        let p = EmptyProtocol;
        assert!(p.vision_encoder().is_none());
        assert!(p.audio_encoder().is_none());
    }

    #[test]
    fn default_placeholders_are_none() {
        let p = EmptyProtocol;
        assert!(p.image_placeholder().is_none());
        assert!(p.audio_placeholder().is_none());
    }

    #[test]
    fn default_token_budget_is_dynamic() {
        let p = EmptyProtocol;
        assert!(matches!(p.image_token_budget(), TokenBudget::Dynamic));
    }

    #[test]
    fn default_precomputed_scaling_is_none() {
        let p = EmptyProtocol;
        assert_eq!(p.precomputed_scaling(), PrecomputedScaling::None);
    }

    #[test]
    fn default_valid_tile_counts_is_empty() {
        let p = EmptyProtocol;
        assert!(p.valid_tile_counts().is_empty());
    }

    #[test]
    fn token_budget_variants_construct_cleanly() {
        // Confirms PerTile field is reachable + Debug formatting works for
        // error messages that interpolate the variant.
        let fixed = TokenBudget::Fixed(256);
        let per_tile = TokenBudget::PerTile {
            tokens_per_tile: 729,
        };
        let dynamic = TokenBudget::Dynamic;
        assert!(format!("{fixed:?}").contains("256"));
        assert!(format!("{per_tile:?}").contains("729"));
        assert!(format!("{dynamic:?}").contains("Dynamic"));
    }

    #[test]
    fn modality_equality_and_hash() {
        use std::collections::HashSet;
        assert_eq!(Modality::Text, Modality::Text);
        assert_ne!(Modality::Image, Modality::Audio);
        let set: HashSet<_> = [Modality::Text, Modality::Image, Modality::Image]
            .iter()
            .collect();
        assert_eq!(set.len(), 2, "Modality must dedupe under Hash");
    }

    #[test]
    fn placeholder_protocol_round_trips_clone() {
        let p = PlaceholderProtocol {
            start: Some(255_999),
            fill: 262_144,
            end: Some(256_000),
        };
        let q = p.clone();
        assert_eq!(p.start, q.start);
        assert_eq!(p.fill, q.fill);
        assert_eq!(p.end, q.end);
    }

    #[test]
    fn modal_input_image_carries_dimensions() {
        let bytes = vec![0u8; 12];
        let input = ModalInput::Image {
            rgb: &bytes,
            width: 2,
            height: 2,
        };
        match input {
            ModalInput::Image { width, height, .. } => {
                assert_eq!((width, height), (2, 2));
            }
            _ => panic!("expected Image variant"),
        }
    }

    #[test]
    fn modal_input_audio_carries_samples_ref() {
        let samples = [0.0f32; 16];
        let input = ModalInput::Audio { samples: &samples };
        match input {
            ModalInput::Audio { samples } => assert_eq!(samples.len(), 16),
            _ => panic!("expected Audio variant"),
        }
    }
}
