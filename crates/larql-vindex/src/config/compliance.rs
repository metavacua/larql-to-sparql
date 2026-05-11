//! Compliance gates + layer-band assignments.
//!
//! - `ComplianceGate` — the self-policing fp4/fp8 quality gate
//!   applied at extract time.
//! - `LayerBands` — per-layer-band classifications (syntax /
//!   knowledge / output) used by DESCRIBE and label matching.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup. `ComplianceGate` carries a `Precision` (defined
//! in the sibling `quantization` module).

use serde::{Deserialize, Serialize};

use super::quantization::Precision;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceGate {
    pub threshold_ratio: f32,
    pub min_compliant_fraction: f32,
    pub fallback_precision: Precision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBands {
    /// Syntax/morphological band (e.g., [0, 13] for Gemma 3 4B).
    pub syntax: (usize, usize),
    /// Knowledge/factual band (e.g., [14, 27] for Gemma 3 4B).
    pub knowledge: (usize, usize),
    /// Output/formatting band (e.g., [28, 33] for Gemma 3 4B).
    pub output: (usize, usize),
}

impl LayerBands {
    /// Known-good layer bands for supported model families.
    /// Returns None if the family isn't recognised — caller should fall back
    /// to treating all layers as a single band.
    pub fn for_family(family: &str, num_layers: usize) -> Option<Self> {
        let last = num_layers.saturating_sub(1);
        match (family, num_layers) {
            // Gemma family — validated via probe analysis
            ("gemma3", 34) => Some(Self {
                syntax: (0, 13),
                knowledge: (14, 27),
                output: (28, 33),
            }),
            ("gemma3", 42) => Some(Self {
                syntax: (0, 16),
                knowledge: (17, 34),
                output: (35, 41),
            }),
            ("gemma2", 26) => Some(Self {
                syntax: (0, 10),
                knowledge: (11, 20),
                output: (21, 25),
            }),
            ("gemma2", 42) => Some(Self {
                syntax: (0, 16),
                knowledge: (17, 34),
                output: (35, 41),
            }),
            ("gemma2", 46) => Some(Self {
                syntax: (0, 18),
                knowledge: (19, 37),
                output: (38, 45),
            }),

            // Gemma 4 family
            ("gemma4", 30) => Some(Self {
                syntax: (0, 11),
                knowledge: (12, 23),
                output: (24, 29),
            }),
            ("gemma4", 36) => Some(Self {
                syntax: (0, 14),
                knowledge: (15, 28),
                output: (29, 35),
            }),
            ("gemma4", 35) => Some(Self {
                syntax: (0, 13),
                knowledge: (14, 27),
                output: (28, 34),
            }),
            ("gemma4", 60) => Some(Self {
                syntax: (0, 23),
                knowledge: (24, 47),
                output: (48, 59),
            }),

            // Llama family
            ("llama", 32) => Some(Self {
                syntax: (0, 12),
                knowledge: (13, 25),
                output: (26, 31),
            }),
            ("llama", 40) => Some(Self {
                syntax: (0, 15),
                knowledge: (16, 32),
                output: (33, 39),
            }),
            ("llama", 80) => Some(Self {
                syntax: (0, 31),
                knowledge: (32, 63),
                output: (64, 79),
            }),

            // Mistral / Mixtral
            ("mistral", 32) => Some(Self {
                syntax: (0, 12),
                knowledge: (13, 25),
                output: (26, 31),
            }),
            ("mixtral", 32) => Some(Self {
                syntax: (0, 12),
                knowledge: (13, 25),
                output: (26, 31),
            }),

            // Qwen
            ("qwen2", 28) => Some(Self {
                syntax: (0, 10),
                knowledge: (11, 22),
                output: (23, 27),
            }),
            ("qwen2", 32) => Some(Self {
                syntax: (0, 12),
                knowledge: (13, 25),
                output: (26, 31),
            }),
            ("qwen2", 40) => Some(Self {
                syntax: (0, 15),
                knowledge: (16, 32),
                output: (33, 39),
            }),
            ("qwen2", 64) => Some(Self {
                syntax: (0, 25),
                knowledge: (26, 51),
                output: (52, 63),
            }),
            ("qwen2", 80) => Some(Self {
                syntax: (0, 31),
                knowledge: (32, 63),
                output: (64, 79),
            }),

            // Phi
            ("phi", 32) => Some(Self {
                syntax: (0, 12),
                knowledge: (13, 25),
                output: (26, 31),
            }),
            ("phi", 40) => Some(Self {
                syntax: (0, 15),
                knowledge: (16, 32),
                output: (33, 39),
            }),

            // GPT-2 (smaller, denser)
            ("gpt2", 12) => Some(Self {
                syntax: (0, 4),
                knowledge: (5, 9),
                output: (10, 11),
            }),
            ("gpt2", 24) => Some(Self {
                syntax: (0, 9),
                knowledge: (10, 19),
                output: (20, 23),
            }),
            ("gpt2", 36) => Some(Self {
                syntax: (0, 14),
                knowledge: (15, 28),
                output: (29, 35),
            }),
            ("gpt2", 48) => Some(Self {
                syntax: (0, 19),
                knowledge: (20, 38),
                output: (39, 47),
            }),

            // Fallback: estimate from layer count
            // ~40% syntax, ~40% knowledge, ~20% output
            _ if num_layers >= 8 => {
                let syntax_end = num_layers * 2 / 5;
                let knowledge_end = num_layers * 4 / 5;
                Some(Self {
                    syntax: (0, syntax_end.saturating_sub(1)),
                    knowledge: (syntax_end, knowledge_end.saturating_sub(1)),
                    output: (knowledge_end, last),
                })
            }

            // Too few layers to band meaningfully
            _ => None,
        }
    }

    /// Check which band a layer belongs to.
    pub fn band_for_layer(&self, layer: usize) -> &'static str {
        if layer >= self.syntax.0 && layer <= self.syntax.1 {
            "syntax"
        } else if layer >= self.knowledge.0 && layer <= self.knowledge.1 {
            "knowledge"
        } else if layer >= self.output.0 && layer <= self.output.1 {
            "output"
        } else {
            "unknown"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma3_34_layer_bands() {
        let b = LayerBands::for_family("gemma3", 34).unwrap();
        assert_eq!(b.syntax, (0, 13));
        assert_eq!(b.knowledge, (14, 27));
        assert_eq!(b.output, (28, 33));
    }

    #[test]
    fn llama_32_layer_bands() {
        let b = LayerBands::for_family("llama", 32).unwrap();
        assert_eq!(b.syntax, (0, 12));
        assert_eq!(b.knowledge, (13, 25));
        assert_eq!(b.output, (26, 31));
    }

    #[test]
    fn unknown_family_with_sufficient_layers_uses_fallback() {
        let b = LayerBands::for_family("custom_model", 20);
        assert!(b.is_some(), "should fall back to fraction-based estimate");
        let b = b.unwrap();
        // Bands partition [0, 19] into syntax/knowledge/output
        assert!(b.syntax.1 < b.knowledge.0);
        assert!(b.knowledge.1 < b.output.0);
        assert_eq!(b.output.1, 19);
    }

    #[test]
    fn unknown_family_does_not_inherit_known_bands_by_string_prefix() {
        // The fallback path is layer-count driven, not name driven.
        // String prefixes like "gemma3-finetune" or "llama-clone" must
        // NOT pick up the curated bands for the canonical family.
        let gemma = LayerBands::for_family("gemma3", 34).unwrap();
        let llama = LayerBands::for_family("llama", 32).unwrap();

        let gemma_lookalike = LayerBands::for_family("gemma3-clone", 34).unwrap();
        assert_ne!(
            gemma_lookalike.knowledge, gemma.knowledge,
            "fallback must not inherit canonical gemma3 knowledge band by name prefix"
        );

        let llama_lookalike = LayerBands::for_family("llamafied", 32).unwrap();
        assert_ne!(
            llama_lookalike.knowledge, llama.knowledge,
            "fallback must not inherit canonical llama knowledge band by name prefix"
        );

        // The fraction-based fallback is structurally distinct: 2/5 syntax,
        // 4/5 knowledge cutoff. For 32 layers that's syntax=(0, 11),
        // knowledge=(12, 24), which is one layer off from canonical llama.
        assert_eq!(llama_lookalike.syntax, (0, 11));
        assert_eq!(llama_lookalike.knowledge, (12, 24));
    }

    #[test]
    fn too_few_layers_returns_none() {
        assert!(LayerBands::for_family("gpt2", 4).is_none());
        assert!(LayerBands::for_family("tiny", 1).is_none());
    }

    #[test]
    fn band_for_layer_gemma3() {
        let b = LayerBands::for_family("gemma3", 34).unwrap();
        assert_eq!(b.band_for_layer(0), "syntax");
        assert_eq!(b.band_for_layer(13), "syntax");
        assert_eq!(b.band_for_layer(14), "knowledge");
        assert_eq!(b.band_for_layer(27), "knowledge");
        assert_eq!(b.band_for_layer(28), "output");
        assert_eq!(b.band_for_layer(33), "output");
    }

    #[test]
    fn band_for_layer_out_of_range_is_unknown() {
        let b = LayerBands {
            syntax: (0, 5),
            knowledge: (6, 10),
            output: (11, 15),
        };
        assert_eq!(b.band_for_layer(99), "unknown");
    }

    #[test]
    fn layer_bands_serde_round_trip() {
        let b = LayerBands::for_family("gemma3", 34).unwrap();
        let j = serde_json::to_string(&b).unwrap();
        let back: LayerBands = serde_json::from_str(&j).unwrap();
        assert_eq!(back.syntax, b.syntax);
        assert_eq!(back.knowledge, b.knowledge);
        assert_eq!(back.output, b.output);
    }

    #[test]
    fn compliance_gate_serde_round_trip() {
        use crate::config::quantization::Precision;
        let gate = ComplianceGate {
            threshold_ratio: 16.0,
            min_compliant_fraction: 0.99,
            fallback_precision: Precision::Fp8,
        };
        let j = serde_json::to_string(&gate).unwrap();
        let back: ComplianceGate = serde_json::from_str(&j).unwrap();
        assert_eq!(back.threshold_ratio, 16.0);
        assert_eq!(back.min_compliant_fraction, 0.99);
        assert_eq!(back.fallback_precision, Precision::Fp8);
    }

    #[test]
    fn gpt2_12_layer_bands() {
        let b = LayerBands::for_family("gpt2", 12).unwrap();
        assert_eq!(b.syntax, (0, 4));
        assert_eq!(b.knowledge, (5, 9));
        assert_eq!(b.output, (10, 11));
    }
}
