//! Pluggable KV-cache engines.
//!
//! Each engine implements the full prefill + autoregressive decode loop but
//! manages its persistent inference state differently. Engines are selected
//! via [`EngineKind`] and benched via `larql bench --engine`.
//!
//! Correctness contract: `prefill` and `decode_step` return the pre-lm_head
//! hidden state (shape `[1, hidden_dim]`). The caller applies `final_norm +
//! lm_head` to get logits — see `crate::forward::hidden_to_raw_logits`.

pub mod accuracy;
pub mod kv_engines;
pub mod profiler;
/// Synthetic-weight fixtures (`make_test_weights`, `make_test_vindex`,
/// `make_test_tokenizer`, `TestFixtures`). Used by unit tests, integration
/// tests, and the `mech_interp_demo` example so they don't need a vindex
/// on disk. Released as part of the public API because mech-interp tooling
/// downstream of this crate (chuk-mcp-lazarus and similar) wants the same
/// fixtures for self-contained regression tests.
pub mod test_utils;

// Convenience re-exports so existing `engines::markov_residual::*` paths keep working.
pub use kv_engines::apollo;
pub use kv_engines::markov_residual;
pub use kv_engines::turbo_quant;
pub use kv_engines::unlimited_context;

use crate::model::ModelWeights;
use larql_compute::ComputeBackend;
use ndarray::Array2;

// ─── EngineInfo ───────────────────────────────────────────────────────────────

/// Runtime diagnostics reported by each engine.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Short engine name (e.g. `"markov-rs"`).
    pub name: String,
    /// Human-readable description of the engine's state management strategy.
    pub description: String,
    /// Hardware backend name from [`ComputeBackend::name`]: `"cpu"`, `"metal"`, etc.
    pub backend: String,
    /// Key config parameters (e.g. `"window=512"`), empty string if unconfigured.
    pub config: String,
}

impl EngineInfo {
    pub fn summary(&self) -> String {
        if self.config.is_empty() {
            format!("{} [{}]  {}", self.name, self.backend, self.description)
        } else {
            format!(
                "{} [{}] ({})  {}",
                self.name, self.backend, self.config, self.description
            )
        }
    }
}

// ─── KvEngine trait ───────────────────────────────────────────────────────────

/// Common interface shared by all KV-cache engines.
pub trait KvEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over all prompt tokens.
    /// Returns the hidden state at the final token position (shape `[1, hidden_dim]`).
    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>>;

    /// Run one autoregressive decode step for a single new token.
    /// Returns the hidden state (shape `[1, hidden_dim]`).
    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>>;

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;

    /// Token count in the active hot window (varies by engine type).
    fn window_tokens(&self) -> usize {
        0
    }

    /// Cold-tier bytes (residuals or token IDs past the hot window).
    fn cold_bytes(&self) -> usize {
        0
    }

    /// Per-stage timing summary. Returns `None` if profiling was not enabled.
    fn stage_summary(&self) -> Option<profiler::DecodeStageSummary> {
        None
    }

    /// Prefill using Q4K quantised weights from `index` and `backend`.
    ///
    /// When the backend supports the fused Q4 pipeline (Metal), this routes
    /// through `backend.prefill_q4` for full GPU speed. Falls back to the
    /// f32 path when `backend.has_q4() == false` or `index` has no Q4K data.
    ///
    /// `weights` is `&mut` so the engine can lazily insert dequantised f32
    /// attention tensors into `weights.tensors` on the first call (one-time
    /// cost; subsequent decode steps reuse the cached tensors).
    fn prefill_q4k(
        &mut self,
        weights: &mut crate::model::ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        let _ = (index, backend);
        self.prefill(weights, token_ids) // default: f32 fallback
    }

    /// One autoregressive decode step using Q4K weights.
    ///
    /// Same routing semantics as [`prefill_q4k`]: Metal via `decode_token`
    /// when available, f32 fallback otherwise.
    fn decode_step_q4k(
        &mut self,
        weights: &mut crate::model::ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        let _ = (index, backend);
        self.decode_step(weights, token_id) // default: f32 fallback
    }
}

// ─── EngineKind ───────────────────────────────────────────────────────────────

/// Engine selector. Parse with [`EngineKind::from_name`]; build with [`EngineKind::build`].
#[derive(Debug, Clone)]
pub enum EngineKind {
    MarkovResidual {
        window_size: Option<usize>,
    },
    UnlimitedContext {
        window_size: usize,
    },
    TurboQuant {
        bits: u8,
    },
    Apollo {
        injection_layer: usize,
        inject_coefficient: f32,
        top_k: usize,
    },
}

impl EngineKind {
    /// Parse a CLI engine spec. Accepts `name` or `name:key=value[,key=value]`.
    ///
    /// Examples:
    /// ```text
    /// markov-rs
    /// markov-rs:window=1024
    /// unlimited-context:window=256
    /// turbo-quant:bits=3
    /// tq4
    /// apollo:layer=25,coef=8.0,top_k=12
    /// ```
    pub fn from_name(spec: &str) -> Option<Self> {
        // Split "name:key=val,key=val" into name + param pairs.
        let (name, params_str) = spec.split_once(':').unwrap_or((spec, ""));
        let params: std::collections::HashMap<&str, &str> = params_str
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|kv| kv.split_once('='))
            .collect();

        let get_usize = |key: &str, default: usize| -> usize {
            params
                .get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            params
                .get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };

        match name.trim() {
            "markov-rs" | "markov_rs" | "markov-residual" | "markov_residual" => {
                let window_size = params.get("window").and_then(|v| v.parse().ok());
                Some(EngineKind::MarkovResidual { window_size })
            }
            "unlimited" | "unlimited-context" | "unlimited_context" => {
                Some(EngineKind::UnlimitedContext {
                    window_size: get_usize("window", 512),
                })
            }
            "turbo-quant" | "turbo_quant" | "turboquant" | "tq4" => Some(EngineKind::TurboQuant {
                bits: get_usize("bits", 4) as u8,
            }),
            "tq3" => Some(EngineKind::TurboQuant { bits: 3 }),
            "apollo" => {
                let cfg = apollo::entry::InjectionConfig::default();
                Some(EngineKind::Apollo {
                    injection_layer: get_usize("layer", cfg.injection_layer),
                    inject_coefficient: get_f32("coef", cfg.inject_coefficient),
                    top_k: get_usize("top_k", cfg.top_k),
                })
            }
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            EngineKind::MarkovResidual { .. } => "markov-rs",
            EngineKind::UnlimitedContext { .. } => "unlimited-context",
            EngineKind::TurboQuant { .. } => "turbo-quant",
            EngineKind::Apollo { .. } => "apollo",
        }
    }

    /// Build a boxed engine, dispatching compute through `backend`.
    pub fn build(self, backend: Box<dyn ComputeBackend>) -> Box<dyn KvEngine> {
        self.build_with_profiling(backend, false)
    }

    /// Build a boxed engine with optional per-stage decode profiling.
    pub fn build_with_profiling(
        self,
        backend: Box<dyn ComputeBackend>,
        profiling: bool,
    ) -> Box<dyn KvEngine> {
        match self {
            EngineKind::MarkovResidual { window_size } => Box::new(
                markov_residual::MarkovResidualEngine::with_backend(window_size, backend)
                    .with_profiling(profiling),
            ),
            EngineKind::UnlimitedContext { window_size } => Box::new(
                unlimited_context::UnlimitedContextEngine::with_backend(window_size, backend),
            ),
            EngineKind::TurboQuant { bits } => {
                Box::new(turbo_quant::TurboQuantEngine::with_backend(bits, backend))
            }
            EngineKind::Apollo {
                injection_layer,
                inject_coefficient,
                top_k,
            } => Box::new(apollo::ApolloEngine::new(apollo::InjectionConfig {
                injection_layer,
                inject_coefficient,
                top_k,
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_kind_from_name_roundtrip() {
        for name in &[
            "markov-rs",
            "markov_rs",
            "markov-residual",
            "markov_residual",
        ] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::MarkovResidual { .. })
                ),
                "failed to parse {name:?}"
            );
        }
        for name in &["unlimited", "unlimited-context", "unlimited_context"] {
            assert!(
                matches!(
                    EngineKind::from_name(name),
                    Some(EngineKind::UnlimitedContext { .. })
                ),
                "failed to parse {name:?}"
            );
        }
        assert!(EngineKind::from_name("unknown").is_none());
        assert!(EngineKind::from_name("").is_none());
    }

    #[test]
    fn engine_kind_from_name_with_params() {
        // window param
        match EngineKind::from_name("markov-rs:window=1024") {
            Some(EngineKind::MarkovResidual {
                window_size: Some(1024),
            }) => {}
            other => panic!("expected MarkovResidual{{window=1024}}, got {other:?}"),
        }
        // unlimited window
        match EngineKind::from_name("unlimited-context:window=256") {
            Some(EngineKind::UnlimitedContext { window_size: 256 }) => {}
            other => panic!("expected UnlimitedContext{{window=256}}, got {other:?}"),
        }
        // turbo-quant bits
        match EngineKind::from_name("turbo-quant:bits=3") {
            Some(EngineKind::TurboQuant { bits: 3 }) => {}
            other => panic!("expected TurboQuant{{bits=3}}, got {other:?}"),
        }
        // apollo params
        match EngineKind::from_name("apollo:layer=25,coef=8.0,top_k=12") {
            Some(EngineKind::Apollo {
                injection_layer: 25,
                top_k: 12,
                ..
            }) => {}
            other => panic!("expected Apollo{{layer=25,top_k=12}}, got {other:?}"),
        }
        // unknown param is silently ignored, defaults apply
        match EngineKind::from_name("markov-rs:unknown=999") {
            Some(EngineKind::MarkovResidual { window_size: None }) => {}
            other => panic!("expected MarkovResidual{{window=None}}, got {other:?}"),
        }
    }

    #[test]
    fn engine_info_summary_with_config() {
        let info = EngineInfo {
            name: "markov-rs".into(),
            description: "residual KV".into(),
            backend: "cpu".into(),
            config: "window=512".into(),
        };
        let s = info.summary();
        assert!(s.contains("markov-rs"));
        assert!(s.contains("cpu"));
        assert!(s.contains("window=512"));
    }

    #[test]
    fn engine_info_summary_no_config() {
        let info = EngineInfo {
            name: "test".into(),
            description: "desc".into(),
            backend: "metal".into(),
            config: String::new(),
        };
        let s = info.summary();
        assert!(!s.contains("()"));
    }
}

// ─── Cross-engine trait compliance ───────────────────────────────────────────

#[cfg(test)]
mod compliance_tests {
    use super::*;
    use larql_compute::cpu_backend;

    fn all_kinds() -> Vec<EngineKind> {
        vec![
            EngineKind::MarkovResidual { window_size: None },
            EngineKind::MarkovResidual {
                window_size: Some(32),
            },
            EngineKind::UnlimitedContext { window_size: 64 },
            EngineKind::TurboQuant { bits: 4 },
            EngineKind::TurboQuant { bits: 3 },
            EngineKind::Apollo {
                injection_layer: 30,
                inject_coefficient: 10.0,
                top_k: 8,
            },
        ]
    }

    #[test]
    fn all_engines_memory_zero_before_prefill() {
        for kind in all_kinds() {
            let engine = kind.clone().build(cpu_backend());
            assert_eq!(
                engine.memory_bytes(),
                0,
                "{} should have 0 memory before prefill",
                kind.display_name()
            );
        }
    }

    #[test]
    fn all_engines_have_valid_name() {
        let expected = [
            "markov-rs",
            "markov-rs",
            "unlimited-context",
            "turbo-quant",
            "turbo-quant",
            "apollo",
        ];
        for (kind, expected_name) in all_kinds().into_iter().zip(expected.iter()) {
            let engine = kind.build(cpu_backend());
            assert_eq!(engine.name(), *expected_name);
        }
    }

    #[test]
    fn all_engines_info_has_nonempty_fields() {
        for kind in all_kinds() {
            let name = kind.display_name();
            let engine = kind.build(cpu_backend());
            let info = engine.info();
            assert!(!info.name.is_empty(), "{name}: empty name");
            assert!(!info.backend.is_empty(), "{name}: empty backend");
        }
    }

    #[test]
    fn all_engines_window_tokens_zero_before_prefill() {
        for kind in all_kinds() {
            let engine = kind.clone().build(cpu_backend());
            assert_eq!(
                engine.window_tokens(),
                0,
                "{} window_tokens should be 0 before prefill",
                kind.display_name()
            );
        }
    }

    #[test]
    fn all_engines_cold_bytes_zero_before_prefill() {
        for kind in all_kinds() {
            let engine = kind.clone().build(cpu_backend());
            assert_eq!(
                engine.cold_bytes(),
                0,
                "{} cold_bytes should be 0 before prefill",
                kind.display_name()
            );
        }
    }

    #[test]
    fn all_engines_stage_summary_none_before_decode() {
        for kind in all_kinds() {
            let engine = kind.clone().build_with_profiling(cpu_backend(), true);
            assert!(
                engine.stage_summary().is_none(),
                "{} stage_summary should be None before decode",
                kind.display_name()
            );
        }
    }

    #[test]
    fn from_name_unknown_param_ignored_defaults_apply() {
        match EngineKind::from_name("unlimited-context:unknown=42") {
            Some(EngineKind::UnlimitedContext { window_size: 512 }) => {}
            other => panic!("unknown param should use default, got {other:?}"),
        }
    }

    #[test]
    fn from_name_all_engines_parseable() {
        let specs = [
            ("markov-rs", "markov-rs"),
            ("unlimited-context", "unlimited-context"),
            ("turbo-quant", "turbo-quant"),
            ("tq3", "turbo-quant"),
            ("apollo", "apollo"),
        ];
        for (spec, expected_display) in specs {
            let kind =
                EngineKind::from_name(spec).unwrap_or_else(|| panic!("{spec:?} failed to parse"));
            assert_eq!(
                kind.display_name(),
                expected_display,
                "{spec} parsed to wrong display_name"
            );
        }
    }
}
