//! [`FfnBackendKind`] — the FFN-axis enum mirroring
//! [`larql_kv::EngineKind`] for the KV axis.

use crate::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// FFN backend selector. Parse with [`Self::from_name`].
///
/// **Kept distinct from `Walk { k: None }` and `Dense`** even though
/// they produce bit-identical output at K=None (proven April 2026,
/// bit-identical across all 34 layers on Gemma 3 4B and 4 E2B). The
/// distinction matters for diagnosing regressions: if a future kernel
/// change breaks bit-identity, the accuracy JSON's `ffn_backend`
/// column tells you which code path actually ran ([`crate::ffn::WeightFfn`]
/// matmul vs [`crate::vindex::WalkFfn`] saxpy), so the diff is
/// one-axis instead of two. The "honest about what ran" framing
/// argues *for* keeping them separate, not against.
///
/// Exhaustive — adding a variant later is a deliberate schema change
/// that breaks every consumer until updated. **Do not add
/// `#[non_exhaustive]`**; defaulting new variants into existing arms
/// reproduces the silent-drop problem one layer down.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FfnBackendKind {
    /// Dense matmul via [`crate::ffn::WeightFfn`]. All three FFN
    /// matmuls (gate, up, down) computed directly from weight
    /// tensors. Reference path for correctness.
    Dense,
    /// [`crate::vindex::WalkFfn`]. `k = None` is the dense walk
    /// (bit-identical to [`Self::Dense`] at full K); `k = Some(n)` is
    /// sparse top-K, computing only `n` features instead of all of
    /// them. The April K-sweep showed K=100 sufficient at the
    /// retrieval layers (L14–27) with no parametric degradation.
    Walk { k: Option<usize> },
    /// [`crate::ffn::RemoteWalkBackend`]. Dispatches FFN computation
    /// to a remote shard over the wire. The `wire` field is the
    /// unparsed wire-preference spec — a `WirePreference` parser
    /// lives in [`crate::ffn::remote`] and is applied at build time,
    /// not parse time.
    RemoteWalk {
        endpoint: String,
        /// Unparsed wire-preference spec (e.g. `"q4k,bf16"`).
        /// `None` means "let the backend pick its default."
        wire: Option<String>,
    },
    /// [`crate::ffn::NullFfn`]. Returns zeros; debug-only.
    Null,
}

impl FfnBackendKind {
    /// Parse a CLI FFN spec. Accepts `name` or `name:key=value[,key=value]`.
    /// Returns `None` on unrecognised names; **unrecognised parameter
    /// keys are silently ignored** (matches `EngineKind::from_name`
    /// behaviour).
    ///
    /// Examples:
    /// ```text
    /// dense
    /// walk
    /// walk:k=100
    /// walk:k=none                   (k=None form)
    /// remote-walk:endpoint=http://shard:8080
    /// remote-walk:endpoint=http://shard:8080,wire=q4k
    /// null
    /// ```
    pub fn from_name(spec: &str) -> Option<Self> {
        let (name, params_str) = spec.split_once(':').unwrap_or((spec, ""));
        let params: HashMap<&str, &str> = params_str
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|kv| kv.split_once('='))
            .collect();

        match name.trim() {
            "dense" | "weight" | "weights" => Some(FfnBackendKind::Dense),
            "walk" => {
                let k = match params.get("k") {
                    None => None,
                    Some(v) if v.eq_ignore_ascii_case("none") => None,
                    Some(v) => match v.parse::<usize>() {
                        Ok(n) => Some(n),
                        Err(_) => return None,
                    },
                };
                Some(FfnBackendKind::Walk { k })
            }
            "remote-walk" | "remote_walk" => {
                let endpoint = params.get("endpoint")?.to_string();
                let wire = params.get("wire").map(|s| (*s).to_string());
                Some(FfnBackendKind::RemoteWalk { endpoint, wire })
            }
            "null" | "off" => Some(FfnBackendKind::Null),
            _ => None,
        }
    }

    /// Canonical name. Matches the names returned by
    /// [`Self::supported_names`].
    pub fn display_name(&self) -> &'static str {
        match self {
            FfnBackendKind::Dense => "dense",
            FfnBackendKind::Walk { .. } => "walk",
            FfnBackendKind::RemoteWalk { .. } => "remote-walk",
            FfnBackendKind::Null => "null",
        }
    }

    /// All backend names recognised by [`Self::from_name`]. Single
    /// source of truth for CLI help text, mirroring
    /// [`larql_kv::EngineKind::supported_names`].
    ///
    /// Aliases (`weight`/`weights` for `dense`, `remote_walk` for
    /// `remote-walk`, `off` for `null`) are intentionally omitted —
    /// the help text shows the recommended spelling.
    pub fn supported_names() -> &'static [&'static str] {
        &["dense", "walk", "remote-walk", "null"]
    }
}

/// Distinguish "unrecognised backend name" from "known backend with
/// malformed params" — both surface as `None` from
/// [`FfnBackendKind::from_name`], but the policy-level error layer
/// should distinguish them so users get a useful message. Lives in
/// this module because the classification logic is backend-aware
/// (knows the supported names + aliases).
pub(super) fn classify_backend_parse_error(spec: &str) -> super::PolicyParseError {
    let (name, _) = spec.split_once(':').unwrap_or((spec, ""));
    let name = name.trim();
    if FfnBackendKind::supported_names().contains(&name)
        || matches!(name, "weight" | "weights" | "remote_walk" | "off")
    {
        super::PolicyParseError::MalformedBackend {
            spec: spec.to_string(),
        }
    } else {
        super::PolicyParseError::UnknownBackend {
            name: name.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_dense_aliases() {
        for name in &["dense", "weight", "weights"] {
            assert_eq!(
                FfnBackendKind::from_name(name),
                Some(FfnBackendKind::Dense),
                "{name:?} should parse to Dense"
            );
        }
    }

    #[test]
    fn from_name_walk_default_k_is_none() {
        assert_eq!(
            FfnBackendKind::from_name("walk"),
            Some(FfnBackendKind::Walk { k: None })
        );
    }

    #[test]
    fn from_name_walk_with_k_parses_integer() {
        assert_eq!(
            FfnBackendKind::from_name("walk:k=100"),
            Some(FfnBackendKind::Walk { k: Some(100) })
        );
    }

    #[test]
    fn from_name_walk_with_k_none_explicit() {
        for spec in &["walk:k=none", "walk:k=None", "walk:k=NONE"] {
            assert_eq!(
                FfnBackendKind::from_name(spec),
                Some(FfnBackendKind::Walk { k: None }),
                "{spec:?} should parse to Walk{{k=None}}"
            );
        }
    }

    #[test]
    fn from_name_walk_with_invalid_k_returns_none() {
        assert!(FfnBackendKind::from_name("walk:k=abc").is_none());
    }

    #[test]
    fn from_name_remote_walk_requires_endpoint() {
        assert!(FfnBackendKind::from_name("remote-walk").is_none());
        assert!(FfnBackendKind::from_name("remote-walk:wire=q4k").is_none());

        assert_eq!(
            FfnBackendKind::from_name("remote-walk:endpoint=http://shard:8080"),
            Some(FfnBackendKind::RemoteWalk {
                endpoint: "http://shard:8080".to_string(),
                wire: None,
            })
        );
        assert_eq!(
            FfnBackendKind::from_name("remote-walk:endpoint=http://shard:8080,wire=q4k"),
            Some(FfnBackendKind::RemoteWalk {
                endpoint: "http://shard:8080".to_string(),
                wire: Some("q4k".to_string()),
            })
        );
    }

    #[test]
    fn from_name_remote_walk_underscore_alias() {
        assert!(matches!(
            FfnBackendKind::from_name("remote_walk:endpoint=http://x"),
            Some(FfnBackendKind::RemoteWalk { .. })
        ));
    }

    #[test]
    fn from_name_null_aliases() {
        assert_eq!(
            FfnBackendKind::from_name("null"),
            Some(FfnBackendKind::Null)
        );
        assert_eq!(FfnBackendKind::from_name("off"), Some(FfnBackendKind::Null));
    }

    #[test]
    fn from_name_unknown_returns_none() {
        assert!(FfnBackendKind::from_name("unknown-thing").is_none());
        assert!(FfnBackendKind::from_name("").is_none());
    }

    #[test]
    fn from_name_unknown_param_is_silently_ignored() {
        assert_eq!(
            FfnBackendKind::from_name("walk:unknown=42"),
            Some(FfnBackendKind::Walk { k: None })
        );
    }

    #[test]
    fn display_name_matches_canonical_supported_name() {
        let kinds = [
            FfnBackendKind::Dense,
            FfnBackendKind::Walk { k: None },
            FfnBackendKind::RemoteWalk {
                endpoint: "http://x".into(),
                wire: None,
            },
            FfnBackendKind::Null,
        ];
        for k in &kinds {
            let name = k.display_name();
            assert!(
                FfnBackendKind::supported_names().contains(&name),
                "display_name {name:?} missing from supported_names"
            );
        }
    }

    #[test]
    fn supported_names_every_entry_parses_back() {
        for name in FfnBackendKind::supported_names() {
            let parsed = match *name {
                "remote-walk" => FfnBackendKind::from_name(&format!("{name}:endpoint=http://x")),
                other => FfnBackendKind::from_name(other),
            };
            let parsed = parsed.unwrap_or_else(|| {
                panic!("supported_names lists {name:?} but from_name rejected it")
            });
            assert_eq!(
                parsed.display_name(),
                *name,
                "{name:?} parsed to a different display_name"
            );
        }
    }

    #[test]
    fn supported_names_count_matches_variant_count() {
        let one_of_each = [
            FfnBackendKind::Dense,
            FfnBackendKind::Walk { k: None },
            FfnBackendKind::RemoteWalk {
                endpoint: "x".into(),
                wire: None,
            },
            FfnBackendKind::Null,
        ];
        assert_eq!(
            FfnBackendKind::supported_names().len(),
            one_of_each.len(),
            "supported_names and the variant set are out of sync"
        );
    }

    #[test]
    fn ffn_backend_kind_serde_is_internally_tagged() {
        let dense_json = serde_json::to_string(&FfnBackendKind::Dense).unwrap();
        assert_eq!(dense_json, r#"{"kind":"dense"}"#);

        let walk = FfnBackendKind::Walk { k: Some(100) };
        let walk_json = serde_json::to_string(&walk).unwrap();
        assert_eq!(walk_json, r#"{"kind":"walk","k":100}"#);

        let walk_none = FfnBackendKind::Walk { k: None };
        let walk_none_json = serde_json::to_string(&walk_none).unwrap();
        assert_eq!(walk_none_json, r#"{"kind":"walk","k":null}"#);

        let round: FfnBackendKind = serde_json::from_str(&walk_json).unwrap();
        assert_eq!(round, walk);
    }
}
