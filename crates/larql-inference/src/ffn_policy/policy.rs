//! [`FfnLayerPolicy`] — the parsed-but-unvalidated form of an FFN
//! per-layer routing policy, plus its parse-time and validation-time
//! error types.

use core::ops::Range;

use super::backend_kind::{classify_backend_parse_error, FfnBackendKind};
use super::routing::RoutingPredicate;
use super::validated::ValidatedFfnLayerPolicy;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Walk the spec string and split on top-level commas (outside any
/// braces). Commas inside `{...}` are part of the braced spec
/// (kv-params or `layers=N-M,...` ranges) and must be preserved.
fn split_on_top_level_commas(spec: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut brace_depth: i32 = 0;
    for ch in spec.chars() {
        match ch {
            '{' => {
                brace_depth += 1;
                current.push(ch);
            }
            '}' => {
                brace_depth = brace_depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if brace_depth == 0 => {
                let piece = current.trim();
                if !piece.is_empty() {
                    result.push(piece.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let piece = current.trim();
    if !piece.is_empty() {
        result.push(piece.to_string());
    }
    result
}

/// Per-layer FFN dispatch policy.
///
/// A policy binds backend kinds to layer ranges. Validated at parse
/// time: layers must be exactly partitioned (no gaps, no overlaps
/// except via [`RoutingPredicate::Otherwise`]). Use
/// [`Self::from_spec`] to build from a spec string,
/// [`Self::bindings`] to inspect, [`Self::validate_for`] to upgrade
/// to a [`ValidatedFfnLayerPolicy`], and then `build_router` on the
/// validated form to produce a live router.
///
/// **Construction-error on overlap** (not last-wins). Routing
/// predicates are *partitions* of layer space; last-wins hides typos
/// in spec strings (`layers=0-33;layers=14-27` would silently become
/// "dense for the middle"). Refusing at construction surfaces the bug
/// at the API boundary instead of in a downstream regression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfnLayerPolicy {
    bindings: Vec<(RoutingPredicate, FfnBackendKind)>,
}

/// Errors from [`FfnLayerPolicy::from_spec`]. Distinguishes syntactic
/// failures from semantic ones so callers can produce useful error
/// messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyParseError {
    /// Empty spec or all-whitespace.
    EmptySpec,
    /// Malformed brace grouping (unclosed `{`, missing `}`, etc.).
    MalformedBraces { spec: String },
    /// Missing `@` separator between backend and predicate in a
    /// braced binding.
    MissingPredicate { binding: String },
    /// Unrecognised backend name (not in
    /// [`FfnBackendKind::supported_names`]).
    UnknownBackend { name: String },
    /// Unrecognised predicate clause.
    UnknownPredicate { clause: String },
    /// Backend params malformed (e.g. `walk:k=abc`).
    MalformedBackend { spec: String },
    /// More than one `@otherwise` binding in the same policy.
    MultipleOtherwise,
    /// `@otherwise` appears as the only binding (it has nothing to
    /// be "other than").
    OtherwiseAsOnlyBinding,
    /// Two `Layers` bindings overlap on at least one layer.
    OverlappingLayers { layer: usize },
}

impl core::fmt::Display for PolicyParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolicyParseError::EmptySpec => write!(f, "empty FFN policy spec"),
            PolicyParseError::MalformedBraces { spec } => {
                write!(f, "malformed brace grouping in {spec:?}")
            }
            PolicyParseError::MissingPredicate { binding } => {
                write!(f, "missing @predicate in binding {binding:?}")
            }
            PolicyParseError::UnknownBackend { name } => {
                write!(
                    f,
                    "unknown FFN backend {name:?} — supported: {}",
                    FfnBackendKind::supported_names().join(", "),
                )
            }
            PolicyParseError::UnknownPredicate { clause } => {
                write!(
                    f,
                    "unknown routing predicate {clause:?} — supported: all, otherwise, layers=N-M"
                )
            }
            PolicyParseError::MalformedBackend { spec } => {
                write!(f, "malformed FFN backend spec {spec:?}")
            }
            PolicyParseError::MultipleOtherwise => {
                write!(f, "at most one @otherwise binding per policy")
            }
            PolicyParseError::OtherwiseAsOnlyBinding => {
                write!(
                    f,
                    "@otherwise needs at least one other binding to be other than"
                )
            }
            PolicyParseError::OverlappingLayers { layer } => {
                write!(
                    f,
                    "layer {layer} is bound by more than one explicit Layers predicate"
                )
            }
        }
    }
}

impl core::error::Error for PolicyParseError {}

/// Validation errors against a known layer count. Separate from
/// [`PolicyParseError`] because they need `num_layers` to evaluate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyValidationError {
    /// One or more layers in `[0, num_layers)` is not bound by any
    /// predicate (and no `Otherwise` to catch them).
    LayerUnbound { layer: usize },
    /// A `Layers` predicate references a layer >= `num_layers`.
    LayerOutOfRange { layer: usize, num_layers: usize },
}

impl core::fmt::Display for PolicyValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolicyValidationError::LayerUnbound { layer } => {
                write!(
                    f,
                    "layer {layer} is not bound by any predicate (no @otherwise)"
                )
            }
            PolicyValidationError::LayerOutOfRange { layer, num_layers } => {
                write!(
                    f,
                    "layer {layer} is out of range for model with {num_layers} layers"
                )
            }
        }
    }
}

impl core::error::Error for PolicyValidationError {}

impl FfnLayerPolicy {
    /// Parse a policy spec. Accepts two forms:
    ///
    /// - **Uniform** (no braces): `dense`, `walk:k=100`. Becomes a
    ///   single `(RoutingPredicate::All, kind)` binding.
    /// - **Per-layer** (braced):
    ///   `{walk:k=100}@layers=14-27;{dense}@otherwise`. Each
    ///   `{ffn}@pred` is one binding; bindings are joined by `;`.
    ///
    /// Parse-time checks:
    ///
    /// - At most one `@otherwise` binding.
    /// - `@otherwise` cannot be the only binding.
    /// - Explicit `Layers` predicates don't overlap each other.
    ///
    /// Layer-coverage checks (no gaps, no out-of-range) need
    /// `num_layers` and live on [`Self::validate_for`].
    pub fn from_spec(spec: &str) -> Result<Self, PolicyParseError> {
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            return Err(PolicyParseError::EmptySpec);
        }

        // Uniform form — no '{' in the spec.
        if !trimmed.contains('{') {
            let kind = FfnBackendKind::from_name(trimmed)
                .ok_or_else(|| classify_backend_parse_error(trimmed))?;
            return Ok(Self {
                bindings: vec![(RoutingPredicate::All, kind)],
            });
        }

        // Per-layer form — split on ';' and parse each piece as
        // `{ffn}@pred`.
        let mut bindings: Vec<(RoutingPredicate, FfnBackendKind)> = Vec::new();
        let mut otherwise_count = 0usize;
        let mut explicit_layers: Vec<Range<usize>> = Vec::new();

        for raw_piece in trimmed.split(';') {
            let piece = raw_piece.trim();
            if piece.is_empty() {
                continue;
            }

            let inner =
                piece
                    .strip_prefix('{')
                    .ok_or_else(|| PolicyParseError::MalformedBraces {
                        spec: piece.to_string(),
                    })?;
            let close_idx = inner
                .find('}')
                .ok_or_else(|| PolicyParseError::MalformedBraces {
                    spec: piece.to_string(),
                })?;
            let ffn_spec = &inner[..close_idx];
            let after_brace = &inner[close_idx + 1..];

            let predicate_spec = after_brace.trim_start().strip_prefix('@').ok_or_else(|| {
                PolicyParseError::MissingPredicate {
                    binding: piece.to_string(),
                }
            })?;

            let kind = FfnBackendKind::from_name(ffn_spec.trim())
                .ok_or_else(|| classify_backend_parse_error(ffn_spec.trim()))?;
            let predicate =
                RoutingPredicate::parse_clause(predicate_spec.trim()).ok_or_else(|| {
                    PolicyParseError::UnknownPredicate {
                        clause: predicate_spec.trim().to_string(),
                    }
                })?;

            // Overlap check applies only to explicit Layers
            // predicates — All and Otherwise are independent kinds
            // and don't compose with overlap semantics.
            if let RoutingPredicate::Layers(ref rs) = predicate {
                for r in rs {
                    for existing in &explicit_layers {
                        for layer in r.clone() {
                            if existing.contains(&layer) {
                                return Err(PolicyParseError::OverlappingLayers { layer });
                            }
                        }
                    }
                }
                explicit_layers.extend(rs.iter().cloned());
            }
            if matches!(predicate, RoutingPredicate::Otherwise) {
                otherwise_count += 1;
            }

            bindings.push((predicate, kind));
        }

        if otherwise_count > 1 {
            return Err(PolicyParseError::MultipleOtherwise);
        }
        if bindings.len() == 1 && matches!(bindings[0].0, RoutingPredicate::Otherwise) {
            return Err(PolicyParseError::OtherwiseAsOnlyBinding);
        }
        if bindings.is_empty() {
            return Err(PolicyParseError::EmptySpec);
        }

        Ok(Self { bindings })
    }

    /// Inspect the parsed bindings in declaration order.
    pub fn bindings(&self) -> &[(RoutingPredicate, FfnBackendKind)] {
        &self.bindings
    }

    /// Split a comma-separated list of policy specs into individual
    /// spec strings. Mirrors [`larql_kv::EngineKind::split_specs`].
    ///
    /// The challenge: a single FFN spec may contain commas (inside
    /// braces for per-layer routing, or inside `layers=14-27,28-33`
    /// ranges). The splitter walks the string, tracking brace depth,
    /// and splits only on top-level commas. Then it re-parses each
    /// piece via [`Self::from_spec`] — if a piece doesn't parse on its
    /// own, it's a continuation of the previous spec's commas (the
    /// `walk:k=100,foo=bar` kv-param shape, even though `foo=bar` is
    /// not currently a recognised key — defensive against future
    /// param additions).
    ///
    /// Examples:
    ///
    /// ```text
    /// "dense"                        → ["dense"]
    /// "dense,walk:k=100"             → ["dense", "walk:k=100"]
    /// "{walk:k=100}@layers=14-27;{dense}@otherwise"
    ///                                → ["{walk:k=100}@layers=14-27;{dense}@otherwise"]
    /// "dense,{walk:k=100}@layers=14-27,28-33;{dense}@otherwise"
    ///                                → ["dense", "{walk:k=100}@layers=14-27,28-33;{dense}@otherwise"]
    /// ```
    pub fn split_specs(spec: &str) -> Vec<String> {
        let pieces = split_on_top_level_commas(spec);

        // Re-parse pass: if a piece fails to parse, merge it back
        // into the previous spec (it was a kv-comma continuation).
        let mut result: Vec<String> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            if Self::from_spec(&piece).is_ok() {
                result.push(piece);
            } else if let Some(last) = result.last_mut() {
                last.push(',');
                last.push_str(&piece);
            } else {
                // First piece doesn't parse — keep it so the caller
                // can surface the parse error rather than silently
                // dropping it.
                result.push(piece);
            }
        }
        result
    }

    /// Validate against a concrete model's layer count and produce a
    /// validated handle. Consumes the policy (so callers can't
    /// accidentally use the unvalidated form afterwards) and returns
    /// a [`ValidatedFfnLayerPolicy`] on success.
    ///
    /// Checks every layer in `[0, num_layers)` is bound by at least
    /// one predicate (either explicit `Layers`, `All`, or covered by
    /// `Otherwise`), and that no explicit `Layers` predicate
    /// references a layer out of range.
    pub fn validate_for(
        self,
        num_layers: usize,
    ) -> Result<ValidatedFfnLayerPolicy, PolicyValidationError> {
        self.validate_for_impl(num_layers)?;
        Ok(ValidatedFfnLayerPolicy::new_unchecked(self, num_layers))
    }

    fn validate_for_impl(&self, num_layers: usize) -> Result<(), PolicyValidationError> {
        let has_all = self
            .bindings
            .iter()
            .any(|(p, _)| matches!(p, RoutingPredicate::All));
        let has_otherwise = self
            .bindings
            .iter()
            .any(|(p, _)| matches!(p, RoutingPredicate::Otherwise));

        // Out-of-range check.
        for (p, _) in &self.bindings {
            if let RoutingPredicate::Layers(rs) = p {
                for r in rs {
                    if r.end > num_layers {
                        return Err(PolicyValidationError::LayerOutOfRange {
                            layer: r.end - 1,
                            num_layers,
                        });
                    }
                }
            }
        }

        if has_all || has_otherwise {
            return Ok(());
        }

        let mut covered = vec![false; num_layers];
        for (p, _) in &self.bindings {
            if let RoutingPredicate::Layers(rs) = p {
                for r in rs {
                    for layer in r.clone() {
                        if layer < num_layers {
                            covered[layer] = true;
                        }
                    }
                }
            }
        }
        for (layer, c) in covered.iter().enumerate() {
            if !c {
                return Err(PolicyValidationError::LayerUnbound { layer });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Uniform form ────────────────────────────────────────────────────────

    #[test]
    fn uniform_dense_no_braces() {
        let p = FfnLayerPolicy::from_spec("dense").unwrap();
        assert_eq!(p.bindings().len(), 1);
        assert_eq!(p.bindings()[0].0, RoutingPredicate::All);
        assert_eq!(p.bindings()[0].1, FfnBackendKind::Dense);
    }

    #[test]
    fn uniform_walk_with_k_no_braces() {
        let p = FfnLayerPolicy::from_spec("walk:k=100").unwrap();
        assert_eq!(p.bindings().len(), 1);
        assert_eq!(p.bindings()[0].0, RoutingPredicate::All);
        assert_eq!(p.bindings()[0].1, FfnBackendKind::Walk { k: Some(100) });
    }

    #[test]
    fn uniform_empty_spec_rejected() {
        assert_eq!(
            FfnLayerPolicy::from_spec(""),
            Err(PolicyParseError::EmptySpec)
        );
        assert_eq!(
            FfnLayerPolicy::from_spec("   "),
            Err(PolicyParseError::EmptySpec)
        );
    }

    #[test]
    fn uniform_unknown_backend_rejected() {
        match FfnLayerPolicy::from_spec("snake-ffn") {
            Err(PolicyParseError::UnknownBackend { name }) => {
                assert_eq!(name, "snake-ffn");
            }
            other => panic!("expected UnknownBackend, got {other:?}"),
        }
    }

    #[test]
    fn uniform_known_backend_bad_param_reports_malformed() {
        match FfnLayerPolicy::from_spec("walk:k=abc") {
            Err(PolicyParseError::MalformedBackend { spec }) => {
                assert_eq!(spec, "walk:k=abc");
            }
            other => panic!("expected MalformedBackend, got {other:?}"),
        }
    }

    // ── Per-layer form ──────────────────────────────────────────────────────

    #[test]
    fn per_layer_hybrid_map() {
        let spec = "{walk:k=100}@layers=14-27;{dense}@layers=0-13,28-33";
        let p = FfnLayerPolicy::from_spec(spec).unwrap();
        assert_eq!(p.bindings().len(), 2);
        assert_eq!(p.bindings()[0].1, FfnBackendKind::Walk { k: Some(100) });
        assert_eq!(p.bindings()[1].1, FfnBackendKind::Dense);
    }

    #[test]
    fn per_layer_otherwise_sentinel() {
        let spec = "{walk:k=100}@layers=14-27;{dense}@otherwise";
        let p = FfnLayerPolicy::from_spec(spec).unwrap();
        assert_eq!(p.bindings().len(), 2);
        assert!(matches!(p.bindings()[1].0, RoutingPredicate::Otherwise));
    }

    #[test]
    fn per_layer_multiple_otherwise_rejected() {
        let spec = "{dense}@otherwise;{walk}@otherwise";
        assert_eq!(
            FfnLayerPolicy::from_spec(spec),
            Err(PolicyParseError::MultipleOtherwise)
        );
    }

    #[test]
    fn per_layer_otherwise_as_only_binding_rejected() {
        assert_eq!(
            FfnLayerPolicy::from_spec("{dense}@otherwise"),
            Err(PolicyParseError::OtherwiseAsOnlyBinding)
        );
    }

    #[test]
    fn per_layer_overlapping_layers_rejected() {
        let spec = "{walk:k=100}@layers=14-20;{dense}@layers=18-27";
        match FfnLayerPolicy::from_spec(spec) {
            Err(PolicyParseError::OverlappingLayers { layer }) => {
                assert!(
                    (18..=20).contains(&layer),
                    "expected overlap in 18..=20, got layer {layer}"
                );
            }
            other => panic!("expected OverlappingLayers, got {other:?}"),
        }
    }

    #[test]
    fn per_layer_missing_at_predicate_rejected() {
        let spec = "{dense}";
        match FfnLayerPolicy::from_spec(spec) {
            Err(PolicyParseError::MissingPredicate { binding }) => {
                assert_eq!(binding, "{dense}");
            }
            other => panic!("expected MissingPredicate, got {other:?}"),
        }
    }

    #[test]
    fn per_layer_malformed_braces_rejected() {
        match FfnLayerPolicy::from_spec("{dense@all") {
            Err(PolicyParseError::MalformedBraces { .. }) => {}
            other => panic!("expected MalformedBraces, got {other:?}"),
        }
    }

    #[test]
    fn per_layer_unknown_predicate_rejected() {
        match FfnLayerPolicy::from_spec("{dense}@confidence>0.9") {
            Err(PolicyParseError::UnknownPredicate { clause }) => {
                assert_eq!(clause, "confidence>0.9");
            }
            other => panic!("expected UnknownPredicate, got {other:?}"),
        }
    }

    #[test]
    fn per_layer_tolerates_whitespace_and_empty_pieces() {
        let p = FfnLayerPolicy::from_spec(" {walk:k=100} @ layers=14-27 ; ; {dense} @ otherwise ")
            .unwrap();
        assert_eq!(p.bindings().len(), 2);
    }

    // ── validate_for (Err cases only — Ok cases are covered in validated.rs) ──

    #[test]
    fn validate_for_gap_in_coverage_returns_unbound() {
        let p = FfnLayerPolicy::from_spec("{walk}@layers=0-13;{dense}@layers=20-33").unwrap();
        match p.validate_for(34) {
            Err(PolicyValidationError::LayerUnbound { layer }) => {
                assert!(
                    (14..20).contains(&layer),
                    "expected unbound in 14..20, got {layer}"
                );
            }
            other => panic!("expected LayerUnbound, got {other:?}"),
        }
    }

    #[test]
    fn validate_for_out_of_range_layer_returns_out_of_range() {
        let p = FfnLayerPolicy::from_spec("{dense}@layers=0-99").unwrap();
        match p.validate_for(34) {
            Err(PolicyValidationError::LayerOutOfRange { layer, num_layers }) => {
                assert_eq!(layer, 99);
                assert_eq!(num_layers, 34);
            }
            other => panic!("expected LayerOutOfRange, got {other:?}"),
        }
    }

    // ── split_specs ─────────────────────────────────────────────────────────

    #[test]
    fn split_specs_single_uniform_returns_one_piece() {
        assert_eq!(FfnLayerPolicy::split_specs("dense"), vec!["dense"]);
        assert_eq!(
            FfnLayerPolicy::split_specs("walk:k=100"),
            vec!["walk:k=100"]
        );
    }

    #[test]
    fn split_specs_comma_separated_uniform_splits_cleanly() {
        // Two FFN backends across a cross-product sweep.
        let pieces = FfnLayerPolicy::split_specs("dense,walk:k=100");
        assert_eq!(pieces, vec!["dense", "walk:k=100"]);
    }

    #[test]
    fn split_specs_three_way_cross_product() {
        let pieces = FfnLayerPolicy::split_specs("dense,walk:k=100,null");
        assert_eq!(pieces, vec!["dense", "walk:k=100", "null"]);
    }

    #[test]
    fn split_specs_preserves_braced_per_layer_form_intact() {
        // A single braced spec contains semicolons (between bindings)
        // and commas (inside `layers=N-M,...`). split_specs must NOT
        // split the inside.
        let spec = "{walk:k=100}@layers=14-27,28-33;{dense}@otherwise";
        let pieces = FfnLayerPolicy::split_specs(spec);
        assert_eq!(pieces, vec![spec]);
    }

    #[test]
    fn split_specs_mixes_uniform_and_braced() {
        // Cross-product where one ffn is uniform and one is per-layer.
        let pieces =
            FfnLayerPolicy::split_specs("dense,{walk:k=100}@layers=14-27;{dense}@otherwise");
        assert_eq!(
            pieces,
            vec!["dense", "{walk:k=100}@layers=14-27;{dense}@otherwise"]
        );
    }

    #[test]
    fn split_specs_tolerates_extra_whitespace_and_empty_pieces() {
        let pieces = FfnLayerPolicy::split_specs("  dense , , walk:k=100  ");
        assert_eq!(pieces, vec!["dense", "walk:k=100"]);
    }

    #[test]
    fn split_specs_empty_input_returns_empty_vec() {
        assert!(FfnLayerPolicy::split_specs("").is_empty());
        assert!(FfnLayerPolicy::split_specs("   ").is_empty());
    }

    #[test]
    fn split_specs_remote_walk_with_url_separates_cleanly() {
        // The endpoint URL contains `:` and `/` but no commas, so
        // it doesn't tangle with the comma-separator logic.
        let pieces = FfnLayerPolicy::split_specs("dense,remote-walk:endpoint=http://shard:8080");
        assert_eq!(
            pieces,
            vec!["dense", "remote-walk:endpoint=http://shard:8080"]
        );
    }

    #[test]
    fn split_specs_remote_walk_with_wire_kv_merges_via_reparse() {
        // `remote-walk:endpoint=X,wire=Y` is one spec with two
        // kv-params. The naive comma-split would produce
        // ["remote-walk:endpoint=X", "wire=Y"], but the re-parse
        // pass detects "wire=Y" doesn't parse on its own and merges
        // it back into the previous piece.
        let pieces = FfnLayerPolicy::split_specs("remote-walk:endpoint=http://x,wire=q4k");
        assert_eq!(pieces, vec!["remote-walk:endpoint=http://x,wire=q4k"]);
    }

    #[test]
    fn split_specs_first_piece_unparseable_is_preserved() {
        // If the first piece doesn't parse, we keep it so the caller
        // can surface the error. Mirrors EngineKind::split_specs.
        let pieces = FfnLayerPolicy::split_specs("nonsense");
        assert_eq!(pieces, vec!["nonsense"]);
    }

    #[test]
    fn split_specs_every_piece_round_trips_through_from_spec() {
        // Property check: a valid cross-product spec splits into
        // valid pieces, each of which parses on its own.
        let pieces = FfnLayerPolicy::split_specs(
            "dense,walk:k=100,{walk:k=50}@layers=14-27;{dense}@otherwise,null",
        );
        for p in &pieces {
            assert!(
                FfnLayerPolicy::from_spec(p).is_ok(),
                "piece {p:?} should parse on its own"
            );
        }
        assert_eq!(pieces.len(), 4);
    }

    // ── Error formatting ────────────────────────────────────────────────────

    #[test]
    fn parse_error_display_includes_supported_backends() {
        let e = PolicyParseError::UnknownBackend {
            name: "snake".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("snake"));
        assert!(msg.contains("dense"));
        assert!(msg.contains("walk"));
    }

    #[test]
    fn validation_error_display_mentions_layer_and_count() {
        let e = PolicyValidationError::LayerOutOfRange {
            layer: 99,
            num_layers: 34,
        };
        let msg = format!("{e}");
        assert!(msg.contains("99"));
        assert!(msg.contains("34"));
    }
}
