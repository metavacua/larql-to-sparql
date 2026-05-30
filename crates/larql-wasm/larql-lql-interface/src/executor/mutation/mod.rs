//! Mutation executor: INSERT, DELETE, UPDATE, MERGE, REBALANCE.
//!
//! All mutations go through the `PatchedVindex` overlay — base vindex
//! files on disk are never modified.

mod delete;
mod insert;
mod merge;
mod rebalance;
mod update;

use std::collections::HashMap;

use crate::ast::{CompareOp, Condition, Value};
use crate::error::LqlError;
use crate::executor::Session;
use crate::relations::RelationClassifier;

type PatchVectorSnapshot = (Option<String>, Option<String>, Option<String>);

/// Filters extracted from a `WHERE …` clause for slot-targeting mutations.
///
/// `find_features` only takes (entity, relation, layer); a `feature` predicate
/// has to be applied as a post-filter on the candidate list. Centralising this
/// extraction means a `WHERE feature = N` without a `layer` cannot fall through
/// to "match every slot" the way it did before.
pub(super) struct WhereFilters<'a> {
    pub layer: Option<usize>,
    pub feature: Option<usize>,
    pub entity: Option<&'a str>,
    pub relation: Option<(&'a CompareOp, &'a str)>,
}

impl<'a> WhereFilters<'a> {
    pub(super) fn from_conditions(conditions: &'a [Condition]) -> Self {
        let layer = integer_condition(conditions, "layer");
        let feature = integer_condition(conditions, "feature");
        let entity = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| match &c.value {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            });
        let relation = string_condition(conditions, "relation");
        Self {
            layer,
            feature,
            entity,
            relation,
        }
    }

    /// Resolve the (layer, feature) candidate set against the base vindex.
    ///
    /// Honours all three filters (entity, layer, feature). When both layer
    /// and feature are pinned the lookup short-circuits without scanning.
    pub(super) fn resolve_candidates(
        &self,
        base: &larql_vindex::VectorIndex,
    ) -> Vec<(usize, usize)> {
        if let (Some(layer), Some(feature)) = (self.layer, self.feature) {
            return vec![(layer, feature)];
        }
        let mut candidates = base.find_features(self.entity, None, self.layer);
        if let Some(wanted) = self.feature {
            candidates.retain(|&(_layer, feature)| feature == wanted);
        }
        candidates
    }
}

fn integer_condition(conditions: &[Condition], field: &str) -> Option<usize> {
    conditions
        .iter()
        .find(|c| c.field == field)
        .and_then(|c| match c.value {
            // Negative values are kept as a sentinel that won't match any
            // (layer, feature) — preferable to widening the filter to "all".
            Value::Integer(n) if n >= 0 => Some(n as usize),
            Value::Integer(_) => Some(usize::MAX),
            _ => None,
        })
}

pub(super) fn string_condition<'a>(
    conditions: &'a [Condition],
    field: &str,
) -> Option<(&'a CompareOp, &'a str)> {
    conditions
        .iter()
        .find(|c| c.field == field)
        .and_then(|c| match &c.value {
            Value::String(s) => Some((&c.op, s.as_str())),
            _ => None,
        })
}

pub(super) fn relation_filter_matches(
    classifier: Option<&RelationClassifier>,
    relation_filter: Option<(&CompareOp, &str)>,
    layer: usize,
    feature: usize,
) -> Result<bool, LqlError> {
    let Some((op, wanted)) = relation_filter else {
        return Ok(true);
    };
    let Some(classifier) = classifier else {
        return Err(LqlError::Execution(
            "relation filters require relation labels for the active vindex; \
             target by layer/feature or omit relation"
                .into(),
        ));
    };
    let label = classifier.label_for_feature(layer, feature).unwrap_or("");
    Ok(match op {
        CompareOp::Eq => relation_eq(label, wanted),
        CompareOp::Neq => !label.is_empty() && !relation_eq(label, wanted),
        CompareOp::Like => relation_like(label, wanted),
        _ => {
            return Err(LqlError::Execution(format!(
                "unsupported relation predicate operator: {:?}",
                op
            )))
        }
    })
}

fn relation_eq(label: &str, wanted: &str) -> bool {
    let label = label.to_lowercase();
    let wanted = wanted.to_lowercase();
    !label.is_empty() && (label.contains(&wanted) || wanted.contains(&label))
}

fn relation_like(label: &str, pattern: &str) -> bool {
    let label = label.to_lowercase();
    let pattern = pattern.to_lowercase();
    if pattern == "%" {
        return !label.is_empty();
    }
    let needle = pattern.trim_matches('%');
    if needle.is_empty() {
        return !label.is_empty();
    }
    match (pattern.starts_with('%'), pattern.ends_with('%')) {
        (true, true) => label.contains(needle),
        (true, false) => label.ends_with(needle),
        (false, true) => label.starts_with(needle),
        (false, false) => label == needle,
    }
}

impl Session {
    pub(crate) fn refresh_recorded_patch_ops_for_slots(
        &mut self,
        slots: &[(usize, usize)],
    ) -> Result<(), LqlError> {
        if slots.is_empty() || self.patch_recording.is_none() {
            return Ok(());
        }

        let mut snapshots: HashMap<(usize, usize), PatchVectorSnapshot> = HashMap::new();
        {
            let (_, _, patched) = self.require_vindex()?;
            for &(layer, feature) in slots {
                let gate = patched.overrides_gate_at(layer, feature).map(encode_vector);
                let up = patched.up_override_at(layer, feature).map(encode_vector);
                let down = patched.down_override_at(layer, feature).map(encode_vector);
                snapshots.insert((layer, feature), (gate, up, down));
            }
        }

        let Some(recording) = self.patch_recording.as_mut() else {
            return Ok(());
        };
        for op in &mut recording.operations {
            if let larql_vindex::PatchOp::Insert {
                layer,
                feature,
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } = op
            {
                if let Some((gate, up, down)) = snapshots.get(&(*layer, *feature)) {
                    if let Some(gate) = gate {
                        *gate_vector_b64 = Some(gate.clone());
                    }
                    if let Some(up) = up {
                        *up_vector_b64 = Some(up.clone());
                    }
                    if let Some(down) = down {
                        *down_vector_b64 = Some(down.clone());
                    }
                }
            }
        }

        Ok(())
    }
}

fn encode_vector(vec: &[f32]) -> String {
    larql_vindex::patch::core::encode_gate_vector(vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::CompareOp;

    fn cond(field: &str, value: Value) -> Condition {
        Condition {
            field: field.into(),
            op: CompareOp::Eq,
            value,
        }
    }

    #[test]
    fn extracts_layer_feature_entity_relation() {
        let conditions = vec![
            cond("layer", Value::Integer(3)),
            cond("feature", Value::Integer(7)),
            cond("entity", Value::String("France".into())),
            cond("relation", Value::String("capital".into())),
        ];
        let filters = WhereFilters::from_conditions(&conditions);
        assert_eq!(filters.layer, Some(3));
        assert_eq!(filters.feature, Some(7));
        assert_eq!(filters.entity, Some("France"));
        assert_eq!(filters.relation.map(|(_, s)| s), Some("capital"));
    }

    #[test]
    fn extracts_none_when_field_absent() {
        let filters = WhereFilters::from_conditions(&[]);
        assert!(filters.layer.is_none());
        assert!(filters.feature.is_none());
        assert!(filters.entity.is_none());
        assert!(filters.relation.is_none());
    }

    #[test]
    fn negative_integer_becomes_unmatchable_sentinel() {
        // Regression: negative `WHERE layer = -1` previously cast to a huge
        // value via `n as usize`; we keep that behaviour explicitly so a
        // negative filter never widens to "no filter".
        let conditions = vec![cond("layer", Value::Integer(-1))];
        let filters = WhereFilters::from_conditions(&conditions);
        assert_eq!(filters.layer, Some(usize::MAX));
    }

    #[test]
    fn non_integer_layer_is_ignored() {
        // String value where an integer is expected — silently dropped, not
        // an error (matches the legacy behaviour of the per-field extractors).
        let conditions = vec![cond("layer", Value::String("oops".into()))];
        let filters = WhereFilters::from_conditions(&conditions);
        assert_eq!(filters.layer, None);
    }

    #[test]
    fn relation_eq_pattern_matches_substring() {
        assert!(relation_eq("capital_of", "capital"));
        assert!(relation_eq("capital", "capital_of"));
        assert!(!relation_eq("", "capital"));
    }

    #[test]
    fn relation_like_handles_anchors_and_percent_wildcard() {
        assert!(relation_like("capital_of", "%capital%"));
        assert!(relation_like("capital_of", "%of"));
        assert!(relation_like("capital_of", "capital%"));
        assert!(relation_like("capital", "capital"));
        // Pattern `%` matches any non-empty label.
        assert!(relation_like("anything", "%"));
        assert!(!relation_like("", "%"));
        // Anchored-but-mismatched cases.
        assert!(!relation_like("capital_of", "%xyz"));
        assert!(!relation_like("capital_of", "xyz%"));
        assert!(!relation_like("capital_of", "different"));
    }
}
