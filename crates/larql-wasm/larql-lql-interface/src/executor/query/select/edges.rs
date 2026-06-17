//! `SELECT * FROM EDGES` — the default SELECT verb.
//!
//! Two modes:
//!   1. With both `entity` and `relation` filters, embed the entity
//!      and walk the FFN at every layer to find features whose
//!      relation label matches. Scales the lookup to "capital
//!      features that fire on France" rather than "features whose
//!      top token contains the substring 'France'".
//!   2. Otherwise scan `feature_meta` directly across the requested
//!      layer/feature filters (handles both heap and mmap modes).
//!
//! After collection, optional WHERE-score filter, ORDER BY, and
//! LIMIT are applied before formatting.

use crate::ast::{CompareOp, Condition, Field, NearestClause, OrderBy, Value};
use crate::error::LqlError;
use crate::executor::Session;

use super::format::{
    also_display, banner, format_also, EDGES_DEFAULT_LIMIT, EDGES_WALK_TOP_K, SCORE_EQ_TOLERANCE,
};

/// One row of `SELECT * FROM EDGES` output before formatting.
struct EdgeRow {
    layer: usize,
    feature: usize,
    top_token: String,
    also: String,
    relation: String,
    c_score: f32,
}

/// All filters extracted from a `WHERE` clause for SELECT EDGES.
struct EdgeFilters<'a> {
    entity: Option<&'a str>,
    relation: Option<&'a str>,
    layer: Option<usize>,
    feature: Option<usize>,
    /// `(operator, threshold)` from `WHERE score CMP N` /
    /// `WHERE confidence CMP N`. None when no score predicate.
    score: Option<(CompareOp, f32)>,
}

impl<'a> EdgeFilters<'a> {
    fn from_conditions(conditions: &'a [Condition]) -> Self {
        let entity = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| match &c.value {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            });
        let relation = conditions
            .iter()
            .find(|c| c.field == "relation")
            .and_then(|c| match &c.value {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            });
        let layer = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| match c.value {
                Value::Integer(n) if n >= 0 => Some(n as usize),
                _ => None,
            });
        let feature = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| match c.value {
                Value::Integer(n) if n >= 0 => Some(n as usize),
                _ => None,
            });
        let score = conditions
            .iter()
            .find(|c| c.field == "score" || c.field == "confidence")
            .and_then(|c| {
                let v = match &c.value {
                    Value::Number(n) => Some(*n as f32),
                    Value::Integer(n) => Some(*n as f32),
                    _ => None,
                }?;
                Some((c.op.clone(), v))
            });
        Self {
            entity,
            relation,
            layer,
            feature,
            score,
        }
    }

    /// `score` predicate matches a row's `c_score`.
    fn score_matches(&self, c_score: f32) -> bool {
        match &self.score {
            None => true,
            Some((CompareOp::Gt, t)) => c_score > *t,
            Some((CompareOp::Lt, t)) => c_score < *t,
            Some((CompareOp::Gte, t)) => c_score >= *t,
            Some((CompareOp::Lte, t)) => c_score <= *t,
            Some((CompareOp::Eq, t)) => (c_score - t).abs() < SCORE_EQ_TOLERANCE,
            Some((CompareOp::Neq, t)) => (c_score - t).abs() >= SCORE_EQ_TOLERANCE,
            Some(_) => true,
        }
    }
}

/// Substring-or-substring relation match: empty label always
/// excludes the row; otherwise the user's relation needs to overlap
/// the labelled relation in either direction.
fn relation_match(label: &str, wanted: &str) -> bool {
    if label.is_empty() {
        return false;
    }
    let label_norm = label.to_lowercase();
    let wanted_norm = wanted.to_lowercase();
    label_norm.contains(&wanted_norm) || wanted_norm.contains(&label_norm)
}

impl Session {
    pub(crate) fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, patched) = self.require_vindex()?;

        if let Some(nc) = nearest {
            return self.exec_select_nearest(patched, path, nc, limit);
        }

        let filters = EdgeFilters::from_conditions(conditions);

        let all_layers = patched.loaded_layers();
        // With a feature filter the user expects to see that feature at
        // every layer; otherwise the page-size default applies.
        let default_limit = if filters.feature.is_some() {
            patched.num_layers()
        } else {
            EDGES_DEFAULT_LIMIT as usize
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let classifier = self.relation_classifier();

        let scan_layers: Vec<usize> = if let Some(l) = filters.layer {
            vec![l]
        } else {
            all_layers.clone()
        };

        let mut rows: Vec<EdgeRow> = Vec::new();

        if let (Some(entity), Some(rel)) = (filters.entity, filters.relation) {
            collect_via_walk(
                patched,
                path,
                classifier,
                entity,
                rel,
                filters.feature,
                &scan_layers,
                &mut rows,
            )?;
        } else {
            collect_via_scan(
                patched,
                classifier,
                filters.entity,
                filters.relation,
                filters.feature,
                &scan_layers,
                &mut rows,
            );
        }

        if let Some(ord) = order {
            sort_rows(&mut rows, ord);
        }

        rows.retain(|r| filters.score_matches(r.c_score));
        rows.truncate(limit);

        Ok(format_rows(&rows, filters.relation.is_some()))
    }
}

/// Walk-anchored collection: embed the entity, walk every requested
/// layer, filter hits by the relation label.
#[allow(clippy::too_many_arguments)]
fn collect_via_walk(
    patched: &larql_vindex::PatchedVindex,
    path: &std::path::Path,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity: &str,
    rel: &str,
    feature_filter: Option<usize>,
    scan_layers: &[usize],
    rows: &mut Vec<EdgeRow>,
) -> Result<(), LqlError> {
    let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
        .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)
        .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

    let Some(query) =
        crate::executor::helpers::entity_query_vec(&tokenizer, &embed, embed_scale, entity)?
    else {
        return Ok(());
    };

    let trace = patched.walk(&query, scan_layers, EDGES_WALK_TOP_K);

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if let Some(ff) = feature_filter {
                if hit.feature != ff {
                    continue;
                }
            }
            let rel_label = classifier
                .and_then(|rc| rc.label_for_feature(*layer_idx, hit.feature))
                .unwrap_or("")
                .to_string();
            if !relation_match(&rel_label, rel) {
                continue;
            }
            rows.push(EdgeRow {
                layer: *layer_idx,
                feature: hit.feature,
                top_token: hit.meta.top_token.clone(),
                also: format_also(&hit.meta.top_k),
                relation: rel_label,
                c_score: hit.gate_score,
            });
        }
    }

    Ok(())
}

/// Direct metadata scan: enumerate features at the requested layers
/// and apply optional entity/relation/feature filters.
fn collect_via_scan(
    patched: &larql_vindex::PatchedVindex,
    classifier: Option<&crate::relations::RelationClassifier>,
    entity_filter: Option<&str>,
    relation_filter: Option<&str>,
    feature_filter: Option<usize>,
    scan_layers: &[usize],
    rows: &mut Vec<EdgeRow>,
) {
    for layer in scan_layers {
        let nf = patched.num_features(*layer);
        for feat_idx in 0..nf {
            if let Some(ff) = feature_filter {
                if feat_idx != ff {
                    continue;
                }
            }
            let Some(meta) = patched.feature_meta(*layer, feat_idx) else {
                continue;
            };
            if let Some(ent) = entity_filter {
                if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                    continue;
                }
            }
            let rel_label = classifier
                .and_then(|rc| rc.label_for_feature(*layer, feat_idx))
                .unwrap_or("")
                .to_string();
            if let Some(rel) = relation_filter {
                if !relation_match(&rel_label, rel) {
                    continue;
                }
            }
            rows.push(EdgeRow {
                layer: *layer,
                feature: feat_idx,
                top_token: meta.top_token.clone(),
                also: format_also(&meta.top_k),
                relation: rel_label,
                c_score: meta.c_score,
            });
        }
    }
}

fn sort_rows(rows: &mut [EdgeRow], ord: &OrderBy) {
    match ord.field.as_str() {
        "confidence" | "c_score" => {
            rows.sort_by(|a, b| {
                let cmp = a
                    .c_score
                    .partial_cmp(&b.c_score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if ord.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        "layer" => {
            rows.sort_by(|a, b| {
                let cmp = a.layer.cmp(&b.layer);
                if ord.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }
        _ => {}
    }
}

fn format_rows(rows: &[EdgeRow], explicit_relation_filter: bool) -> Vec<String> {
    let show_relation = explicit_relation_filter || rows.iter().any(|r| !r.relation.is_empty());
    let show_also = rows.iter().any(|r| !r.also.is_empty());

    let mut out = Vec::new();

    let (header, banner_len) = match (show_relation, show_also) {
        (true, true) => (
            format!(
                "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
                "Layer", "Feature", "Token", "Also", "Relation", "Score"
            ),
            86,
        ),
        (true, false) => (
            format!(
                "{:<8} {:<8} {:<20} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Relation", "Score"
            ),
            70,
        ),
        (false, true) => (
            format!(
                "{:<8} {:<8} {:<16} {:<28} {:>8}",
                "Layer", "Feature", "Token", "Also", "Score"
            ),
            72,
        ),
        (false, false) => (
            format!(
                "{:<8} {:<8} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Score"
            ),
            50,
        ),
    };
    out.push(header);
    out.push(banner(banner_len));

    for row in rows {
        let also = also_display(&row.also);
        match (show_relation, show_also) {
            (true, true) => out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                row.layer, row.feature, row.top_token, also, row.relation, row.c_score
            )),
            (true, false) => out.push(format!(
                "L{:<7} F{:<7} {:20} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.relation, row.c_score
            )),
            (false, true) => out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:>8.4}",
                row.layer, row.feature, row.top_token, also, row.c_score
            )),
            (false, false) => out.push(format!(
                "L{:<7} F{:<7} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.c_score
            )),
        }
    }

    if rows.is_empty() {
        out.push("  (no matching edges)".into());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::CompareOp;

    fn cond(field: &str, op: CompareOp, value: Value) -> Condition {
        Condition {
            field: field.into(),
            op,
            value,
        }
    }

    #[test]
    fn edge_filters_extracts_all_predicates() {
        let cs = vec![
            cond("entity", CompareOp::Eq, Value::String("France".into())),
            cond("relation", CompareOp::Eq, Value::String("capital".into())),
            cond("layer", CompareOp::Eq, Value::Integer(5)),
            cond("feature", CompareOp::Eq, Value::Integer(7)),
            cond("score", CompareOp::Gt, Value::Number(0.5)),
        ];
        let f = EdgeFilters::from_conditions(&cs);
        assert_eq!(f.entity, Some("France"));
        assert_eq!(f.relation, Some("capital"));
        assert_eq!(f.layer, Some(5));
        assert_eq!(f.feature, Some(7));
        assert!(matches!(f.score, Some((CompareOp::Gt, _))));
    }

    #[test]
    fn edge_filters_score_matches_each_op() {
        let mk = |op, t: f32| EdgeFilters {
            entity: None,
            relation: None,
            layer: None,
            feature: None,
            score: Some((op, t)),
        };
        assert!(mk(CompareOp::Gt, 0.5).score_matches(0.6));
        assert!(!mk(CompareOp::Gt, 0.5).score_matches(0.4));
        assert!(mk(CompareOp::Lt, 0.5).score_matches(0.4));
        assert!(!mk(CompareOp::Lt, 0.5).score_matches(0.6));
        assert!(mk(CompareOp::Gte, 0.5).score_matches(0.5));
        assert!(mk(CompareOp::Lte, 0.5).score_matches(0.5));
        assert!(mk(CompareOp::Eq, 0.5).score_matches(0.5));
        assert!(mk(CompareOp::Eq, 0.5).score_matches(0.5005));
        assert!(!mk(CompareOp::Eq, 0.5).score_matches(0.6));
        assert!(mk(CompareOp::Neq, 0.5).score_matches(0.6));
    }

    #[test]
    fn edge_filters_no_score_predicate_matches_anything() {
        let f = EdgeFilters {
            entity: None,
            relation: None,
            layer: None,
            feature: None,
            score: None,
        };
        assert!(f.score_matches(-1e9));
        assert!(f.score_matches(1e9));
    }

    #[test]
    fn relation_match_handles_substring_in_either_direction() {
        assert!(relation_match("capital_of", "capital"));
        assert!(relation_match("capital", "capital_of"));
        assert!(!relation_match("", "capital"));
        assert!(!relation_match("director", "actor"));
    }

    #[test]
    fn sort_rows_by_layer_descending() {
        let mut rows = vec![
            EdgeRow {
                layer: 1,
                feature: 0,
                top_token: "".into(),
                also: "".into(),
                relation: "".into(),
                c_score: 0.0,
            },
            EdgeRow {
                layer: 5,
                feature: 0,
                top_token: "".into(),
                also: "".into(),
                relation: "".into(),
                c_score: 0.0,
            },
            EdgeRow {
                layer: 3,
                feature: 0,
                top_token: "".into(),
                also: "".into(),
                relation: "".into(),
                c_score: 0.0,
            },
        ];
        sort_rows(
            &mut rows,
            &OrderBy {
                field: "layer".into(),
                descending: true,
            },
        );
        assert_eq!(rows[0].layer, 5);
        assert_eq!(rows[1].layer, 3);
        assert_eq!(rows[2].layer, 1);
    }

    #[test]
    fn format_rows_empty_emits_no_match_line() {
        let out = format_rows(&[], false);
        assert!(out.last().unwrap().contains("no matching edges"));
    }

    #[test]
    fn format_rows_chooses_widest_layout_with_relation_and_also() {
        let row = EdgeRow {
            layer: 1,
            feature: 2,
            top_token: "Paris".into(),
            also: "French, Europe".into(),
            relation: "capital".into(),
            c_score: 0.95,
        };
        let out = format_rows(&[row], true);
        assert!(out[0].contains("Relation"));
        assert!(out[0].contains("Also"));
        assert!(out.iter().any(|l| l.contains("Paris")));
        assert!(out.iter().any(|l| l.contains("[French, Europe]")));
    }

    #[test]
    fn format_rows_drops_relation_column_when_no_filter_and_no_label() {
        let row = EdgeRow {
            layer: 0,
            feature: 0,
            top_token: "Foo".into(),
            also: "".into(),
            relation: "".into(),
            c_score: 0.5,
        };
        let out = format_rows(&[row], false);
        assert!(!out[0].contains("Relation"));
        assert!(!out[0].contains("Also"));
    }
}
