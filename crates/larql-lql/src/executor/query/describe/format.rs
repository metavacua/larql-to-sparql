//! Phases 4–5 of `DESCRIBE <entity>`: resolve relation labels,
//! split edges into bands, render rows.

use crate::ast::DescribeMode;

use super::collect::DescribeEdge;

/// A formatted edge ready to be rendered into the output buffer.
/// Built from a `DescribeEdge` by `describe_format_and_split` after
/// label resolution and the RELATIONS ONLY filter.
pub(super) struct FormattedEdge {
    /// Probe label, raw cluster label, or empty when no label is known.
    pub label: String,
    pub is_probe: bool,
    pub is_cluster: bool,
    pub target: String,
    pub gate: f32,
    pub primary_layer: usize,
    pub layers: Vec<usize>,
    pub count: usize,
    pub also: Vec<String>,
}

/// The three formatted-edge buckets returned by
/// `describe_format_and_split`, one per layer band.
pub(super) struct DescribeBands {
    pub syntax: Vec<FormattedEdge>,
    pub knowledge: Vec<FormattedEdge>,
    pub output_band: Vec<FormattedEdge>,
}

/// Resolve relation labels from the optional `RelationClassifier`,
/// apply the RELATIONS ONLY filter, and split into per-band buckets
/// according to which band the primary layer falls in.
pub(super) fn describe_format_and_split(
    edges: &[DescribeEdge],
    classifier: Option<&crate::relations::RelationClassifier>,
    relations_only: bool,
    bands: &larql_vindex::LayerBands,
) -> DescribeBands {
    let formatted: Vec<FormattedEdge> = edges
        .iter()
        .map(|info| {
            let (label, is_probe, is_cluster) = resolve_label(classifier, info);
            FormattedEdge {
                label,
                is_probe,
                is_cluster,
                target: info.original.clone(),
                gate: info.gate,
                primary_layer: info.best_layer,
                layers: info.layers.clone(),
                count: info.count,
                also: info.also.clone(),
            }
        })
        .filter(|e| !relations_only || e.is_probe || e.is_cluster)
        .collect();

    let mut out = DescribeBands {
        syntax: Vec::new(),
        knowledge: Vec::new(),
        output_band: Vec::new(),
    };
    for edge in formatted {
        let primary = edge.primary_layer;
        if primary >= bands.syntax.0 && primary <= bands.syntax.1 {
            out.syntax.push(edge);
        } else if primary >= bands.knowledge.0 && primary <= bands.knowledge.1 {
            out.knowledge.push(edge);
        } else if primary >= bands.output.0 && primary <= bands.output.1 {
            out.output_band.push(edge);
        } else {
            // Layer outside any band — fall back to knowledge.
            out.knowledge.push(edge);
        }
    }
    out
}

fn resolve_label(
    classifier: Option<&crate::relations::RelationClassifier>,
    info: &DescribeEdge,
) -> (String, bool, bool) {
    let Some(rc) = classifier else {
        return (String::new(), false, false);
    };
    let Some(lbl) = rc.label_for_feature(info.best_layer, info.best_feature) else {
        return (String::new(), false, false);
    };
    let probe = rc.is_probe_label(info.best_layer, info.best_feature);
    (lbl.to_string(), probe, !probe)
}

/// Render a single `FormattedEdge` into one line of DESCRIBE output.
///
///   - **Verbose** (default): `[relation]    → target  gate  L20-L27  Nx  also: ...`
///   - **Brief**: compact `relation    → target  gate  L26`, no also-tokens
///   - **Raw**: no labels, otherwise like Verbose
pub(super) fn format_describe_edge(edge: &FormattedEdge, mode: DescribeMode) -> String {
    match mode {
        DescribeMode::Verbose => format_verbose(edge),
        DescribeMode::Brief => format_brief(edge),
        DescribeMode::Raw => format_raw(edge),
    }
}

fn format_verbose(edge: &FormattedEdge) -> String {
    let bracket_label = if edge.label.is_empty() {
        format!("{:<14}", "[—]")
    } else {
        let tag = format!("[{}]", edge.label);
        format!("{:<14}", tag)
    };
    let layer_str = format_layer_range(&edge.layers);
    let also = format_also(&edge.also);
    format!(
        "    {} → {:20} {:>7.1}  {:<8} {}x{}",
        bracket_label, edge.target, edge.gate, layer_str, edge.count, also,
    )
}

fn format_brief(edge: &FormattedEdge) -> String {
    let label = if edge.is_probe {
        format!("{:<12}", edge.label)
    } else {
        format!("{:<12}", "")
    };
    format!(
        "    {} → {:20} {:>7.1}  L{:<3}",
        label, edge.target, edge.gate, edge.primary_layer,
    )
}

fn format_raw(edge: &FormattedEdge) -> String {
    let layer_str = format_layer_range(&edge.layers);
    let also = format_also(&edge.also);
    format!(
        "                 → {:20} {:>7.1}  {:<8} {}x{}",
        edge.target, edge.gate, layer_str, edge.count, also,
    )
}

fn format_layer_range(layers: &[usize]) -> String {
    let min_l = *layers.iter().min().unwrap_or(&0);
    let max_l = *layers.iter().max().unwrap_or(&0);
    if min_l == max_l {
        format!("L{min_l}")
    } else {
        format!("L{min_l}-{max_l}")
    }
}

fn format_also(also: &[String]) -> String {
    if also.is_empty() {
        String::new()
    } else {
        format!("  also: {}", also.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(primary: usize, label: &str, target: &str, gate: f32) -> FormattedEdge {
        FormattedEdge {
            label: label.into(),
            is_probe: !label.is_empty(),
            is_cluster: false,
            target: target.into(),
            gate,
            primary_layer: primary,
            layers: vec![primary],
            count: 1,
            also: Vec::new(),
        }
    }

    #[test]
    fn format_layer_range_single_layer() {
        assert_eq!(format_layer_range(&[5]), "L5");
    }

    #[test]
    fn format_layer_range_multi_layer() {
        // The format is "L{min}-{max}", not "L{min}-L{max}".
        assert_eq!(format_layer_range(&[2, 3, 5]), "L2-5");
    }

    #[test]
    fn format_layer_range_empty_falls_back_to_zero() {
        assert_eq!(format_layer_range(&[]), "L0");
    }

    #[test]
    fn format_also_empty_returns_empty_string() {
        assert_eq!(format_also(&[]), "");
    }

    #[test]
    fn format_also_joins_with_comma() {
        assert_eq!(
            format_also(&["a".into(), "b".into(), "c".into()]),
            "  also: a, b, c"
        );
    }

    #[test]
    fn format_describe_edge_verbose_includes_label_and_target() {
        let e = edge(5, "capital", "Paris", 12.5);
        let s = format_describe_edge(&e, DescribeMode::Verbose);
        assert!(s.contains("[capital]"));
        assert!(s.contains("Paris"));
        assert!(s.contains("L5"));
    }

    #[test]
    fn format_describe_edge_brief_drops_layer_range() {
        let mut e = edge(5, "capital", "Paris", 12.5);
        e.layers = vec![3, 5, 7];
        let s = format_describe_edge(&e, DescribeMode::Brief);
        // Brief uses primary_layer only.
        assert!(s.contains("L5"));
        assert!(!s.contains("L3-L7"));
    }

    #[test]
    fn format_describe_edge_raw_omits_label() {
        let e = edge(5, "capital", "Paris", 12.5);
        let s = format_describe_edge(&e, DescribeMode::Raw);
        assert!(!s.contains("[capital]"));
        assert!(s.contains("Paris"));
    }

    #[test]
    fn split_falls_back_to_knowledge_when_layer_outside_all_bands() {
        let edge = DescribeEdge {
            gate: 5.0,
            layers: vec![99],
            count: 1,
            original: "Stub".into(),
            also: vec![],
            best_layer: 99,
            best_feature: 0,
        };
        let bands = larql_vindex::LayerBands {
            syntax: (0, 4),
            knowledge: (5, 9),
            output: (10, 12),
        };
        let split = describe_format_and_split(&[edge], None, false, &bands);
        assert_eq!(split.knowledge.len(), 1);
        assert!(split.syntax.is_empty());
        assert!(split.output_band.is_empty());
    }

    #[test]
    fn relations_only_drops_edges_without_label() {
        let with_label = DescribeEdge {
            gate: 8.0,
            layers: vec![6],
            count: 1,
            original: "Paris".into(),
            also: vec![],
            best_layer: 6,
            best_feature: 0,
        };
        let without_label = DescribeEdge {
            gate: 9.0,
            layers: vec![6],
            count: 1,
            original: "Foo".into(),
            also: vec![],
            best_layer: 6,
            best_feature: 0,
        };
        let bands = larql_vindex::LayerBands {
            syntax: (0, 4),
            knowledge: (5, 9),
            output: (10, 12),
        };
        // Without classifier, no edges have a label, so RELATIONS ONLY drops everything.
        let split = describe_format_and_split(&[with_label, without_label], None, true, &bands);
        assert!(split.knowledge.is_empty());
    }
}
