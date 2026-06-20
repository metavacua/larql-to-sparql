use larql_codebase_core::{edges_to_weight_repr, BasisTransform, WeightRepr};
use larql_core::core::graph::Graph;

/// Convert a `larql_core::Graph` into a `WeightRepr` using the given basis.
///
/// This wrapper lives in `larql-codebase` (Tier 1) because `larql-codebase-core`
/// is `no_std` and cannot depend on `larql-core`. It bridges the two crates by
/// converting the graph's edges to the raw `(&str, &str, f64)` form that
/// `edges_to_weight_repr` expects.
pub fn graph_to_weight_repr(graph: &Graph, basis: &dyn BasisTransform) -> WeightRepr {
    // Collect owned strings first so we can take &str slices below.
    let owned: Vec<(String, String, f64)> = graph
        .edges()
        .iter()
        .map(|e| (e.subject.clone(), e.object.clone(), e.confidence))
        .collect();
    let edge_refs: Vec<(&str, &str, f64)> = owned
        .iter()
        .map(|(s, o, c)| (s.as_str(), o.as_str(), *c))
        .collect();
    edges_to_weight_repr(&edge_refs, basis)
}
