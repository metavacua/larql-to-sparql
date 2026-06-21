use larql_core::core::graph::Graph;

/// Serialize a `Graph` to sorted N-Triples (RDFC-1.0 for all-named-node graphs).
///
/// Each edge `(subject, relation, object)` becomes one N-Triples line:
/// `<subject_iri> <relation_iri> <object_iri> .\n`
///
/// IRIs that are already absolute (`http://`, `https://`, `urn:`, `file://`) are
/// used verbatim. All other strings are percent-encoded and prefixed with
/// `base_iri` (e.g. `"urn:larql:"`).
///
/// The output is sorted lexicographically — a stable, content-addressable
/// fingerprint equivalent to RDFC-1.0 for graphs with no blank nodes.
///
/// Note: this emits 3-term triples `<s> <p> <o> .` without a graph label —
/// that is N-Triples format, not N-Quads (which requires a 4th graph IRI).
pub fn graph_to_ntriples(graph: &Graph, base_iri: &str) -> String {
    let mut lines: Vec<String> = graph
        .edges()
        .iter()
        .map(|e| {
            let s = mint_iri(&e.subject, base_iri);
            let p = mint_iri(&e.relation, base_iri);
            let o = mint_iri(&e.object, base_iri);
            format!("{} {} {} .\n", s, p, o)
        })
        .collect();
    lines.sort();
    lines.join("")
}

fn mint_iri(s: &str, base_iri: &str) -> String {
    if s.starts_with("http://")
        || s.starts_with("https://")
        || s.starts_with("urn:")
        || s.starts_with("file://")
    {
        format!("<{}>", s)
    } else {
        let encoded = percent_encode(s);
        format!("<{}{}>", base_iri, encoded)
    }
}

fn percent_encode(s: &str) -> String {
    s.chars()
        .flat_map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~' | '/' | ':') {
                c.to_string().chars().collect::<Vec<char>>()
            } else {
                // encode as UTF-8 percent-escaped bytes
                c.to_string()
                    .as_bytes()
                    .iter()
                    .flat_map(|b| format!("%{:02X}", b).chars().collect::<Vec<char>>())
                    .collect()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};

    fn make_graph_with_edge(s: &str, r: &str, o: &str) -> Graph {
        let mut g = Graph::new();
        g.add_edge(Edge::new(s, r, o));
        g
    }

    #[test]
    fn ntriples_output_ends_with_newline() {
        let g = make_graph_with_edge("Alice", "knows", "Bob");
        let out = graph_to_ntriples(&g, "urn:larql:");
        assert!(out.ends_with('\n'), "N-Triples must end with newline");
    }

    #[test]
    fn ntriples_output_is_sorted() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("Z", "rel", "A"));
        g.add_edge(Edge::new("A", "rel", "Z"));
        let out = graph_to_ntriples(&g, "urn:larql:");
        let lines: Vec<&str> = out.lines().collect();
        let mut sorted = lines.clone();
        sorted.sort();
        assert_eq!(lines, sorted, "N-Triples lines must be sorted");
    }

    #[test]
    fn http_iris_not_re_minted() {
        let g = make_graph_with_edge(
            "http://example.org/Alice",
            "http://schema.org/knows",
            "http://example.org/Bob",
        );
        let out = graph_to_ntriples(&g, "urn:larql:");
        assert!(
            out.contains("<http://example.org/Alice>"),
            "HTTP IRIs must appear verbatim in angle brackets"
        );
    }

    #[test]
    fn non_iri_strings_are_minted() {
        let g = make_graph_with_edge("hello world", "is", "greeting");
        let out = graph_to_ntriples(&g, "urn:larql:");
        assert!(out.contains("urn:larql:"), "non-IRI strings must be minted");
    }
}
