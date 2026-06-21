extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

/// Extract a structural edge from an LQL source string.
///
/// Parses with `larql_lql_core::parse()` and maps a `Statement::Insert` to a
/// `(entity, relation, target, confidence)` tuple. Parse errors are silenced —
/// callers decide how to surface them. All other statement variants are ignored;
/// they are query/execution operations, not structural assertions.
///
/// Because `larql_lql_core::parse` accepts exactly one statement, this
/// function returns at most one edge. Pass one INSERT statement per call.
pub fn lql_to_edges(source: &str) -> Vec<(String, String, String, f64)> {
    use larql_lql_core::{parse, Statement};

    let mut edges = Vec::new();
    if let Ok(Statement::Insert {
        entity,
        relation,
        target,
        confidence,
        ..
    }) = parse(source)
    {
        let conf = confidence.map(f64::from).unwrap_or(1.0);
        edges.push((entity, relation, target, conf));
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_produces_edge() {
        // Syntax verified against crates/larql-lql-core/src/parser/mutation.rs:
        //   INSERT INTO EDGES (entity, relation, target) VALUES ("e", "r", "t")
        let src = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Alice", "KNOWS", "Bob");"#;
        let edges = lql_to_edges(src);
        assert!(!edges.is_empty(), "Expected at least one edge from INSERT statement");
        assert_eq!(edges[0].0, "Alice");
        assert_eq!(edges[0].1, "KNOWS");
        assert_eq!(edges[0].2, "Bob");
    }

    #[test]
    fn parse_error_returns_empty() {
        let edges = lql_to_edges("this is not valid LQL !!!@#$");
        assert!(edges.is_empty(), "Parse errors must return empty vec, not panic");
    }

    #[test]
    fn confidence_defaults_to_one() {
        let src = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("X", "IS_A", "Y");"#;
        let edges = lql_to_edges(src);
        assert!(!edges.is_empty(), "Expected at least one edge");
        assert!((edges[0].3 - 1.0).abs() < 1e-9, "Default confidence must be 1.0");
    }

    #[test]
    fn explicit_confidence_is_preserved() {
        let src = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Alice", "lives-in", "Colchester") CONFIDENCE 0.8;"#;
        let edges = lql_to_edges(src);
        assert!(!edges.is_empty(), "Expected at least one edge");
        assert!(
            (edges[0].3 - 0.8).abs() < 1e-6,
            "Explicit confidence must be preserved, got {}",
            edges[0].3
        );
    }
}
