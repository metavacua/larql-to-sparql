use larql_codebase_core::languages::{Endpoint, LanguageQueries};
use larql_core::core::graph::Graph;
use tree_sitter::{Language, Node, Parser, Query, QueryCursor, StreamingIterator};

use super::ast_edge;
use super::LanguageExtractor;

/// A generic extractor driven by tree-sitter S-expression query strings.
///
/// `rules` holds the query strings and edge mappings (Tier-0 data from
/// `larql-codebase-core`). `language_fn` returns the tree-sitter `Language`
/// for this grammar (Tier-1).
///
/// # Scope tracking
///
/// The extractor performs a single depth-first tree walk, maintaining a shared
/// `scope_stack`. At each node, ALL templates are checked against the node kind.
/// When a template's `scope_capture` is `Some(cap)`, the captured value is
/// pushed onto the scope stack before recursing into child nodes, then popped
/// after. This means `Endpoint::Scope` templates (like `calls`) correctly see
/// the enclosing scope set by `Endpoint::Capture`-based templates (like
/// `defined_in`) in the same walk.
pub struct QueryExtractor {
    rules: &'static LanguageQueries,
    language_fn: fn() -> Language,
}

impl QueryExtractor {
    pub fn new(rules: &'static LanguageQueries, language_fn: fn() -> Language) -> Self {
        Self { rules, language_fn }
    }
}

/// Extract the outermost node kind from a tree-sitter S-expression query.
///
/// E.g. `"(function_item name: (identifier) @name)"` → `"function_item"`.
/// Returns `""` if the pattern doesn't start with `(` followed by an identifier.
fn kind_from_query(q: &str) -> &str {
    let q = q.trim();
    // Must start with '('
    if !q.starts_with('(') {
        return "";
    }
    let rest = &q[1..];
    // Find the end of the first identifier (alphanumeric or '_')
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    &rest[..end]
}

impl LanguageExtractor for QueryExtractor {
    fn extensions(&self) -> &[&'static str] {
        self.rules.extensions
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let lang = (self.language_fn)();

        let mut parser = Parser::new();
        if parser.set_language(&lang).is_err() {
            return;
        }
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return,
        };
        let src_bytes = source.as_bytes();

        // Compile all template queries once.
        let compiled: Vec<Option<Query>> = self
            .rules
            .templates
            .iter()
            .map(|t| Query::new(&lang, t.query).ok())
            .collect();

        // Precompute node kinds for each template.
        let kinds: Vec<&str> = self
            .rules
            .templates
            .iter()
            .map(|t| kind_from_query(t.query))
            .collect();

        let mut scope_stack: Vec<String> = Vec::new();
        walk_node(
            tree.root_node(),
            src_bytes,
            path,
            self.rules.templates,
            &compiled,
            &kinds,
            graph,
            &mut scope_stack,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_node<'a>(
    node: Node<'a>,
    src: &[u8],
    path: &str,
    templates: &'static [larql_codebase_core::languages::EdgeTemplate],
    compiled: &[Option<Query>],
    kinds: &[&str],
    graph: &mut Graph,
    scope_stack: &mut Vec<String>,
) {
    let node_kind = node.kind();
    // Track how many scopes we push at this node level so we can pop them
    // after recursing into children.
    let mut scope_push_count: usize = 0;

    for ((template, query_opt), kind) in
        templates.iter().zip(compiled.iter()).zip(kinds.iter())
    {
        // Only check templates whose query targets the current node kind.
        if *kind != node_kind {
            continue;
        }
        let query = match query_opt {
            Some(q) => q,
            None => continue,
        };

        // Run the query starting at this node, but only allow matches
        // rooted here (depth 0). This avoids double-counting when we
        // also recurse into children that have the same node kind.
        let mut cursor = QueryCursor::new();
        cursor.set_max_start_depth(Some(0));
        let mut matches = cursor.matches(query, node, src);
        while let Some(m) = matches.next() {
            let subject = resolve(template.subject, m, query, src, path, scope_stack);
            let object = resolve(template.object, m, query, src, path, scope_stack);

            if let (Some(s), Some(o)) = (subject, object) {
                graph.add_edge(ast_edge(&s, template.relation, &o));
            }

            // Push scope after emitting the edge so the emitted edge itself
            // is not attributed to the scope it opens.
            if let Some(scope_cap) = template.scope_capture {
                if let Some(cap_idx) = query.capture_index_for_name(scope_cap) {
                    let cap_text = m
                        .captures
                        .iter()
                        .find(|c| c.index == cap_idx)
                        .and_then(|c| c.node.utf8_text(src).ok())
                        .map(|s| s.to_owned());
                    if let Some(val) = cap_text {
                        scope_stack.push(val);
                        scope_push_count += 1;
                    }
                }
            }
        }
    }

    // Recurse into children with the updated scope_stack.
    for i in 0..node.child_count() {
        walk_node(
            node.child(i as u32).unwrap(),
            src,
            path,
            templates,
            compiled,
            kinds,
            graph,
            scope_stack,
        );
    }

    // Pop all scopes opened at this node level.
    for _ in 0..scope_push_count {
        scope_stack.pop();
    }
}

fn resolve(
    endpoint: Endpoint,
    m: &tree_sitter::QueryMatch<'_, '_>,
    query: &Query,
    src: &[u8],
    path: &str,
    scope_stack: &[String],
) -> Option<String> {
    match endpoint {
        Endpoint::Capture(name) => {
            let idx = query.capture_index_for_name(name)?;
            m.captures
                .iter()
                .find(|c| c.index == idx)
                .and_then(|c| c.node.utf8_text(src).ok())
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        }
        Endpoint::Scope => scope_stack.last().cloned(),
        Endpoint::File => Some(path.to_owned()),
        Endpoint::Literal(s) => Some(s.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_codebase_core::languages::RUST_QUERIES;
    use larql_core::core::graph::Graph;

    fn make_rust_extractor() -> QueryExtractor {
        QueryExtractor::new(&RUST_QUERIES, || tree_sitter_rust::LANGUAGE.into())
    }

    #[test]
    fn extracts_function_defined_in() {
        let src = "fn hello() { }";
        let mut g = Graph::new();
        make_rust_extractor().extract(src, "src/lib.rs", &mut g);
        let entities = g.list_entities();
        assert!(
            entities.iter().any(|e| e.contains("hello")),
            "Expected 'hello' in graph entities, got: {:?}",
            entities
        );
    }

    #[test]
    fn extensions_match_rules() {
        let ex = make_rust_extractor();
        assert!(ex.extensions().contains(&"rs"));
    }

    #[test]
    fn extracts_import_edge() {
        let src = "use std::collections::HashMap;";
        let mut g = Graph::new();
        make_rust_extractor().extract(src, "src/main.rs", &mut g);
        let edges = g.edges();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "Expected an 'imports' edge, got: {:?}",
            edges
        );
    }

    #[test]
    fn extracts_calls_edge_inside_function() {
        // Use a real function call (not a macro) to test calls edge emission.
        let src = "fn a() { b(); }";
        let mut g = Graph::new();
        make_rust_extractor().extract(src, "src/lib.rs", &mut g);
        let edges = g.edges();
        assert!(
            edges.iter().any(|e| e.relation == "calls"),
            "Expected a 'calls' edge from a() to b(), got: {:?}",
            edges
        );
        assert!(
            edges
                .iter()
                .any(|e| e.relation == "calls" && e.subject.contains('a')),
            "Expected subject of 'calls' to contain 'a', got: {:?}",
            edges
        );
    }

    #[test]
    fn kind_from_query_extracts_kind() {
        assert_eq!(
            kind_from_query("(function_item name: (identifier) @name)"),
            "function_item"
        );
        assert_eq!(
            kind_from_query("(call_expression function: (_) @callee)"),
            "call_expression"
        );
        assert_eq!(kind_from_query("(use_declaration) @import"), "use_declaration");
        assert_eq!(kind_from_query("not_an_sexp"), "");
    }
}
