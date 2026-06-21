use larql_codebase_core::languages::{Endpoint, LanguageQueries};
use larql_core::core::graph::Graph;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

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
/// Each `EdgeTemplate` with `scope_capture = Some(cap)` pushes the matched
/// capture text onto a per-template scope stack. The scope stack is reset at
/// the start of each template's match pass (i.e. templates are processed
/// independently). As a result, `Endpoint::Scope` subjects only see the scope
/// built within the *same* template pass.
///
/// **Known limitation:** the `"calls"` template in `RUST_QUERIES` has
/// `subject: Endpoint::Scope`, but scope is pushed by the separate
/// `"defined_in"` template pass. Because each template gets its own empty
/// stack, `calls` edges are never emitted by this implementation. A correct
/// solution requires a single tree-walk that threads scope through all
/// templates simultaneously (planned for Task 7: migrate `RustExtractor` to
/// `QueryExtractor`).
pub struct QueryExtractor {
    rules: &'static LanguageQueries,
    language_fn: fn() -> Language,
}

impl QueryExtractor {
    pub fn new(rules: &'static LanguageQueries, language_fn: fn() -> Language) -> Self {
        Self { rules, language_fn }
    }
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

        for template in self.rules.templates {
            let query = match Query::new(&lang, template.query) {
                Ok(q) => q,
                Err(_) => continue, // malformed query — skip, don't panic
            };

            let mut cursor = QueryCursor::new();

            // Each template pass gets a fresh scope stack.
            // See doc comment above for the known limitation this implies.
            let mut scope_stack: Vec<String> = Vec::new();

            let mut matches = cursor.matches(&query, tree.root_node(), src_bytes);
            while let Some(m) = matches.next() {
                let subject =
                    resolve(template.subject, m, &query, src_bytes, path, &scope_stack);
                let object =
                    resolve(template.object, m, &query, src_bytes, path, &scope_stack);

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
                            .and_then(|c| c.node.utf8_text(src_bytes).ok())
                            .map(|s| s.to_owned());
                        if let Some(val) = cap_text {
                            scope_stack.push(val);
                        }
                    }
                }
            }
        }
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
}
