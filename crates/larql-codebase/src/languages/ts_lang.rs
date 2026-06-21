use larql_core::core::graph::Graph;
use tree_sitter::{Node, Parser};

use super::{ast_edge, LanguageExtractor};

pub struct TsExtractor;

impl LanguageExtractor for TsExtractor {
    fn extensions(&self) -> &[&'static str] {
        &["ts", "tsx"]
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            .expect("tree-sitter-typescript");
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return,
        };
        extract_ts(tree.root_node(), source.as_bytes(), path, graph);
    }
}

fn extract_ts(node: Node, src: &[u8], path: &str, graph: &mut Graph) {
    match node.kind() {
        "function_declaration" | "arrow_function" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = name_node.utf8_text(src).unwrap_or("?");
                graph.add_edge(ast_edge(name, "defined_in", path));
            }
        }
        "import_statement" => {
            if let Some(src_node) = node.child_by_field_name("source") {
                let module = src_node
                    .utf8_text(src)
                    .unwrap_or("?")
                    .trim_matches('"')
                    .trim_matches('\'');
                graph.add_edge(ast_edge(path, "imports", module));
            }
        }
        _ => {}
    }
    for i in 0..node.child_count() {
        extract_ts(node.child(i as u32).unwrap(), src, path, graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn ts_function_produces_defined_in() {
        let src = "function greet(name: string): string { return `Hello ${name}`; }";
        let mut g = Graph::new();
        TsExtractor.extract(src, "hello.ts", &mut g);
        let entities = g.list_entities();
        assert!(
            entities.iter().any(|n| n.contains("greet")),
            "expected 'greet' in entities, got: {:?}",
            entities
        );
    }
}
