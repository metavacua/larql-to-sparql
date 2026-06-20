use larql_core::core::{edge::Edge, enums::SourceType, graph::Graph};
use tree_sitter::{Node, Parser};

use super::LanguageExtractor;

pub struct RustExtractor;

fn ast_edge(s: &str, r: &str, o: &str) -> Edge {
    let mut e = Edge::new(s, r, o);
    e.source = SourceType::Ast;
    e.confidence = 1.0;
    e
}

impl LanguageExtractor for RustExtractor {
    fn extensions(&self) -> &[&'static str] {
        &["rs"]
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("tree-sitter-rust load");
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return,
        };
        let bytes = source.as_bytes();
        extract_node(tree.root_node(), bytes, path, graph, None);
    }
}

fn extract_node<'a>(
    node: Node<'a>,
    src: &[u8],
    path: &str,
    graph: &mut Graph,
    scope: Option<&str>,
) {
    match node.kind() {
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let fn_name = name_node.utf8_text(src).unwrap_or("?");
                let qualified = match scope {
                    Some(s) => format!("{s}::{fn_name}"),
                    None => fn_name.to_string(),
                };
                graph.add_edge(ast_edge(&qualified, "defined_in", path));
                // Walk body for call_expression children
                for i in 0..node.child_count() {
                    extract_node(
                        node.child(i as u32).unwrap(),
                        src,
                        path,
                        graph,
                        Some(&qualified),
                    );
                }
                return;
            }
        }
        "call_expression" => {
            if let Some(func) = node.child_by_field_name("function") {
                let callee = func.utf8_text(src).unwrap_or("?");
                if let Some(s) = scope {
                    graph.add_edge(ast_edge(s, "calls", callee));
                }
            }
        }
        "use_declaration" => {
            let text = node.utf8_text(src).unwrap_or("");
            let path_str = text.trim_start_matches("use ").trim_end_matches(';');
            graph.add_edge(ast_edge(path, "imports", path_str));
        }
        _ => {}
    }
    for i in 0..node.child_count() {
        extract_node(node.child(i as u32).unwrap(), src, path, graph, scope);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn extracts_function_def_edge() {
        let source = r#"
            fn hello_world() {
                println!("hi");
            }
        "#;
        let mut g = Graph::new();
        let extractor = RustExtractor;
        extractor.extract(source, "src/lib.rs", &mut g);
        // Should produce at least one edge involving "hello_world"
        let entities = g.list_entities();
        assert!(
            entities.iter().any(|n| n.contains("hello_world")),
            "expected hello_world to appear as a node, got: {:?}",
            entities
        );
    }

    #[test]
    fn use_statement_produces_imports_edge() {
        let source = "use std::collections::HashMap;";
        let mut g = Graph::new();
        RustExtractor.extract(source, "src/main.rs", &mut g);
        let edges = g.edges();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge for use statement"
        );
    }
}
