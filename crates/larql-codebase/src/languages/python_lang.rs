use larql_core::core::graph::Graph;
use tree_sitter::{Node, Parser};

use super::{ast_edge, LanguageExtractor};

pub struct PythonExtractor;

impl LanguageExtractor for PythonExtractor {
    fn extensions(&self) -> &[&'static str] {
        &["py"]
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("tree-sitter-python");
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return,
        };
        extract_py(tree.root_node(), source.as_bytes(), path, graph);
    }
}

fn extract_py(node: Node, src: &[u8], path: &str, graph: &mut Graph) {
    match node.kind() {
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = name_node.utf8_text(src).unwrap_or("?");
                graph.add_edge(ast_edge(name, "defined_in", path));
            }
        }
        "import_statement" => {
            let text = node.utf8_text(src).unwrap_or("");
            graph.add_edge(ast_edge(path, "imports", text.trim()));
        }
        "import_from_statement" => {
            let text = node.utf8_text(src).unwrap_or("");
            graph.add_edge(ast_edge(path, "imports", text.trim()));
        }
        _ => {}
    }
    for i in 0..node.child_count() {
        extract_py(node.child(i as u32).unwrap(), src, path, graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn python_def_produces_defined_in() {
        let src = "def compute(x):\n    return x * 2\n";
        let mut g = Graph::new();
        PythonExtractor.extract(src, "utils.py", &mut g);
        let entities = g.list_entities();
        assert!(
            entities.iter().any(|n| n.contains("compute")),
            "expected 'compute' in entities, got: {:?}",
            entities
        );
    }

    #[test]
    fn python_import_produces_imports_edge() {
        let src = "import os\nfrom pathlib import Path\n";
        let mut g = Graph::new();
        PythonExtractor.extract(src, "main.py", &mut g);
        let edges = g.edges();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge for import statement"
        );
    }
}
