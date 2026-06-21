use larql_core::core::graph::Graph;
use wasmparser::{ExternalKind, Parser as WasmParser, Payload, TypeRef};

use super::{ast_edge, LanguageExtractor};

pub struct WasmExtractor;

impl LanguageExtractor for WasmExtractor {
    fn extensions(&self) -> &[&'static str] {
        // "wasm" (binary) not yet supported — requires binary file read path
        &["wat"]
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        // .wat text → .wasm binary via the `wat` crate.
        // For .wasm binary files, LanguageExtractor passes source as &str
        // (read_to_string fails on non-UTF-8, so only .wat text files reach us
        // in the standard codebase walker). The binary branch is provided for
        // completeness but .wasm files are effectively skipped by the walker.
        let binary: Vec<u8> = if path.ends_with(".wasm") {
            // Binary .wasm files can't be read as UTF-8 text; skip gracefully.
            // Future work: add a separate binary-file walk path in extractor.rs.
            return;
        } else {
            match wat::parse_str(source) {
                Ok(b) => b,
                Err(_) => return,
            }
        };

        extract_binary(&binary, path, graph);
    }
}

fn extract_binary(binary: &[u8], path: &str, graph: &mut Graph) {
    for payload in WasmParser::new(0).parse_all(binary) {
        let payload = match payload {
            Ok(p) => p,
            Err(_) => return,
        };

        match payload {
            Payload::ImportSection(reader) => {
                // In wasmparser 0.252.0, ImportSectionReader yields Imports<'_>
                // (which may encode compact groups). Call .into_imports() to flatten
                // into individual Import items.
                for import in reader.into_imports() {
                    let import = match import {
                        Ok(i) => i,
                        Err(_) => continue,
                    };
                    // Track all imports (func, global, memory, table)
                    let kind = match import.ty {
                        TypeRef::Func(_) | TypeRef::FuncExact(_) => "func",
                        TypeRef::Global(_) => "global",
                        TypeRef::Memory(_) => "memory",
                        TypeRef::Table(_) => "table",
                        TypeRef::Tag(_) => "tag",
                    };
                    let name = format!("{}::{}", import.module, import.name);
                    graph.add_edge(ast_edge(path, &format!("imports_{kind}"), &name));
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = match export {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                    match export.kind {
                        ExternalKind::Func | ExternalKind::FuncExact => {
                            graph.add_edge(ast_edge(export.name, "defined_in", path));
                            graph.add_edge(ast_edge(path, "exports", export.name));
                        }
                        ExternalKind::Global => {
                            graph.add_edge(ast_edge(path, "exports_global", export.name));
                        }
                        ExternalKind::Memory => {
                            graph.add_edge(ast_edge(path, "exports_memory", export.name));
                        }
                        ExternalKind::Table => {
                            graph.add_edge(ast_edge(path, "exports_table", export.name));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    const SIMPLE_WAT: &str = r#"
(module
  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)
  (export "add" (func $add))
)
"#;

    const IMPORT_WAT: &str = r#"
(module
  (import "env" "log" (func (param i32)))
  (func $main)
)
"#;

    #[test]
    fn wat_export_produces_defined_in() {
        let mut g = Graph::new();
        WasmExtractor.extract(SIMPLE_WAT, "math.wat", &mut g);
        assert!(!g.edges().is_empty(), "Expected edges from WAT module");
        assert!(
            g.edges()
                .iter()
                .any(|e| e.relation == "defined_in" || e.relation == "exports"),
            "Expected 'defined_in' or 'exports' edge from exported function"
        );
    }

    #[test]
    fn wat_export_subject_is_function_name() {
        let mut g = Graph::new();
        WasmExtractor.extract(SIMPLE_WAT, "math.wat", &mut g);
        assert!(
            g.edges()
                .iter()
                .any(|e| e.subject == "add" && e.relation == "defined_in" && e.object == "math.wat"),
            "Expected edge: add --defined_in--> math.wat"
        );
    }

    #[test]
    fn wat_import_produces_imports_func_edge() {
        let mut g = Graph::new();
        WasmExtractor.extract(IMPORT_WAT, "app.wat", &mut g);
        assert!(
            g.edges()
                .iter()
                .any(|e| e.relation == "imports_func" && e.object == "env::log"),
            "Expected imports_func edge for env::log"
        );
    }

    #[test]
    fn wasm_extractor_extensions() {
        assert!(WasmExtractor.extensions().contains(&"wat"));
        // "wasm" (binary) not yet supported — binary read path not implemented
        assert!(!WasmExtractor.extensions().contains(&"wasm"));
    }
}
