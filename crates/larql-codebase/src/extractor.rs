use std::path::Path;

use thiserror::Error;
use walkdir::WalkDir;

use larql_core::core::graph::Graph;

use crate::languages::{LanguageExtractor, PythonExtractor, RustExtractor, TsExtractor};

#[derive(Error, Debug)]
pub enum CodebaseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

fn extractors() -> Vec<Box<dyn LanguageExtractor>> {
    vec![
        Box::new(RustExtractor),
        Box::new(PythonExtractor),
        Box::new(TsExtractor),
    ]
}

/// Walk `root` recursively and extract AST edges from all recognized source files.
pub fn extract_codebase(root: &Path) -> Result<Graph, CodebaseError> {
    let exts: Vec<Box<dyn LanguageExtractor>> = extractors();
    let mut graph = Graph::new();

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();

        // Skip build artifacts and VCS metadata
        if rel_path.starts_with("target/") || rel_path.starts_with(".git/") {
            continue;
        }

        for extractor in &exts {
            if extractor.extensions().contains(&ext) {
                match std::fs::read_to_string(path) {
                    Ok(source) => extractor.extract(&source, &rel_path, &mut graph),
                    Err(_) => continue, // skip unreadable / binary files
                }
                break;
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn extract_rust_files_from_fixture() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() { println!(\"hi\"); }").unwrap();
        fs::write(src.join("lib.rs"), "use std::fmt; fn helper() {}").unwrap();

        let graph = extract_codebase(dir.path()).unwrap();
        assert!(graph.node_count() > 0, "should have nodes from Rust files");
        let edges = graph.edges();
        assert!(
            edges.iter().any(|e| e.relation == "defined_in"),
            "expected a 'defined_in' edge"
        );
    }
}
