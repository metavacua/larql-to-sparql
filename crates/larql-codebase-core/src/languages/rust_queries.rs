use super::edge_template::{EdgeTemplate, Endpoint, LanguageQueries};

pub const RUST_QUERIES: LanguageQueries = LanguageQueries {
    extensions: &["rs"],
    templates: &[
        // Function definitions → ("fn_name" or "Mod::fn_name", "defined_in", "path.rs")
        // scope_capture pushes "fn_name" so nested call_expression nodes see it as scope.
        EdgeTemplate {
            query: "(function_item name: (identifier) @name)",
            relation: "defined_in",
            subject: Endpoint::Capture("name"),
            object: Endpoint::File,
            scope_capture: Some("name"),
        },
        // Call expressions inside a function scope → ("caller", "calls", "callee_text")
        // subject=Scope suppresses the edge when not inside any function.
        EdgeTemplate {
            query: "(call_expression function: (_) @callee)",
            relation: "calls",
            subject: Endpoint::Scope,
            object: Endpoint::Capture("callee"),
            scope_capture: None,
        },
        // use declarations → ("path.rs", "imports", "use std::collections::HashMap;")
        EdgeTemplate {
            query: "(use_declaration) @import",
            relation: "imports",
            subject: Endpoint::File,
            object: Endpoint::Capture("import"),
            scope_capture: None,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_queries_has_extensions() {
        assert!(RUST_QUERIES.extensions.contains(&"rs"));
    }

    #[test]
    fn rust_queries_has_defined_in_template() {
        assert!(RUST_QUERIES.templates.iter().any(|t| t.relation == "defined_in"));
    }

    #[test]
    fn rust_queries_has_calls_template() {
        assert!(RUST_QUERIES.templates.iter().any(|t| t.relation == "calls"));
    }
}
