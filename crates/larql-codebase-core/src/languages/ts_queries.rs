use super::edge_template::{EdgeTemplate, Endpoint, LanguageQueries};

// Base templates shared between TypeScript and AssemblyScript.
// AssemblyScript is a strict TypeScript subset that compiles to Wasm.
const TS_BASE_TEMPLATES: &[EdgeTemplate] = &[
    EdgeTemplate {
        query: "(function_declaration name: (identifier) @name)",
        relation: "defined_in",
        subject: Endpoint::Capture("name"),
        object: Endpoint::File,
        scope_capture: None,
    },
    EdgeTemplate {
        query: "(method_definition name: (property_identifier) @name)",
        relation: "defined_in",
        subject: Endpoint::Capture("name"),
        object: Endpoint::File,
        scope_capture: None,
    },
    EdgeTemplate {
        query: "(import_statement) @import",
        relation: "imports",
        subject: Endpoint::File,
        object: Endpoint::Capture("import"),
        scope_capture: None,
    },
];

/// Rules for TypeScript (.ts, .tsx) using `tree-sitter-typescript`.
pub const TS_QUERIES: LanguageQueries = LanguageQueries {
    extensions: &["ts", "tsx"],
    templates: TS_BASE_TEMPLATES,
};

/// Rules for AssemblyScript (.ts subset compiling to Wasm).
/// Uses the same tree-sitter-typescript grammar and the same query set.
/// Compilation target support is documented in the `COMPILATION_MATRIX`.
pub const ASSEMBLYSCRIPT_QUERIES: LanguageQueries = LanguageQueries {
    extensions: &["ts"],
    templates: TS_BASE_TEMPLATES,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_queries_extensions() {
        assert!(TS_QUERIES.extensions.contains(&"ts"));
    }

    #[test]
    fn assemblyscript_shares_templates() {
        assert!(std::ptr::eq(
            TS_QUERIES.templates.as_ptr(),
            ASSEMBLYSCRIPT_QUERIES.templates.as_ptr()
        ));
    }

    #[test]
    fn ts_queries_has_defined_in_template() {
        assert!(TS_QUERIES.templates.iter().any(|t| t.relation == "defined_in"));
    }
}
