use super::edge_template::{EdgeTemplate, Endpoint, LanguageQueries};

pub const PYTHON_QUERIES: LanguageQueries = LanguageQueries {
    extensions: &["py"],
    templates: &[
        EdgeTemplate {
            query: "(function_definition name: (identifier) @name)",
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
        EdgeTemplate {
            query: "(import_from_statement) @import",
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
    fn python_queries_extensions() {
        assert!(PYTHON_QUERIES.extensions.contains(&"py"));
    }
}
