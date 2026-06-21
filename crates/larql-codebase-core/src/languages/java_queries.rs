use super::edge_template::{EdgeTemplate, Endpoint, LanguageQueries};

pub const JAVA_QUERIES: LanguageQueries = LanguageQueries {
    extensions: &["java"],
    templates: &[
        // method_declaration inside a class → scope gives "ClassName::methodName"
        EdgeTemplate {
            query: "(method_declaration name: (identifier) @name)",
            relation: "defined_in",
            subject: Endpoint::Capture("name"),
            object: Endpoint::File,
            scope_capture: None,
        },
        // class_declaration → ("ClassName", "has_class", "File.java"), opens scope
        EdgeTemplate {
            query: "(class_declaration name: (identifier) @name)",
            relation: "has_class",
            subject: Endpoint::Capture("name"),
            object: Endpoint::File,
            scope_capture: Some("name"),
        },
        // interface_declaration
        EdgeTemplate {
            query: "(interface_declaration name: (identifier) @name)",
            relation: "has_interface",
            subject: Endpoint::Capture("name"),
            object: Endpoint::File,
            scope_capture: None,
        },
        // import_declaration — no named field; capture full node text
        EdgeTemplate {
            query: "(import_declaration) @import",
            relation: "imports",
            subject: Endpoint::File,
            object: Endpoint::Capture("import"),
            scope_capture: None,
        },
        // method_invocation inside scope → ("ClassName", "calls", "methodName")
        EdgeTemplate {
            query: "(method_invocation name: (identifier) @name)",
            relation: "calls",
            subject: Endpoint::Scope,
            object: Endpoint::Capture("name"),
            scope_capture: None,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_queries_defined_in() {
        assert!(JAVA_QUERIES.templates.iter().any(|t| t.relation == "defined_in"));
    }

    #[test]
    fn java_queries_imports() {
        assert!(JAVA_QUERIES.templates.iter().any(|t| t.relation == "imports"));
    }
}
