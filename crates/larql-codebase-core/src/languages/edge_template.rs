/// How to resolve one endpoint (subject or object) of an emitted edge.
///
/// All variants are `Copy` and contain only `&'static str` or unit — safe
/// in `const` arrays and on `wasm32v1-none`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// The value of the named tree-sitter capture (e.g. `"name"` for `@name`).
    Capture(&'static str),
    /// The current scope stack value (enclosing named construct). Edge is
    /// suppressed when the scope stack is empty.
    Scope,
    /// The source file path supplied to the extractor.
    File,
    /// A compile-time string literal.
    Literal(&'static str),
}

/// Maps tree-sitter query captures to a LARQL edge.
///
/// `query` is a standard tree-sitter S-expression pattern string. It is stored
/// as `&'static str` (Tier-0 data) and compiled to a `tree_sitter::Query`
/// by the Tier-1 `QueryExtractor` at extraction time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeTemplate {
    /// Tree-sitter S-expression pattern, e.g.
    /// `"(function_item name: (identifier) @name)"`.
    pub query: &'static str,
    /// Relation label for the emitted edge, e.g. `"defined_in"`.
    pub relation: &'static str,
    /// How to determine the edge subject from the query captures.
    pub subject: Endpoint,
    /// How to determine the edge object from the query captures.
    pub object: Endpoint,
    /// When `Some(cap)`, the value of capture `cap` is pushed onto the
    /// scope stack before processing child nodes, enabling call-graph tracking.
    pub scope_capture: Option<&'static str>,
}

/// All extraction rules for one language.
#[derive(Clone, Debug)]
pub struct LanguageQueries {
    /// File extensions this language handles, e.g. `&["rs"]`.
    pub extensions: &'static [&'static str],
    /// Edge templates; each template's query is compiled independently.
    pub templates: &'static [EdgeTemplate],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_copy() {
        let e = Endpoint::File;
        let _a = e;
        let _b = e; // Copy — would fail to compile if not Copy
    }

    #[test]
    fn edge_template_const_array() {
        // Verifies that EdgeTemplate can appear in a const array (wasm32v1-none requirement)
        const _T: &[EdgeTemplate] = &[EdgeTemplate {
            query: "(function_item name: (identifier) @name)",
            relation: "defined_in",
            subject: Endpoint::Capture("name"),
            object: Endpoint::File,
            scope_capture: Some("name"),
        }];
    }
}
