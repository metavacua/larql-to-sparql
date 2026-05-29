/// An entity node — always derived from edges, never stored directly.
/// Node type is an optional free-form string, inferred from schema rules at runtime.
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub node_type: Option<String>,
    pub degree: usize,
    pub out_degree: usize,
    pub in_degree: usize,
}
