use graphql_parser::schema::{parse_schema, Definition, TypeDefinition, Type};
use larql_core::core::graph::Graph;

use super::{ast_edge, LanguageExtractor};

pub struct GraphqlExtractor;

impl LanguageExtractor for GraphqlExtractor {
    fn extensions(&self) -> &[&'static str] {
        &["graphql", "gql"]
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let doc = match parse_schema::<String>(source) {
            Ok(d) => d,
            Err(_) => return,
        };

        for def in &doc.definitions {
            if let Definition::TypeDefinition(td) = def {
                match td {
                    TypeDefinition::Object(obj) => {
                        graph.add_edge(ast_edge(&obj.name, "defined_in", path));
                        for field in &obj.fields {
                            graph.add_edge(ast_edge(&field.name, "field_of", &obj.name));
                            let type_name = type_to_str(&field.field_type);
                            graph.add_edge(ast_edge(&field.name, "returns_type", &type_name));
                        }
                    }
                    TypeDefinition::Interface(iface) => {
                        graph.add_edge(ast_edge(&iface.name, "has_interface", path));
                        for field in &iface.fields {
                            graph.add_edge(ast_edge(&field.name, "field_of", &iface.name));
                        }
                    }
                    TypeDefinition::Enum(en) => {
                        graph.add_edge(ast_edge(&en.name, "has_enum", path));
                    }
                    TypeDefinition::InputObject(input) => {
                        graph.add_edge(ast_edge(&input.name, "has_input", path));
                    }
                    _ => {}
                }
            }
        }
    }
}

fn type_to_str(t: &Type<String>) -> String {
    match t {
        Type::NamedType(name) => name.clone(),
        Type::ListType(inner) => format!("[{}]", type_to_str(inner)),
        Type::NonNullType(inner) => format!("{}!", type_to_str(inner)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    const SCHEMA: &str = r#"
type User {
  id: ID!
  name: String
  posts: [Post!]!
}
type Post {
  id: ID!
  title: String
  author: User
}
"#;

    #[test]
    fn graphql_object_type_produces_defined_in() {
        let mut g = Graph::new();
        GraphqlExtractor.extract(SCHEMA, "schema.graphql", &mut g);
        assert!(
            g.list_entities().iter().any(|e| e.contains("User")),
            "Expected 'User' in entities, got: {:?}",
            g.list_entities()
        );
    }

    #[test]
    fn graphql_field_produces_field_of() {
        let mut g = Graph::new();
        GraphqlExtractor.extract(SCHEMA, "schema.graphql", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "field_of"),
            "Expected 'field_of' edges for GraphQL fields"
        );
    }

    #[test]
    fn graphql_field_returns_type() {
        let mut g = Graph::new();
        GraphqlExtractor.extract(SCHEMA, "schema.graphql", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "returns_type"),
            "Expected 'returns_type' edges for GraphQL field types"
        );
    }
}
