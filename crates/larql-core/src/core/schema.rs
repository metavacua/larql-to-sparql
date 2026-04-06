use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata for a relation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationMeta {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subject_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_types: Vec<String>,
    #[serde(default = "default_true")]
    pub reversible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse_name: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Type inference rule: if a node has any of the given outgoing or incoming
/// relations, assign it the specified type string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRule {
    pub node_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outgoing: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub incoming: Vec<String>,
}

/// Registry of known relation types and type inference rules.
/// All content is loaded from external config — nothing is hardcoded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schema {
    #[serde(default)]
    relations: HashMap<String, RelationMeta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    type_rules: Vec<TypeRule>,
}

impl Schema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, meta: RelationMeta) {
        self.relations.insert(meta.name.clone(), meta);
    }

    pub fn get(&self, name: &str) -> Option<&RelationMeta> {
        self.relations.get(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.relations.contains_key(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.relations.keys().map(|s| s.as_str()).collect()
    }

    pub fn add_type_rule(&mut self, rule: TypeRule) {
        self.type_rules.push(rule);
    }

    pub fn type_rules(&self) -> &[TypeRule] {
        &self.type_rules
    }

    /// Infer a node type from its outgoing and incoming relation sets,
    /// using the loaded type rules. Returns None if no rule matches.
    pub fn infer_type(
        &self,
        outgoing: &std::collections::HashSet<String>,
        incoming: &std::collections::HashSet<String>,
    ) -> Option<String> {
        for rule in &self.type_rules {
            let out_match =
                !rule.outgoing.is_empty() && rule.outgoing.iter().any(|r| outgoing.contains(r));
            let in_match =
                !rule.incoming.is_empty() && rule.incoming.iter().any(|r| incoming.contains(r));
            if out_match || in_match {
                return Some(rule.node_type.clone());
            }
        }
        None
    }

    /// Serialize for .larql.json — matches Python format.
    pub fn to_json_value(&self) -> serde_json::Value {
        let relations: Vec<serde_json::Value> = self
            .relations
            .values()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        let mut val = serde_json::json!({ "relations": relations });
        if !self.type_rules.is_empty() {
            val["type_rules"] = serde_json::to_value(&self.type_rules).unwrap();
        }
        val
    }

    /// Deserialize from .larql.json schema block.
    pub fn from_json_value(v: &serde_json::Value) -> Self {
        let mut schema = Self::new();
        if let Some(rels) = v.get("relations").and_then(|r| r.as_array()) {
            for rel in rels {
                if let Ok(meta) = serde_json::from_value::<RelationMeta>(rel.clone()) {
                    schema.add(meta);
                }
            }
        }
        if let Some(rules) = v.get("type_rules").and_then(|r| r.as_array()) {
            for rule in rules {
                if let Ok(tr) = serde_json::from_value::<TypeRule>(rule.clone()) {
                    schema.add_type_rule(tr);
                }
            }
        }
        schema
    }
}
