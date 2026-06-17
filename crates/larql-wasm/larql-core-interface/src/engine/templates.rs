use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A prompt template for probing a specific relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub relation: String,
    pub template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse_template: Option<String>,
    #[serde(default)]
    pub multi_token: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_tokens: Vec<char>,
}

impl PromptTemplate {
    pub fn format(&self, subject: &str) -> String {
        self.template.replace("{subject}", subject)
    }

    pub fn format_reverse(&self, object: &str) -> Option<String> {
        self.reverse_template
            .as_ref()
            .map(|t| t.replace("{object}", object))
    }
}

/// Registry of prompt templates keyed by relation name.
/// All templates are loaded from external config — nothing is hardcoded.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TemplateRegistry {
    templates: HashMap<String, PromptTemplate>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, template: PromptTemplate) {
        self.templates.insert(template.relation.clone(), template);
    }

    pub fn get(&self, relation: &str) -> Option<&PromptTemplate> {
        self.templates.get(relation)
    }

    pub fn has(&self, relation: &str) -> bool {
        self.templates.contains_key(relation)
    }

    pub fn all(&self) -> Vec<&PromptTemplate> {
        self.templates.values().collect()
    }

    pub fn relations(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Load templates from a JSON value. Expected format:
    /// ```json
    /// [
    ///   {"relation": "capital-of", "template": "The capital of {subject} is", "multi_token": true},
    ///   ...
    /// ]
    /// ```
    pub fn from_json_value(v: &serde_json::Value) -> Self {
        let mut reg = Self::new();
        if let Some(arr) = v.as_array() {
            for item in arr {
                if let Ok(tmpl) = serde_json::from_value::<PromptTemplate>(item.clone()) {
                    reg.register(tmpl);
                }
            }
        }
        reg
    }

    /// Serialize templates to JSON.
    pub fn to_json_value(&self) -> serde_json::Value {
        let templates: Vec<serde_json::Value> = self
            .templates
            .values()
            .map(|t| serde_json::to_value(t).unwrap())
            .collect();
        serde_json::Value::Array(templates)
    }
}
