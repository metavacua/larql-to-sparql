use std::collections::BTreeMap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Relation {
    pub pid: String,
    pub template: String,
    pub pairs: Vec<(String, String)>,
}
impl Relation {
    pub fn prompt(&self, entity: &str) -> String { self.template.replace("{entity}", entity) }
}
#[derive(Debug, Deserialize)]
pub struct Catalog(BTreeMap<String, Relation>);
impl Catalog {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
    pub fn relation(&self, name: &str) -> Option<&Relation> { self.0.get(name) }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Relation)> { self.0.iter() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_carries_template_and_pairs() {
        let json = r#"{"capital":{"pid":"P36","template":"The capital of {entity} is","pairs":[["France","Paris"],["Japan","Tokyo"]]}}"#;
        let cat = Catalog::from_json_str(json).unwrap();
        let r = cat.relation("capital").unwrap();
        assert_eq!(r.template, "The capital of {entity} is");
        assert_eq!(r.pairs.len(), 2);
        assert_eq!(r.prompt("France"), "The capital of France is");
    }
}
