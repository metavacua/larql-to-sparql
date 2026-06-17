use crate::relations::RelationSpec;

/// Assemble (spec, pairs) entries into the `label::Catalog` JSON shape:
/// `{ "<name>": { "pid", "template", "pairs": [[s,o],...] }, ... }`.
pub fn assemble(entries: &[(&RelationSpec, Vec<(String, String)>)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (spec, pairs) in entries {
        map.insert(spec.name.to_string(), serde_json::json!({
            "pid": spec.pid,
            "template": spec.template,
            "pairs": pairs,
        }));
    }
    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relations::RelationSpec;

    #[test]
    fn assembled_json_round_trips_through_producer_catalog() {
        let spec = RelationSpec { name: "capital", pid: "P36",
            template: "The capital of {entity} is", instance_of: "Q6256" };
        let pairs = vec![("France".to_string(), "Paris".to_string()),
                         ("Japan".to_string(),  "Tokyo".to_string())];
        let json = assemble(&[(&spec, pairs)]);
        let s = serde_json::to_string(&json).unwrap();

        // The producer's REAL reader must accept it:
        let cat = larql_vindex::label::catalog::Catalog::from_json_str(&s).unwrap();
        let rel = cat.relation("capital").expect("capital relation present");
        assert_eq!(rel.pid, "P36");
        assert_eq!(rel.template, "The capital of {entity} is");
        assert_eq!(rel.pairs, vec![("France".to_string(),"Paris".to_string()),
                                   ("Japan".to_string(),"Tokyo".to_string())]);
    }
}
