/// Parse Wikidata SPARQL JSON results into (subject, object) label pairs,
/// reading the two named bindings `subj_var` and `obj_var`.
pub fn parse_pairs(results_json: &str, subj_var: &str, obj_var: &str)
    -> Result<Vec<(String, String)>, serde_json::Error> {
    let v: serde_json::Value = serde_json::from_str(results_json)?;
    let mut out = Vec::new();
    if let Some(bindings) = v["results"]["bindings"].as_array() {
        for b in bindings {
            let (Some(s), Some(o)) = (b[subj_var]["value"].as_str(), b[obj_var]["value"].as_str())
                else { continue };
            out.push((s.to_string(), o.to_string()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_subject_object_label_pairs() {
        let json = r#"{"results":{"bindings":[
            {"sLabel":{"value":"France"},"oLabel":{"value":"Paris"}},
            {"sLabel":{"value":"Japan"},"oLabel":{"value":"Tokyo"}}
        ]}}"#;
        let pairs = parse_pairs(json, "sLabel", "oLabel").unwrap();
        assert_eq!(pairs, vec![
            ("France".to_string(), "Paris".to_string()),
            ("Japan".to_string(), "Tokyo".to_string()),
        ]);
    }
    #[test]
    fn skips_bindings_missing_a_variable() {
        let json = r#"{"results":{"bindings":[
            {"sLabel":{"value":"France"},"oLabel":{"value":"Paris"}},
            {"sLabel":{"value":"Atlantis"}}
        ]}}"#;
        let pairs = parse_pairs(json, "sLabel", "oLabel").unwrap();
        assert_eq!(pairs, vec![("France".to_string(), "Paris".to_string())]);
    }
}
