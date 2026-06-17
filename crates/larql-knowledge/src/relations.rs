/// One Wikidata relation: its property id, a prompt template
/// (`{entity}` is replaced by the subject), and the instance-of class
/// (Wikidata QID) used to scope subjects.
pub struct RelationSpec {
    pub name: &'static str,
    pub pid: &'static str,
    pub template: &'static str,
    pub instance_of: &'static str,
}

/// Phase-0-validated high-cardinality relations (recovered cross-lineage).
pub const VALIDATED: &[RelationSpec] = &[
    RelationSpec { name: "capital", pid: "P36",
        template: "The capital of {entity} is", instance_of: "Q6256" },
    RelationSpec { name: "official language", pid: "P37",
        template: "The official language of {entity} is", instance_of: "Q6256" },
];

impl RelationSpec {
    /// SPARQL selecting (?sLabel, ?oLabel) for this relation, English labels, capped at `limit`.
    pub fn query(&self, limit: usize) -> String {
        format!(
            "SELECT ?sLabel ?oLabel WHERE {{ ?s wdt:{} ?o . ?s wdt:P31 wd:{} . \
             SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }} }} LIMIT {}",
            self.pid, self.instance_of, limit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validated_has_capital_and_official_language() {
        let names: Vec<&str> = VALIDATED.iter().map(|r| r.name).collect();
        assert!(names.contains(&"capital"));
        assert!(names.contains(&"official language"));
    }
    #[test]
    fn query_embeds_pid_class_and_limit() {
        let cap = VALIDATED.iter().find(|r| r.name == "capital").unwrap();
        let q = cap.query(5);
        assert!(q.contains("wdt:P36"), "q={q}");
        assert!(q.contains("wd:Q6256"), "q={q}");   // country class
        assert!(q.contains("LIMIT 5"), "q={q}");
        assert!(q.contains("?sLabel") && q.contains("?oLabel"), "q={q}");
    }
}
