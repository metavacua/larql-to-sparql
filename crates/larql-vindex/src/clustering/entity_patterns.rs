//! Entity-pattern classes for member-based cluster labeling.
//!
//! An entity-pattern class is a labeled word list ("country", "month",
//! …); when most of a cluster's member tokens fall inside one class,
//! the class label names the cluster. The classes are VOCABULARY, not
//! engine code — they load from `data/entity_patterns.json` through the
//! [`super::data_files`] search chain. Class order in the file is
//! significant: earlier classes win ties (e.g. "language" precedes
//! "country" because french/german/spanish belong to both).
//!
//! A minimal built-in fallback is kept only for when the file cannot be
//! loaded, and the fallback is loud (stderr warning).

use serde::Deserialize;
use std::path::Path;
use std::sync::OnceLock;

use super::data_files::{resolve_data_file, warn_builtin_fallback};

/// Filename of the entity-pattern vocabulary.
pub const ENTITY_PATTERNS_FILE: &str = "entity_patterns.json";

/// One labeled word-list class. `members` are lowercase.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EntityPatternClass {
    /// Label assigned to a cluster dominated by this class.
    pub label: String,
    /// Lowercase member words.
    pub members: Vec<String>,
}

/// The process-wide pattern classes, loaded once from the resolved data
/// file (or, loudly, the built-in fallback).
pub fn entity_pattern_classes() -> &'static [EntityPatternClass] {
    static CLASSES: OnceLock<Vec<EntityPatternClass>> = OnceLock::new();
    CLASSES.get_or_init(|| match resolve_data_file(ENTITY_PATTERNS_FILE) {
        Some(path) => entity_pattern_classes_from(&path),
        None => {
            warn_builtin_fallback(ENTITY_PATTERNS_FILE, "not found in search chain");
            builtin_pattern_classes()
        }
    })
}

/// Load pattern classes from an explicit (config-provided) path.
/// Falls back (loudly) to the built-in classes if the file cannot be
/// loaded. Uncached — the process-wide cache lives in
/// [`entity_pattern_classes`].
pub fn entity_pattern_classes_from(path: &Path) -> Vec<EntityPatternClass> {
    match load_pattern_classes(path) {
        Ok(classes) => classes,
        Err(reason) => {
            warn_builtin_fallback(ENTITY_PATTERNS_FILE, &reason);
            builtin_pattern_classes()
        }
    }
}

fn load_pattern_classes(path: &Path) -> Result<Vec<EntityPatternClass>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read failed: {e}"))?;
    let classes: Vec<EntityPatternClass> =
        serde_json::from_str(&text).map_err(|e| format!("parse failed: {e}"))?;
    if classes.is_empty() {
        return Err("file parsed to an empty class list".to_string());
    }
    Ok(classes)
}

/// Minimal built-in fallback — a few anchor words per class, in the
/// same significance order as the data file. The real vocabulary is
/// `data/entity_patterns.json`.
fn builtin_pattern_classes() -> Vec<EntityPatternClass> {
    let class = |label: &str, members: &[&str]| EntityPatternClass {
        label: label.to_string(),
        members: members.iter().map(|m| m.to_string()).collect(),
    };
    vec![
        class(
            "language",
            &[
                "english", "french", "german", "spanish", "chinese", "japanese",
            ],
        ),
        class(
            "country",
            &["france", "germany", "japan", "china", "australia", "italy"],
        ),
        class(
            "month",
            &[
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ],
        ),
        class(
            "number",
            &[
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_load_from_data_file() {
        // The workspace checkout carries data/entity_patterns.json with
        // the full word lists — e.g. "swahili" (language) and
        // "philippines" (country) are in the file but not the builtin.
        let classes = entity_pattern_classes();
        let lang = classes.iter().find(|c| c.label == "language").unwrap();
        assert!(lang.members.iter().any(|m| m == "swahili"));
        let country = classes.iter().find(|c| c.label == "country").unwrap();
        assert!(country.members.iter().any(|m| m == "philippines"));
    }

    #[test]
    fn data_file_order_puts_language_before_country() {
        let classes = entity_pattern_classes();
        let lang = classes.iter().position(|c| c.label == "language").unwrap();
        let country = classes.iter().position(|c| c.label == "country").unwrap();
        assert!(lang < country, "language must precede country");
    }

    #[test]
    fn explicit_path_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("p.json");
        std::fs::write(
            &path,
            r#"[{"label": "custom", "members": ["alpha", "beta"]}]"#,
        )
        .unwrap();
        let classes = entity_pattern_classes_from(&path);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].label, "custom");
        assert_eq!(classes[0].members, vec!["alpha", "beta"]);
    }

    #[test]
    fn missing_file_falls_back_to_builtin() {
        let classes = entity_pattern_classes_from(Path::new("/nonexistent/patterns.json"));
        assert_eq!(classes, builtin_pattern_classes());
    }

    #[test]
    fn invalid_json_falls_back_to_builtin() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.json");
        std::fs::write(&path, "{not a list").unwrap();
        assert_eq!(
            entity_pattern_classes_from(&path),
            builtin_pattern_classes()
        );
    }

    #[test]
    fn empty_class_list_falls_back_to_builtin() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.json");
        std::fs::write(&path, "[]").unwrap();
        assert_eq!(
            entity_pattern_classes_from(&path),
            builtin_pattern_classes()
        );
    }
}
