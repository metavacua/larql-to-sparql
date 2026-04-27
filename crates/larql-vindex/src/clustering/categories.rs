//! Category vocabulary for cluster labeling.
//!
//! Loads from `data/wikidata_categories.json` (Wikidata property labels).
//! Falls back to a built-in core set if the file is not found.
//!
//! To regenerate: `python3 scripts/fetch_wikidata_properties.py`

use std::path::Path;

/// Load category words from the Wikidata categories file, or fall back to built-in.
pub fn category_words() -> Vec<String> {
    // Try loading from file (relative to cwd or workspace root)
    for path in &[
        "data/wikidata_categories.json",
        "../data/wikidata_categories.json",
        "../../data/wikidata_categories.json",
    ] {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(cats) = serde_json::from_str::<Vec<String>>(&text) {
                if !cats.is_empty() {
                    return cats;
                }
            }
        }
    }

    // Fall back to built-in core set
    builtin_categories()
}

/// Load categories from a specific path.
pub fn category_words_from(path: &Path) -> Vec<String> {
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Ok(cats) = serde_json::from_str::<Vec<String>>(&text) {
            if !cats.is_empty() {
                return cats;
            }
        }
    }
    builtin_categories()
}

/// Built-in core categories (used when wikidata file is not available).
fn builtin_categories() -> Vec<String> {
    vec![
        "country", "nation", "city", "place", "location", "region", "continent",
        "language", "nationality", "person", "people", "animal", "plant", "organism",
        "company", "organization", "institution", "brand", "product",
        "capital", "currency", "population", "leader", "president", "founder",
        "birthplace", "occupation", "profession", "genre", "category",
        "science", "biology", "chemistry", "physics", "mathematics", "medicine",
        "technology", "computer", "software", "internet", "digital",
        "music", "literature", "poetry", "film", "sport", "education",
        "politics", "government", "military", "religion", "philosophy",
        "food", "cooking", "ingredient", "agriculture",
        "art", "culture", "history", "geography", "economics", "business",
        "law", "health", "environment", "weather", "nature",
        "color", "shape", "size", "measurement", "quantity", "number",
        "time", "date", "month", "year", "period", "duration", "age",
        "direction", "position", "distance", "speed", "weight",
        "action", "movement", "creation", "destruction", "communication",
        "transport", "trade", "production", "construction",
        "concept", "quality", "property", "relation", "state", "condition",
        "emotion", "behavior", "process", "event", "structure", "system",
        "method", "theory", "principle",
        "material", "substance", "chemical", "mineral", "metal", "liquid",
        "family", "group", "community", "society", "role", "title",
        "code", "markup", "syntax", "format", "encoding", "protocol",
        "function", "variable", "type", "class", "pattern",
        "suffix", "prefix", "plural", "tense", "conjugation",
        "translation", "foreign", "multilingual",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}

/// Common stop words to exclude from cluster labeling.
pub fn is_stop_word(tok: &str) -> bool {
    matches!(
        tok,
        "the" | "and" | "for" | "but" | "not" | "you" | "all" | "can"
        | "her" | "was" | "one" | "our" | "out" | "are" | "has" | "his"
        | "how" | "its" | "may" | "new" | "now" | "old" | "see" | "way"
        | "who" | "did" | "get" | "let" | "say" | "she" | "too" | "use"
        | "from" | "have" | "been" | "will" | "with" | "this" | "that"
        | "they" | "were" | "some" | "them" | "than" | "when" | "what"
        | "your" | "each" | "make" | "like" | "just" | "over" | "such"
        | "take" | "also" | "into" | "only" | "very" | "more" | "does"
        | "most" | "about" | "which" | "their" | "would" | "there"
        | "could" | "other" | "after" | "being" | "where" | "these"
        | "those" | "first" | "should" | "because" | "through" | "before"
        | "between" | "during" | "while" | "under" | "still" | "then"
        | "here" | "both" | "never" | "every" | "much" | "well" | "same"
        | "further" | "again" | "off" | "always" | "might" | "often"
        | "know" | "need" | "even" | "really" | "back" | "must"
        | "another" | "without" | "along" | "until" | "anything"
        | "something" | "nothing" | "everything" | "however" | "already"
        | "though" | "either" | "rather" | "instead" | "within"
        | "right" | "used" | "using" | "since" | "down" | "many"
        | "long" | "upon" | "whether" | "among" | "later"
        | "different" | "possible" | "given" | "including"
        | "called" | "known" | "based" | "several" | "become"
        | "certain" | "general" | "together" | "following"
        | "number" | "part" | "found" | "small" | "large" | "great"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_words_not_empty() {
        let words = category_words();
        assert!(!words.is_empty());
        assert!(words.len() >= 50); // at least the builtin set
    }

    #[test]
    fn category_words_contains_key_terms() {
        let words = category_words();
        // These should be in either the Wikidata file or the builtin fallback
        for term in &["country", "language", "food", "music"] {
            assert!(
                words.iter().any(|w| w == term || w.contains(term)),
                "missing category: {term}"
            );
        }
    }

    #[test]
    fn stop_words_detected() {
        assert!(is_stop_word("the"));
        assert!(is_stop_word("and"));
        assert!(is_stop_word("from"));
        assert!(is_stop_word("because"));
        assert!(is_stop_word("however"));
    }

    #[test]
    fn content_words_not_stop() {
        assert!(!is_stop_word("country"));
        assert!(!is_stop_word("music"));
        assert!(!is_stop_word("France"));
        assert!(!is_stop_word("president"));
    }

    #[test]
    fn category_words_from_path_fallback() {
        // Non-existent path should fall back to builtin
        let words = category_words_from(std::path::Path::new("/nonexistent/path.json"));
        assert!(!words.is_empty());
    }
}
