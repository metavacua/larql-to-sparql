//! Cluster labeling — auto-generate human-readable labels for discovered clusters.
//!
//! Two approaches:
//! 1. TF-IDF: distinctive tokens per cluster (fallback)
//! 2. Member-based: average member embeddings → nearest category word

use ndarray::Array1;
use std::collections::HashMap;

use super::categories::{category_words, is_stop_word};
use super::entity_patterns::{entity_pattern_classes, EntityPatternClass};

/// Minimum mean cosine similarity between a cluster's member embeddings
/// and a category-word embedding for the category to label the cluster.
/// Below this the category match is considered noise and labeling falls
/// through to entity-pattern / TF-IDF labels.
pub const MIN_CATEGORY_SIMILARITY: f32 = 0.25;

/// Fraction of a cluster's member tokens that must fall inside one
/// entity-pattern class (or match the morphological rule) for that
/// label to be assigned. Majority rule: 50%, rounded up.
pub const PATTERN_MATCH_FRACTION: f64 = 0.5;

/// Maximum token length for the morphological (suffix/prefix) rule in
/// [`detect_entity_pattern`]: short all-alphabetic fragments like
/// "ing" / "tion" are treated as morphological units.
const MORPHOLOGICAL_MAX_LEN: usize = 4;

/// TF-IDF labeling: find distinctive tokens per cluster.
pub fn auto_label_clusters(
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    let mut cluster_tokens: Vec<HashMap<String, usize>> = vec![HashMap::new(); k];
    let mut global_tokens: HashMap<String, usize> = HashMap::new();

    for (i, &cluster) in assignments.iter().enumerate() {
        if cluster < k && i < top_tokens.len() {
            for tok in top_tokens[i].split('|') {
                let tok = tok.trim().to_lowercase();
                if tok.is_empty() || tok.len() < 3 {
                    continue;
                }
                let ascii_count = tok.chars().filter(|c| c.is_ascii_alphanumeric()).count();
                if ascii_count * 2 < tok.chars().count() {
                    continue;
                }
                if is_stop_word(&tok) {
                    continue;
                }
                *cluster_tokens[cluster].entry(tok.clone()).or_default() += 1;
                *global_tokens.entry(tok).or_default() += 1;
            }
        }
    }

    let total_features = assignments.len().max(1) as f64;
    let mut labels = Vec::with_capacity(k);
    let mut top_lists = Vec::with_capacity(k);

    for (c, cluster_tok) in cluster_tokens.iter().enumerate().take(k) {
        let cluster_size = cluster_tok.values().sum::<usize>().max(1) as f64;

        let mut scored: Vec<(String, f64)> = cluster_tok
            .iter()
            .filter(|(_, &count)| count >= 1)
            .map(|(tok, &count)| {
                let tf = count as f64 / cluster_size;
                let global = *global_tokens.get(tok).unwrap_or(&1) as f64;
                let idf = (total_features / global).ln();
                (tok.clone(), tf * idf)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<String> = scored
            .iter()
            .filter(|(t, _)| t.len() >= 3)
            .take(5)
            .map(|(t, _)| t.clone())
            .collect();

        let label = if top.is_empty() {
            let mut freq: Vec<(String, usize)> =
                cluster_tok.iter().map(|(t, &c)| (t.clone(), c)).collect();
            freq.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            let fallback: Vec<String> = freq
                .iter()
                .filter(|(t, _)| t.len() >= 3)
                .take(3)
                .map(|(t, _)| t.clone())
                .collect();
            if fallback.is_empty() {
                format!("cluster-{c}")
            } else {
                fallback.join("/")
            }
        } else {
            top.iter().take(3).cloned().collect::<Vec<_>>().join("/")
        };

        labels.push(label);
        top_lists.push(top);
    }

    (labels, top_lists)
}

/// Member-based labeling: for each cluster, average the embeddings of its
/// top member tokens, then find the nearest category word.
///
/// Duplicates allowed — two clusters both labeled "country" means two
/// country-related feature groups.
pub fn auto_label_clusters_from_embeddings(
    _centres: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    tokenizer: &tokenizers::Tokenizer,
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    let (tfidf_labels, top_lists) = auto_label_clusters(assignments, top_tokens, k);

    let categories = category_words();
    let hidden = embed.shape()[1];

    // Encode category words → embeddings
    let mut cat_embeds: Vec<(String, Array1<f32>)> = Vec::new();
    for word in &categories {
        if let Some(emb) = encode_token_with_tokenizer(word, embed, hidden, tokenizer) {
            cat_embeds.push((word.clone(), emb));
        }
    }

    // Collect member embeddings per cluster
    let mut cluster_member_embeds: Vec<Vec<Array1<f32>>> = vec![Vec::new(); k];
    for (c, member_embeds) in cluster_member_embeds.iter_mut().enumerate().take(k) {
        if let Some(members) = top_lists.get(c) {
            for tok in members.iter().take(10) {
                if let Some(emb) = encode_token_with_tokenizer(tok, embed, hidden, tokenizer) {
                    member_embeds.push(emb);
                }
            }
        }
    }

    // Score each candidate category by mean similarity to cluster members
    let mut labels = Vec::with_capacity(k);

    for (c, members) in cluster_member_embeds.iter().enumerate().take(k) {
        if members.is_empty() {
            labels.push(
                tfidf_labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("cluster-{c}")),
            );
            continue;
        }

        let mut best_word = String::new();
        let mut best_sim = f32::NEG_INFINITY;

        for (word, cat_embed) in &cat_embeds {
            let mean_sim: f32 = members
                .iter()
                .map(|m| larql_compute::dot(&m.view(), &cat_embed.view()))
                .sum::<f32>()
                / members.len() as f32;
            if mean_sim > best_sim {
                best_sim = mean_sim;
                best_word = word.clone();
            }
        }

        if best_sim >= MIN_CATEGORY_SIMILARITY {
            labels.push(best_word);
        } else if let Some(members) = top_lists.get(c) {
            // Fallback: check if members match known entity patterns
            let pattern_label = detect_entity_pattern(members);
            labels.push(pattern_label.unwrap_or_else(|| {
                tfidf_labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("cluster-{c}"))
            }));
        } else {
            labels.push(
                tfidf_labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("cluster-{c}")),
            );
        }
    }

    (labels, top_lists)
}

/// Detect entity patterns from cluster member tokens.
///
/// If at least [`PATTERN_MATCH_FRACTION`] of members fall inside one
/// entity-pattern class (word lists loaded from
/// `data/entity_patterns.json` — see [`super::entity_patterns`]), that
/// class labels the cluster. Classes are checked in file order (e.g.
/// "language" before "country": many words are both). A structural
/// morphological rule (short all-alphabetic fragments) runs after the
/// vocabulary classes.
pub fn detect_entity_pattern(members: &[String]) -> Option<String> {
    detect_entity_pattern_with(entity_pattern_classes(), members)
}

/// [`detect_entity_pattern`] against an explicit class list — the
/// data-file-free version for callers/tests that supply their own
/// vocabulary.
pub fn detect_entity_pattern_with(
    classes: &[EntityPatternClass],
    members: &[String],
) -> Option<String> {
    if members.is_empty() {
        return None;
    }

    let lower_members: Vec<String> = members.iter().map(|m| m.to_lowercase()).collect();
    let n = lower_members.len();
    let threshold = (n as f64 * PATTERN_MATCH_FRACTION).ceil() as usize;

    for class in classes {
        let hits = lower_members
            .iter()
            .filter(|m| class.members.iter().any(|w| w == *m))
            .count();
        if hits >= threshold {
            return Some(class.label.clone());
        }
    }

    // Morphological: structural, not vocabulary — most members are
    // short suffix/prefix fragments.
    let suffix_hits = lower_members
        .iter()
        .filter(|m| m.len() <= MORPHOLOGICAL_MAX_LEN && m.chars().all(|c| c.is_ascii_alphabetic()))
        .count();
    if suffix_hits >= threshold {
        return Some("morphological".into());
    }

    None
}

/// Encode a token using the tokenizer and embedding matrix.
pub fn encode_token_with_tokenizer(
    tok: &str,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    hidden: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<Array1<f32>> {
    let encoding = tokenizer.encode(tok, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return None;
    }
    let mut avg = Array1::<f32>::zeros(hidden);
    let mut n = 0;
    for &id in ids {
        // Skip BOS (0, 1, 2) and EOS tokens — they dilute the embedding
        if id <= 2 {
            continue;
        }
        if (id as usize) < embed.shape()[0] {
            avg += &embed.row(id as usize);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    avg /= n as f32;
    let norm = larql_compute::norm(&avg.view());
    if norm > 1e-8 {
        avg /= norm;
        Some(avg)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfidf_labels_basic() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let tokens = vec![
            "Paris".into(),
            "Berlin".into(),
            "Tokyo".into(),
            "French".into(),
            "German".into(),
            "Japanese".into(),
        ];
        let (labels, tops) = auto_label_clusters(&assignments, &tokens, 2);
        assert_eq!(labels.len(), 2);
        assert_eq!(tops.len(), 2);
    }

    #[test]
    fn tfidf_with_pipe_separated() {
        let assignments = vec![0, 0];
        let tokens = vec![
            "Paris|PARIS|Parisian".into(),
            "Berlin|BERLIN|Berliner".into(),
        ];
        let (labels, tops) = auto_label_clusters(&assignments, &tokens, 1);
        assert_eq!(labels.len(), 1);
        // Should find distinctive tokens from all pipe-separated entries
        assert!(!tops[0].is_empty());
    }

    #[test]
    fn tfidf_filters_stop_words() {
        let assignments = vec![0, 0, 0];
        let tokens = vec!["the".into(), "and".into(), "for".into()];
        let (labels, _) = auto_label_clusters(&assignments, &tokens, 1);
        // Should not contain stop words in label
        assert!(!labels[0].contains("the"));
    }

    #[test]
    fn detect_country_pattern() {
        let members = vec![
            "australia".into(),
            "italy".into(),
            "germany".into(),
            "france".into(),
            "japan".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("country".into()));
    }

    #[test]
    fn detect_language_pattern() {
        let members = vec![
            "english".into(),
            "french".into(),
            "german".into(),
            "spanish".into(),
            "italian".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("language".into()));
    }

    #[test]
    fn detect_month_pattern() {
        let members = vec![
            "january".into(),
            "february".into(),
            "march".into(),
            "october".into(),
            "november".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("month".into()));
    }

    #[test]
    fn detect_number_pattern() {
        let members = vec![
            "one".into(),
            "two".into(),
            "three".into(),
            "four".into(),
            "five".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("number".into()));
    }

    #[test]
    fn detect_morphological_pattern() {
        let members = vec![
            "ing".into(),
            "tion".into(),
            "ness".into(),
            "ment".into(),
            "ity".into(),
        ];
        assert_eq!(
            detect_entity_pattern(&members),
            Some("morphological".into())
        );
    }

    #[test]
    fn detect_no_pattern() {
        let members = vec![
            "Paris".into(),
            "music".into(),
            "running".into(),
            "table".into(),
            "happy".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), None);
    }

    #[test]
    fn detect_empty_members() {
        assert_eq!(detect_entity_pattern(&[]), None);
    }

    #[test]
    fn named_thresholds_hold_reviewed_values() {
        // Pinned by the 2026-07-30 review hygiene item: the doc claimed
        // "60%+" while the code used 0.5 — resolved in favour of the
        // code (majority rule). The similarity floor was a bare 0.25.
        assert_eq!(PATTERN_MATCH_FRACTION, 0.5);
        assert_eq!(MIN_CATEGORY_SIMILARITY, 0.25);
    }

    fn fruit_class() -> Vec<EntityPatternClass> {
        vec![EntityPatternClass {
            label: "fruit".into(),
            members: vec!["apple".into(), "banana".into()],
        }]
    }

    #[test]
    fn pattern_threshold_fires_at_exact_majority() {
        // 2 of 4 members match = exactly PATTERN_MATCH_FRACTION.
        let members: Vec<String> = ["apple", "banana", "zzzzzzzz", "yyyyyyyy"]
            .map(String::from)
            .into();
        assert_eq!(
            detect_entity_pattern_with(&fruit_class(), &members),
            Some("fruit".into())
        );
    }

    #[test]
    fn pattern_threshold_declines_below_majority() {
        // 1 of 4 members = 25% < PATTERN_MATCH_FRACTION; the misses are
        // long/alphabetic-only so the morphological rule stays quiet too.
        let members: Vec<String> = ["apple", "zzzzzzzz", "yyyyyyyy", "xxxxxxxx"]
            .map(String::from)
            .into();
        assert_eq!(detect_entity_pattern_with(&fruit_class(), &members), None);
    }

    #[test]
    fn earlier_class_wins_when_both_match() {
        let classes = vec![
            EntityPatternClass {
                label: "first".into(),
                members: vec!["shared".into()],
            },
            EntityPatternClass {
                label: "second".into(),
                members: vec!["shared".into()],
            },
        ];
        let members = vec!["shared".to_string()];
        assert_eq!(
            detect_entity_pattern_with(&classes, &members),
            Some("first".into())
        );
    }

    /// Minimal WordLevel tokenizer: `words` get ids 3.. (ids 0-2 are
    /// skipped by `encode_token_with_tokenizer` as BOS/EOS-like).
    fn word_level_tokenizer(words: &[&str]) -> tokenizers::Tokenizer {
        let mut entries = vec![
            "\"[UNK]\": 0".to_string(),
            "\"[PAD1]\": 1".to_string(),
            "\"[PAD2]\": 2".to_string(),
        ];
        for (i, w) in words.iter().enumerate() {
            entries.push(format!("\"{w}\": {}", i + 3));
        }
        let vocab = entries.join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "model": {{"type": "WordLevel", "vocab": {{{vocab}}}, "unk_token": "[UNK]"}},
                "pre_tokenizer": {{"type": "Whitespace"}},
                "added_tokens": []
            }}"#
        );
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    /// Embedding fixture for the similarity-floor tests: only "country"
    /// (a category word), "paris" and "berlin" are in-vocab; member
    /// rows are aligned with or orthogonal to the category row.
    fn similarity_fixture(aligned: bool) -> String {
        let tokenizer = word_level_tokenizer(&["country", "paris", "berlin"]);
        let hidden = 4;
        let mut embed = ndarray::Array2::<f32>::zeros((6, hidden));
        embed[[3, 0]] = 1.0; // "country" → e_x
        let member_axis = if aligned { 0 } else { 1 };
        embed[[4, member_axis]] = 1.0; // "paris"
        embed[[5, member_axis]] = 1.0; // "berlin"

        let centres = ndarray::Array2::<f32>::zeros((1, hidden));
        let assignments = vec![0usize, 0];
        let top_tokens = vec!["paris|berlin".to_string(), "paris|berlin".to_string()];
        let (labels, _) = auto_label_clusters_from_embeddings(
            &centres,
            &embed,
            &tokenizer,
            &assignments,
            &top_tokens,
            1,
        );
        labels[0].clone()
    }

    #[test]
    fn category_label_used_when_similarity_clears_floor() {
        // Members' embeddings equal the "country" category embedding →
        // mean cosine 1.0 ≥ MIN_CATEGORY_SIMILARITY → a category label
        // (the best-scoring encodable category word contains "country").
        let label = similarity_fixture(true);
        assert!(
            label.contains("country"),
            "expected a country category label, got {label:?}"
        );
    }

    #[test]
    fn category_label_declined_below_similarity_floor() {
        // Members orthogonal to every encodable category word → best
        // mean cosine 0.0 < MIN_CATEGORY_SIMILARITY → falls through to
        // the TF-IDF label built from the member tokens.
        let label = similarity_fixture(false);
        assert!(
            label.contains("paris") || label.contains("berlin"),
            "expected the TF-IDF member label, got {label:?}"
        );
    }
}
