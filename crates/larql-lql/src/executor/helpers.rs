//! Shared helpers: formatting, token filtering.

use std::path::Path;

/// Get total size of a directory in bytes.
pub(crate) fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

pub(crate) fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

pub(crate) fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.2} GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

/// Heuristic: is a token readable enough to show to the user?
/// Filters out encoding garbage, isolated combining marks, etc.
pub(crate) fn is_readable_token(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    let readable = tok.chars().filter(|c| {
        c.is_ascii_alphanumeric()
            || *c == ' '
            || *c == '-'
            || *c == '\''
            || *c == '.'
            || *c == ','
    }).count();
    let total = tok.chars().count();
    readable * 2 >= total && total > 0
}

/// Stricter filter for SHOW RELATIONS and DESCRIBE: content words only.
/// Must look like a real word — no code tokens, no encoding fragments.
pub(crate) fn is_content_token(tok: &str) -> bool {
    let tok = tok.trim();
    if !is_readable_token(tok) {
        return false;
    }
    let chars: Vec<char> = tok.chars().collect();
    if chars.len() < 3 || chars.len() > 25 {
        return false;
    }
    // Must be mostly alphabetic
    let alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if alpha < chars.len() * 2 / 3 {
        return false;
    }
    // Reject camelCase code tokens
    for w in chars.windows(2) {
        if w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase() {
            return false;
        }
    }
    // Reject if all non-ASCII (encoding fragment)
    if !chars.iter().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    // Filter English stop words and common function words
    let lower = tok.to_lowercase();
    !matches!(
        lower.as_str(),
        "the" | "and" | "for" | "but" | "not" | "you" | "all" | "can"
        | "her" | "was" | "one" | "our" | "out" | "are" | "has" | "his"
        | "how" | "its" | "may" | "new" | "now" | "old" | "see" | "way"
        | "who" | "did" | "get" | "let" | "say" | "she" | "too" | "use"
        | "from" | "have" | "been" | "will" | "with" | "this" | "that"
        | "they" | "were" | "some" | "them" | "than" | "when"
        | "what" | "your" | "each" | "make" | "like" | "just" | "over"
        | "such" | "take" | "also" | "into" | "only" | "very" | "more"
        | "does" | "most" | "about" | "which" | "their" | "would" | "there"
        | "could" | "other" | "after" | "being" | "where" | "these" | "those"
        | "first" | "should" | "because" | "through" | "before"
        | "par" | "aux" | "che" | "del"
    )
}
