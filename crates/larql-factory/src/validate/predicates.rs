//! Pure shape predicates used by [`super::validate`]. Each one is a
//! single testable fact about a string's shape, kept out of the main
//! check flow so that flow reads as a list of rules, not a tangle of
//! inline string munging.

use crate::constants::{GIT_COMMIT_SHA_HEX_LEN, SEMVER_COMPONENT_COUNT};

/// True if `s` is a full lowercase-or-uppercase hex commit SHA of the
/// expected length (§4.1: "REQUIRED, full commit sha").
pub fn is_full_hex_sha(s: &str) -> bool {
    s.len() == GIT_COMMIT_SHA_HEX_LEN && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// True if `s` is non-empty lowercase kebab-case: letters, digits, and
/// internal hyphens only, no leading/trailing hyphen.
pub fn is_kebab_case(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !s.starts_with('-')
        && !s.ends_with('-')
}

/// A released-tag heuristic for PIN §13.3: `MAJOR.MINOR.PATCH` with an
/// optional `-prerelease` suffix, digits-and-dots only in the numeric
/// part. Rejects branch-shaped strings like `main`, `HEAD`, or anything
/// containing `/`.
pub fn looks_like_released_tag(s: &str) -> bool {
    let core = s.split('-').next().unwrap_or(s);
    let parts: Vec<&str> = core.split('.').collect();
    parts.len() == SEMVER_COMPONENT_COUNT
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_hex_sha_accepts_forty_hex_chars() {
        assert!(is_full_hex_sha(&"a".repeat(40)));
        assert!(is_full_hex_sha("9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82"));
    }

    #[test]
    fn full_hex_sha_rejects_wrong_length_or_non_hex() {
        assert!(!is_full_hex_sha("main"));
        assert!(!is_full_hex_sha(&"a".repeat(39)));
        assert!(!is_full_hex_sha(&"a".repeat(41)));
        assert!(!is_full_hex_sha(&"g".repeat(40)));
    }

    #[test]
    fn kebab_case_accepts_lowercase_with_hyphens() {
        assert!(is_kebab_case("gemma-3-4b-it"));
        assert!(is_kebab_case("v11"));
    }

    #[test]
    fn kebab_case_rejects_bad_shapes() {
        assert!(!is_kebab_case(""));
        assert!(!is_kebab_case("Gemma-3"));
        assert!(!is_kebab_case("gemma_3"));
        assert!(!is_kebab_case("-gemma"));
        assert!(!is_kebab_case("gemma-"));
    }

    #[test]
    fn released_tag_accepts_semver_and_prerelease() {
        assert!(looks_like_released_tag("0.14.2"));
        assert!(looks_like_released_tag("1.0.0"));
        assert!(looks_like_released_tag("0.14.2-rc1"));
    }

    #[test]
    fn released_tag_rejects_branch_shapes() {
        assert!(!looks_like_released_tag("main"));
        assert!(!looks_like_released_tag("HEAD"));
        assert!(!looks_like_released_tag("feature/foo"));
        assert!(!looks_like_released_tag("0.14"));
        assert!(!looks_like_released_tag("0.14.2.1"));
    }
}
