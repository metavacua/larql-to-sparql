//! Crate-wide named facts — small numeric/string constants shared by
//! more than one module, so no call site (or test) carries a bare
//! literal for them.

/// Length of a full git commit SHA, in hex characters.
pub const GIT_COMMIT_SHA_HEX_LEN: usize = 40;

/// Number of dot-separated components in a `MAJOR.MINOR.PATCH` semver core.
pub const SEMVER_COMPONENT_COUNT: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_sha_hex_len_matches_a_real_sha() {
        assert_eq!(
            "9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82".len(),
            GIT_COMMIT_SHA_HEX_LEN
        );
    }

    #[test]
    fn semver_component_count_matches_major_minor_patch() {
        assert_eq!("0.14.2".split('.').count(), SEMVER_COMPONENT_COUNT);
    }
}
