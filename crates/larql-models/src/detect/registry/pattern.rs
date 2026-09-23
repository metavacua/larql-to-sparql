//! How a `model_type` string is matched against a registry entry.

/// How a `model_type` string is matched against a registry entry.
#[derive(Clone, Copy, Debug)]
pub enum ModelTypeMatch {
    /// Exact string match.
    Exact(&'static str),
    /// True if the declared `model_type` starts with this prefix.
    Prefix(&'static str),
}

impl ModelTypeMatch {
    pub(super) fn matches(self, model_type: &str) -> bool {
        match self {
            Self::Exact(s) => model_type == s,
            Self::Prefix(p) => model_type.starts_with(p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_only_matches_the_exact_string() {
        let m = ModelTypeMatch::Exact("mistral");
        assert!(m.matches("mistral"));
        assert!(!m.matches("mistral-nemo"));
        assert!(!m.matches("mist"));
    }

    #[test]
    fn prefix_matches_any_extension() {
        let m = ModelTypeMatch::Prefix("gemma3");
        assert!(m.matches("gemma3"));
        assert!(m.matches("gemma3_text"));
        assert!(m.matches("gemma3n"));
        assert!(!m.matches("gemma2"));
    }

    #[test]
    fn prefix_does_not_match_a_shorter_string() {
        assert!(!ModelTypeMatch::Prefix("gemma3").matches("gemma"));
    }
}
