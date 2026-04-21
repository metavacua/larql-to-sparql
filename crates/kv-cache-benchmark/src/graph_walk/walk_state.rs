/// Walk state: derived from tokens, not stored.
///
/// The walk state IS the tokens themselves. No separate state object.
/// State is parsed from context to determine:
///   - Last entity mentioned
///   - Current relation
///   - Computational mode (factual, arithmetic, conversation, etc.)
///
/// This is what makes the graph walk O(1) per conversation:
/// the state doesn't grow, it's just a parse of the current context.

/// Parsed walk state from token context.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WalkState {
    /// The last entity mentioned in the context.
    pub last_entity: Option<String>,
    /// The current relation being queried.
    pub current_relation: Option<String>,
    /// The computational mode.
    pub mode: WalkMode,
    /// Which tier this query resolves at.
    pub tier: WalkTier,
}

/// Walk mode: what kind of computation the graph walk performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum WalkMode {
    /// Factual query: entity × relation → answer
    Factual,
    /// Arithmetic: number × operation → result
    Arithmetic,
    /// Code: syntax pattern → completion
    Code,
    /// Conversation: open-ended generation
    Conversation,
    /// Unknown: needs classification
    Unknown,
}

/// Walk tier: how the query is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum WalkTier {
    /// Tier A: cached template walk. Known template, entity KNN only. <0.1ms.
    CachedTemplate,
    /// Tier B: dynamic graph walk. Full routing table lookup. 1-5ms.
    DynamicWalk,
    /// Tier C: Markov RS fallback. Full forward pass for free-form generation. ~200ms.
    MarkovFallback,
}

impl WalkState {
    /// Parse walk state from a sequence of token strings (simplified).
    pub fn from_tokens(tokens: &[&str]) -> Self {
        let text = tokens.join(" ");
        let lower = text.to_lowercase();

        // Simple template detection
        let (mode, relation) = if lower.contains("capital of") {
            (WalkMode::Factual, Some("capital-of".to_string()))
        } else if lower.contains("born in") || lower.contains("was born") {
            (WalkMode::Factual, Some("birthplace".to_string()))
        } else if lower.contains("currency of") {
            (WalkMode::Factual, Some("currency-of".to_string()))
        } else if lower.contains("freezes at") || lower.contains("boils at") {
            (WalkMode::Factual, Some("physical-property".to_string()))
        } else if lower.contains('+') || lower.contains('*') || lower.contains("divided") {
            (WalkMode::Arithmetic, None)
        } else if lower.contains("def ") || lower.contains("fn ") || lower.contains("function") {
            (WalkMode::Code, None)
        } else {
            (WalkMode::Unknown, None)
        };

        // Extract last entity (simplified: last capitalised word before relation keyword)
        let last_entity = extract_entity(&text);

        let tier = match mode {
            WalkMode::Factual => {
                if relation.is_some() {
                    WalkTier::CachedTemplate
                } else {
                    WalkTier::DynamicWalk
                }
            }
            WalkMode::Unknown | WalkMode::Conversation => WalkTier::MarkovFallback,
            _ => WalkTier::DynamicWalk,
        };

        WalkState {
            last_entity,
            current_relation: relation,
            mode,
            tier,
        }
    }

    /// Estimated latency for this walk tier in microseconds.
    pub fn estimated_latency_us(&self) -> f64 {
        match self.tier {
            WalkTier::CachedTemplate => 100.0,    // <0.1ms
            WalkTier::DynamicWalk => 3_000.0,     // ~3ms
            WalkTier::MarkovFallback => 200_000.0, // ~200ms
        }
    }
}

/// Extract the last likely entity from text (simplified heuristic).
fn extract_entity(text: &str) -> Option<String> {
    // Look for capitalised words that aren't at the start of a sentence
    let words: Vec<&str> = text.split_whitespace().collect();
    for word in words.iter().rev() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() > 1
            && clean.chars().next().map_or(false, |c| c.is_uppercase())
            && !["The", "What", "Who", "Where", "How", "Is", "Was", "Tell", "A"].contains(&clean)
        {
            return Some(clean.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factual_query_detection() {
        let state = WalkState::from_tokens(&["What", "is", "the", "capital", "of", "France"]);
        assert_eq!(state.mode, WalkMode::Factual);
        assert_eq!(state.current_relation.as_deref(), Some("capital-of"));
        assert_eq!(state.last_entity.as_deref(), Some("France"));
        assert_eq!(state.tier, WalkTier::CachedTemplate);
    }

    #[test]
    fn test_birthplace_query() {
        let state = WalkState::from_tokens(&["Mozart", "was", "born", "in"]);
        assert_eq!(state.mode, WalkMode::Factual);
        assert_eq!(state.current_relation.as_deref(), Some("birthplace"));
        assert_eq!(state.last_entity.as_deref(), Some("Mozart"));
    }

    #[test]
    fn test_unknown_falls_back() {
        let state = WalkState::from_tokens(&["tell", "me", "a", "joke"]);
        assert_eq!(state.tier, WalkTier::MarkovFallback);
    }

    #[test]
    fn test_tier_latencies() {
        let cached = WalkState {
            last_entity: None,
            current_relation: None,
            mode: WalkMode::Factual,
            tier: WalkTier::CachedTemplate,
        };
        assert!(cached.estimated_latency_us() < 1_000.0);

        let fallback = WalkState {
            last_entity: None,
            current_relation: None,
            mode: WalkMode::Conversation,
            tier: WalkTier::MarkovFallback,
        };
        assert!(fallback.estimated_latency_us() > 100_000.0);
    }
}
