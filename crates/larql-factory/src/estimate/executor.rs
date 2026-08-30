//! Executor-class recommendation (docs/vindex-factory.md §2.1).

use serde::Serialize;

/// PIN §13.4: "40 GiB is a guess; G0 measures it." Named here so the
/// guess has exactly one home instead of a bare literal at the call
/// site.
pub const INLINE_EXECUTOR_THRESHOLD_GIB: f64 = 40.0;

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Which class of executor a build's declared working set fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutorClass {
    /// Ephemeral Fly machine — tiny models only.
    Inline,
    /// Leased rig worker (Vast/Lambda) — everything else.
    Leased,
}

impl ExecutorClass {
    /// `auto`/`inline`/`leased` string form, matching the recipe
    /// schema's `budget.executor` field.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Leased => "leased",
        }
    }
}

/// Recommend an executor class for a working set of `total_bytes`.
pub fn recommend_executor(total_bytes: u64) -> ExecutorClass {
    let gib = total_bytes as f64 / BYTES_PER_GIB;
    if gib <= INLINE_EXECUTOR_THRESHOLD_GIB {
        ExecutorClass::Inline
    } else {
        ExecutorClass::Leased
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_working_set_recommends_inline() {
        assert_eq!(recommend_executor(1024), ExecutorClass::Inline);
    }

    #[test]
    fn working_set_at_threshold_recommends_inline() {
        let bytes = (INLINE_EXECUTOR_THRESHOLD_GIB * BYTES_PER_GIB) as u64;
        assert_eq!(recommend_executor(bytes), ExecutorClass::Inline);
    }

    #[test]
    fn working_set_over_threshold_recommends_leased() {
        let bytes = ((INLINE_EXECUTOR_THRESHOLD_GIB + 1.0) * BYTES_PER_GIB) as u64;
        assert_eq!(recommend_executor(bytes), ExecutorClass::Leased);
    }

    #[test]
    fn as_str_matches_recipe_schema_values() {
        assert_eq!(ExecutorClass::Inline.as_str(), "inline");
        assert_eq!(ExecutorClass::Leased.as_str(), "leased");
    }

    #[test]
    fn serialises_lowercase() {
        assert_eq!(
            serde_json::to_string(&ExecutorClass::Leased).unwrap(),
            "\"leased\""
        );
    }
}
