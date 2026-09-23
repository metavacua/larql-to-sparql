use crate::ffn::moe_remote::runtime::{ENV_MOE_NO_SPLIT, ENV_MOE_TIMING, ENV_MOE_TOP_K};
use crate::layer_graph::generate::policy::TokenSelectionPolicy;
#[cfg(test)]
use larql_compute::options::ENV_SKIP_MOE;

#[derive(Clone, Debug)]
pub(super) struct GridRuntimeConfig {
    pub moe_top_k_override: Option<usize>,
    pub skip_moe: bool,
    pub timing_enabled: bool,
    pub split_disabled: bool,
    pub token_policy: TokenSelectionPolicy,
}

impl GridRuntimeConfig {
    pub fn from_env() -> Self {
        // Read through the override-aware `options` helpers so tests can toggle
        // these via the thread-local override (NOT `std::env::set_var`, which
        // races concurrent `getenv` on the decode path → SIGSEGV). Behaviour is
        // identical: `env_usize` = `var().ok().and_then(parse)`, `env_flag` =
        // `var().is_ok()` (presence-as-truth, any value incl. empty).
        use larql_compute::options::{env_flag, env_usize, skip_moe_enabled};
        Self {
            moe_top_k_override: env_usize(ENV_MOE_TOP_K),
            // Canonical LARQL_SKIP_MOE with the historical unprefixed
            // SKIP_MOE as a loud deprecated alias (dec-readiness §3e).
            skip_moe: skip_moe_enabled(),
            timing_enabled: env_flag(ENV_MOE_TIMING),
            split_disabled: env_flag(ENV_MOE_NO_SPLIT),
            token_policy: TokenSelectionPolicy::from_env(),
        }
    }

    pub fn moe_top_k(&self, arch_top_k: usize) -> usize {
        self.moe_top_k_override
            .map(|k| k.clamp(1, arch_top_k))
            .unwrap_or(arch_top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_returns_default_when_no_vars_set() {
        // Force every MoE var to act as unset via the thread-local override
        // (NOT `std::env::set_var`, which races concurrent `getenv` on the
        // decode path → SIGSEGV) so we exercise the default arms of every
        // `env_usize` / `env_flag` read regardless of the ambient process env.
        // Cleared on drop; per-thread, so no cross-test leakage.
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                larql_compute::options::clear_fast_path_overrides();
            }
        }
        let _clear = Clear;
        for var in [
            ENV_MOE_TOP_K,
            ENV_SKIP_MOE,
            ENV_MOE_TIMING,
            ENV_MOE_NO_SPLIT,
        ] {
            larql_compute::options::set_env_override(var, None);
        }

        let cfg = GridRuntimeConfig::from_env();
        assert!(cfg.moe_top_k_override.is_none());
        assert!(!cfg.skip_moe);
    }

    #[test]
    fn moe_top_k_falls_back_to_arch_when_no_override() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: None,
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        assert_eq!(cfg.moe_top_k(8), 8);
    }

    #[test]
    fn moe_top_k_clamps_override_to_arch_max() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: Some(99),
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        // Override 99 > arch 8 → clamped to 8.
        assert_eq!(cfg.moe_top_k(8), 8);
    }

    #[test]
    fn moe_top_k_clamps_override_to_min_one() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: Some(0),
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        // Override 0 < 1 → clamped to 1.
        assert_eq!(cfg.moe_top_k(8), 1);
    }

    #[test]
    fn moe_top_k_uses_override_when_in_range() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: Some(4),
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        assert_eq!(cfg.moe_top_k(8), 4);
    }
}
