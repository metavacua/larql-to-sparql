//! Kernel identity and maturity (spec §10).
//!
//! Deliberately outside the traversal's vocabulary. Traversal answers "are the
//! selected bytes sufficient for this programme?" and stops at reference
//! executability; the kernel registry separately answers "what optimised
//! implementation can execute them?". Keeping maturity in a different module
//! from operand availability is what stops the second question leaking into
//! the first — and a traversal that could name a kernel would be a traversal
//! that had already chosen one.

/// How mature the execution path for an operand is (§10's ladder).
///
/// Deliberately excludes `Representable` and `Reference`: those are not kernel
/// bindings, and are carried by [`OperandCapability`] variants instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelMaturity {
    Grouped,
    Dispatched,
    Production,
}

impl KernelMaturity {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Grouped => "grouped",
            Self::Dispatched => "dispatched",
            Self::Production => "production",
        }
    }
}

/// Opaque handle to a registered kernel.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(pub String);

impl KernelId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maturity_orders_from_grouped_to_production() {
        assert!(KernelMaturity::Grouped < KernelMaturity::Dispatched);
        assert!(KernelMaturity::Dispatched < KernelMaturity::Production);
    }

    #[test]
    fn the_ladder_excludes_the_non_kernel_rungs() {
        // Representable and Reference are not kernel bindings. They are carried
        // by the traversal's own vocabulary, so a kernel maturity can never be
        // used to mean "no kernel".
        let names: Vec<&str> = [
            KernelMaturity::Grouped,
            KernelMaturity::Dispatched,
            KernelMaturity::Production,
        ]
        .iter()
        .map(|m| m.name())
        .collect();
        assert_eq!(names, vec!["grouped", "dispatched", "production"]);
        assert!(!names.contains(&"reference"));
        assert!(!names.contains(&"representable"));
    }

    #[test]
    fn maturity_sorts_so_the_best_binding_is_the_maximum() {
        // Kernel binding ranks candidates; `max` must mean "most mature".
        let mut v = [
            KernelMaturity::Production,
            KernelMaturity::Grouped,
            KernelMaturity::Dispatched,
        ];
        v.sort();
        assert_eq!(v.last(), Some(&KernelMaturity::Production));
    }

    #[test]
    fn kernel_ids_are_distinct_by_name() {
        assert_eq!(KernelId::new("a"), KernelId::new("a"));
        assert_ne!(KernelId::new("a"), KernelId::new("b"));
    }

    #[test]
    fn a_kernel_id_carries_its_name_verbatim() {
        assert_eq!(
            KernelId::new("grouped_experts_q6k").0,
            "grouped_experts_q6k"
        );
    }
}
