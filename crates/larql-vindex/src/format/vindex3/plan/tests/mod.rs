//! Tests for the semantic representability plan.

mod capability;
mod carriage;
mod compare;
mod gemma4;
mod hybrid_linear_attention;
mod qw35d_admission;
mod semantics;
mod system;

/// Fixtures live one level up so the graph tests share them.
use super::tests_support as support;
