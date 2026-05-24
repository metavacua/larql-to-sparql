//! Datalog rules for wasm32 call-graph containment check.
//!
//! Determines whether any non-intrinsic host import is reachable from at least
//! one export root (containment violation = NATIVE; none = WASM-SAFE).

use ascent::ascent;

ascent! {
    // ── Input facts ───────────────────────────────────────────────────────────

    /// Static call edge: caller calls callee (both are function indices).
    relation calls(u32, u32);

    /// Function index that imports a non-intrinsic host symbol.
    relation is_non_intrinsic_import(u32);

    /// Wasm export (call graph root): func_index.
    relation is_root(u32);

    // ── Transitive call closure ───────────────────────────────────────────────

    relation calls_tc(u32, u32);
    calls_tc(a, b) <-- calls(a, b);
    calls_tc(a, c) <-- calls_tc(a, b), calls(b, c);

    // ── Reachability from exports ─────────────────────────────────────────────

    relation is_reachable(u32);
    is_reachable(f) <-- is_root(f);
    is_reachable(f) <-- is_root(r), calls_tc(r, f);

    // ── Containment: transitive taint from non-intrinsic imports ─────────────

    relation violates_containment(u32);
    violates_containment(f) <-- is_non_intrinsic_import(f);
    violates_containment(caller) <-- calls_tc(caller, callee), violates_containment(callee);

    // ── Containment violations reachable from exports ─────────────────────────

    relation containment_violation(u32);
    containment_violation(f) <--
        is_root(root), calls_tc(root, f), violates_containment(f);
    containment_violation(root) <--
        is_root(root), violates_containment(root);
}

/// Result of running the Datalog containment analysis.
pub struct AnalysisResult {
    pub prog: AscentProgram,
}

impl AnalysisResult {
    /// True iff no containment violations are reachable from exports.
    pub fn is_sandbox_contained(&self) -> bool {
        self.prog.containment_violation.is_empty()
    }

    pub fn containment_violation_indices(&self) -> Vec<u32> {
        self.prog
            .containment_violation
            .iter()
            .map(|(idx,)| *idx)
            .collect()
    }

}

/// Run containment analysis given extracted wasm facts.
pub fn analyze(
    calls: Vec<(u32, u32)>,
    non_intrinsic_imports: Vec<u32>,
    roots: Vec<u32>,
) -> AnalysisResult {
    let mut prog = AscentProgram::default();
    prog.calls = calls;
    prog.is_non_intrinsic_import = non_intrinsic_imports.into_iter().map(|x| (x,)).collect();
    prog.is_root = roots.into_iter().map(|x| (x,)).collect();
    prog.run();
    AnalysisResult { prog }
}
