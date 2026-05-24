//! Datalog rules for wasm32 call-graph closure verification.
//!
//! Four certification tiers:
//!   Tier 0  — fails cargo check --target wasm32-unknown-unknown
//!   Tier 1  — compiles; does not run in Node.js
//!   Tier 2  — runs in Node.js; may have call_indirect or non-intrinsic imports
//!   Tier 3  — Reified: closed call graph; F_defined = F_reachable; Firefox-portable
//!   Native  — no lib target; host OS/IO layer only
//!
//! Import capability labels (diagnostic only — not certification tiers):
//!   local   — fs/IPC/LAN: patterns like readFile, open, stat, spawn, pipe
//!   remote  — WAN/HTTP/WS: everything else (conservative; blocks Tier 3)
//!
//! The `ascent` macro IS the resolution system — Horn clause resolution over a
//! finite closed-world fact base.  Input facts from wasm_facts.rs; optional
//! MIR-level facts from mir_facts.rs (nightly `mir-analysis` feature).

use ascent::ascent;

ascent! {
    // ── Input facts (populated by analyze() from wasm_facts) ─────────────────

    /// Static call edge: caller calls callee (both are function indices).
    relation calls(u32, u32);

    /// Function index that imports a non-intrinsic host symbol.
    relation is_non_intrinsic_import(u32);

    /// Non-intrinsic import classified as local capability (fs/IPC/LAN).
    relation is_local_import(u32);

    /// Non-intrinsic import classified as remote capability (WAN/HTTP/WS).
    relation is_remote_import(u32);

    /// Function index that contains at least one call_indirect.
    relation has_indirect_call(u32);

    /// Wasm export (call graph root): func_index.
    relation is_root(u32);

    /// Every function index that exists in the binary (0..total_func_count).
    /// Used to derive orphaned = defined but not reachable.
    relation is_defined(u32);

    // ── Derived: transitive call closure ─────────────────────────────────────

    /// Transitive call: a can reach b through any number of static calls.
    relation calls_tc(u32, u32);
    calls_tc(a, b) <-- calls(a, b);
    calls_tc(a, c) <-- calls_tc(a, b), calls(b, c);

    // ── Reachability ──────────────────────────────────────────────────────────

    /// Function is reachable from at least one export root.
    relation is_reachable(u32);
    is_reachable(f) <-- is_root(f);
    is_reachable(f) <-- is_root(r), calls_tc(r, f);

    // ── Reification criterion: F_defined = F_reachable ───────────────────────
    // Orphaned = defined but not reachable from any export.
    // Zero orphaned functions is necessary for Tier 3.

    relation orphaned(u32);
    orphaned(f) <-- is_defined(f), !is_reachable(f);

    // ── Containment violation ─────────────────────────────────────────────────
    // A non-intrinsic host import IS a sandbox breach — the only true source of
    // undecidability in this analysis (OS/IO resources outside the wasm sandbox).

    relation violates_containment(u32);
    violates_containment(f) <-- is_non_intrinsic_import(f);
    violates_containment(caller) <--
        calls_tc(caller, callee),
        violates_containment(callee);

    // ── Unresolved dispatch ───────────────────────────────────────────────────
    // call_indirect = dynamic dispatch.

    relation unresolved_dispatch(u32);
    unresolved_dispatch(f) <-- has_indirect_call(f);

    // ── Containment violation witnesses (reachable from exports) ─────────────

    relation containment_violation(u32);
    containment_violation(f) <--
        is_root(root),
        calls_tc(root, f),
        violates_containment(f);
    containment_violation(root) <--
        is_root(root),
        violates_containment(root);

    // ── Dispatch witnesses (reachable from exports) ───────────────────────────

    relation dispatch_witness(u32);
    dispatch_witness(f) <--
        is_root(root),
        calls_tc(root, f),
        unresolved_dispatch(f);
    dispatch_witness(root) <--
        is_root(root),
        unresolved_dispatch(root);

    // ── Import capability reachability (diagnostic) ───────────────────────────

    /// Local-capability import reachable from at least one export.
    relation local_reachable(u32);
    local_reachable(f) <-- is_local_import(f), is_reachable(f);

    /// Remote-capability import reachable from at least one export.
    relation remote_reachable(u32);
    remote_reachable(f) <-- is_remote_import(f), is_reachable(f);
}

/// Certification tier claimed in `[package.metadata.wasm-cert]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum WasmTier {
    #[default]
    Level0,
    Level1,
    Level2,
    Level3,
}

impl WasmTier {
    pub fn from_str(s: &str) -> Self {
        match s.trim() {
            "1" => WasmTier::Level1,
            "2" => WasmTier::Level2,
            "3" => WasmTier::Level3,
            _ => WasmTier::Level0,
        }
    }

    /// Parse from the legacy `claimed-level: u8` field.
    pub fn from_u8(n: u8) -> Self {
        match n {
            1 => WasmTier::Level1,
            2 => WasmTier::Level2,
            3 | 4 | 5 | 6 => WasmTier::Level3,
            _ => WasmTier::Level0,
        }
    }
}

impl std::fmt::Display for WasmTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmTier::Level0 => write!(f, "0"),
            WasmTier::Level1 => write!(f, "1"),
            WasmTier::Level2 => write!(f, "2"),
            WasmTier::Level3 => write!(f, "3"),
        }
    }
}

/// Partition label for call-graph static analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmPartition {
    /// Call graph closed: no call_indirect, no non-intrinsic imports.
    Static,
    /// call_indirect present; dispatch unclassified (no MIR analysis).
    Dynamic,
    /// Non-intrinsic host imports reachable from exports, or no lib target.
    Native,
}

impl std::fmt::Display for WasmPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmPartition::Static => write!(f, "STATIC"),
            WasmPartition::Dynamic => write!(f, "DYNAMIC"),
            WasmPartition::Native => write!(f, "NATIVE"),
        }
    }
}

/// Result of running the Datalog analysis.
pub struct AnalysisResult {
    pub prog: AscentProgram,
}

impl AnalysisResult {
    /// True iff no containment violations are reachable from exports.
    pub fn is_sandbox_contained(&self) -> bool {
        self.prog.containment_violation.is_empty()
    }

    /// True iff no dispatch witnesses are reachable from exports.
    pub fn is_statically_resolved(&self) -> bool {
        self.prog.dispatch_witness.is_empty()
    }

    pub fn containment_violation_indices(&self) -> Vec<u32> {
        self.prog.containment_violation.iter().map(|(idx,)| *idx).collect()
    }

    pub fn dispatch_witness_indices(&self) -> Vec<u32> {
        self.prog.dispatch_witness.iter().map(|(idx,)| *idx).collect()
    }

    pub fn orphaned_indices(&self) -> Vec<u32> {
        self.prog.orphaned.iter().map(|(idx,)| *idx).collect()
    }

    pub fn local_reachable_indices(&self) -> Vec<u32> {
        self.prog.local_reachable.iter().map(|(idx,)| *idx).collect()
    }

    pub fn remote_reachable_indices(&self) -> Vec<u32> {
        self.prog.remote_reachable.iter().map(|(idx,)| *idx).collect()
    }

    pub fn reachable_indices(&self) -> Vec<u32> {
        self.prog.is_reachable.iter().map(|(idx,)| *idx).collect()
    }

    /// Assign the stable-path partition label (no MIR facts available).
    pub fn partition_stable(&self) -> WasmPartition {
        if !self.is_sandbox_contained() {
            WasmPartition::Native
        } else if !self.is_statically_resolved() {
            WasmPartition::Dynamic
        } else {
            WasmPartition::Static
        }
    }
}

/// Run the Datalog rules given populated input facts.
pub fn analyze(
    calls: Vec<(u32, u32)>,
    non_intrinsic_imports: Vec<u32>,
    local_imports: Vec<u32>,
    remote_imports: Vec<u32>,
    indirect_calls: Vec<u32>,
    roots: Vec<u32>,
    total_func_count: u32,
) -> AnalysisResult {
    let mut prog = AscentProgram::default();
    prog.calls = calls;
    prog.is_non_intrinsic_import = non_intrinsic_imports.into_iter().map(|x| (x,)).collect();
    prog.is_local_import = local_imports.into_iter().map(|x| (x,)).collect();
    prog.is_remote_import = remote_imports.into_iter().map(|x| (x,)).collect();
    prog.has_indirect_call = indirect_calls.into_iter().map(|x| (x,)).collect();
    prog.is_root = roots.into_iter().map(|x| (x,)).collect();
    prog.is_defined = (0..total_func_count).map(|x| (x,)).collect();
    prog.run();
    AnalysisResult { prog }
}
