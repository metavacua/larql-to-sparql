#![feature(rustc_private)]
#![warn(unused_extern_crates)]

// A list of available compiler crates can be found here:
// https://doc.rust-lang.org/nightly/nightly-rustc/
extern crate rustc_errors;
extern crate rustc_hir;

use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};

dylint_linting::declare_late_lint! {
    /// ### What it does
    ///
    /// Flags a free function whose name starts with `render_` and that has
    /// no `#[cfg(...)]` guard at all, suggesting it be gated
    /// `#[cfg(not(target_arch = "wasm32"))]`.
    ///
    /// ### Why is this bad?
    ///
    /// It isn't, in general -- this is a deliberately narrow
    /// proof-of-concept, not a real analysis. See "Known problems".
    ///
    /// The real motivating scenario this stands in for: a portable helper
    /// (e.g. `render_overview`) whose *only* caller is itself
    /// `#[cfg(not(target_arch = "wasm32"))]`-gated. On a `wasm32` build
    /// that caller disappears, the helper becomes unreachable, and rustc's
    /// own `dead_code` lint (`rustc_passes::dead`) fires on it -- but
    /// `dead_code` is a hardcoded pass, not a pluggable `LintPass`, so it
    /// cannot carry a `span_lint_and_sugg`-style fix. This lint exists to
    /// show that a *pluggable* lint occupying that niche, with a real
    /// `MachineApplicable` suggestion, is buildable.
    ///
    /// ### Known problems
    ///
    /// This PoC does **not** perform the caller-graph reachability
    /// analysis that would make it a faithful replacement for the
    /// wasm32-orphaned-by-a-gated-caller case above. Reimplementing that
    /// (walking the crate-visible call graph via HIR and checking whether
    /// every path to an item passes through a `cfg(not(target_arch =
    /// "wasm32"))`-gated node) is exactly `rustc_passes::dead`'s job and
    /// is substantial on its own -- out of scope for this bounded
    /// prototype. Instead the lint uses a naming-convention stand-in
    /// (`render_` prefix) as the trigger condition. Consequently it:
    /// - over-fires on any `render_*` fn regardless of whether it is
    ///   actually wasm32-orphaned or even dead at all, and
    /// - under-fires on a genuinely wasm32-orphaned fn named anything
    ///   else.
    ///
    /// It also does not check whether the item already carries a
    /// `target_arch` cfg guard before firing again on a re-run: `#[cfg]`
    /// attributes that evaluate to "keep this item" are stripped from the
    /// surviving item during macro expansion, before HIR (and therefore
    /// before any `LateLintPass`) ever sees them, so there is no cheap
    /// HIR-level "already gated" check to make. A real, upstreamable
    /// version of this lint would need a source-text or early-AST check
    /// instead. See this crate's `README.md` for the fuller writeup.
    ///
    /// The mechanism demonstrated here -- a working `LateLintPass`,
    /// `span_lint_and_sugg`, `Applicability::MachineApplicable` via
    /// `Span::shrink_to_lo()`, and a real `cargo dylint --fix` applying
    /// it -- is real and is the point of this crate.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn render_overview() {}
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust
    /// #[cfg(not(target_arch = "wasm32"))]
    /// fn render_overview() {}
    /// ```
    pub WASM32_ORPHANED_PORTABLE,
    Warn,
    "portable fn named `render_*` with no cfg guard (PoC marker, not real reachability analysis -- see Known problems)"
}

impl<'tcx> LateLintPass<'tcx> for Wasm32OrphanedPortable {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &Item<'tcx>) {
        if !matches!(item.kind, ItemKind::Fn { .. }) {
            return;
        }
        if item.span.from_expansion() {
            return;
        }
        let Some(ident) = item.kind.ident() else {
            return;
        };
        if !ident.as_str().starts_with("render_") {
            return;
        }

        span_lint_and_sugg(
            cx,
            WASM32_ORPHANED_PORTABLE,
            item.span.shrink_to_lo(),
            "portable fn `render_*` has no `target_arch` cfg guard",
            "consider gating it out of wasm32 builds",
            "#[cfg(not(target_arch = \"wasm32\"))]\n".to_string(),
            Applicability::MachineApplicable,
        );
    }
}

#[test]
fn ui() {
    // Deliberately not wired up as a real golden-file UI test for this
    // bounded prototype (see this branch's task writeup / final report):
    // every `.stderr` mismatch would cost a full CI round trip, since
    // local `cargo` invocations are forbidden on this dev machine. Firing
    // + fix evidence for this PoC comes from `cargo dylint` /
    // `cargo dylint --fix` run directly against `dylint/example` in CI
    // (see `.github/workflows/dylint-dead-code-prototype.yml`), not from
    // this test.
}
