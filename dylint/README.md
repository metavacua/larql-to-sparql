# `wasm32_orphaned_portable` -- dylint dead-code-fix prototype

Bounded proof-of-concept for `experiment/dylint-dead-code-prototype`. Goal:
show that a *pluggable* lint (unlike rustc's own `dead_code`, implemented
in `rustc_passes::dead` and not a `LintPass`) can detect a
wasm32-orphaned-portable-item shape and carry a real, `MachineApplicable`
`cargo dylint --fix` suggestion.

## Layout

- `wasm32_orphaned_portable/` -- the lint library (`cdylib`), scaffolded
  from trailofbits/dylint's own currently-maintained `internal/template`
  (what `cargo dylint new` generates), **not** the older, now-archived
  `trailofbits/dylint-template` repo (which is pinned to
  `nightly-2022-02-24` / `dylint_linting` 2.0.1 and says as much in its own
  README -- rejected as stale for this reason, confirmed by reading it
  directly rather than assumed).
- `example/` -- a tiny standalone crate (`[workspace.metadata.dylint]
  libraries = [{ path = "../wasm32_orphaned_portable" }]`) used to
  demonstrate the lint firing and `--fix` applying its suggestion.

Both crates carry their own empty `[workspace]` table so they are
self-contained workspace roots -- the repo root's `Cargo.toml` lists
`members` as an explicit `crates/*` array (not a glob), so `dylint/` was
never at risk of being swept in, but the empty `[workspace]` tables are
still required so `cargo` doesn't walk up and try (and fail) to join the
outer workspace when run with `dylint/*` as the working directory.

## Detection heuristic (deliberately simplified -- see crate-level lint doc)

Real reachability analysis (does every path back to this item's callers
terminate at a `cfg(not(target_arch = "wasm32"))`-gated node?) is
`rustc_passes::dead`'s job and is substantial on its own. This PoC
sidesteps it: it fires on any free `fn` named `render_*` with no further
analysis. This is a stated, deliberate simplification of the *detection
heuristic* only -- the lint mechanics around it (`LateLintPass`,
`span_lint_and_sugg`, `Applicability::MachineApplicable` via
`Span::shrink_to_lo()`, and `cargo dylint --fix` applying the result) are
the real thing, not simulated.

One consequence worth recording: the lint does **not** check whether the
item already carries a `target_arch` cfg guard before firing (which would
make it non-idempotent across repeated `--fix` runs). That's not merely
"not implemented for time" -- attributes that evaluate to "keep this item"
during `#[cfg(...)]` processing are stripped from the item before HIR (and
therefore before any `LateLintPass`) sees it, so there's no cheap
HIR-attribute check available; a real version would need a source-text or
early-AST check instead.

## Toolchain

`nightly-2026-05-28` with `rustc-dev` + `llvm-tools-preview`, paired with
`dylint_linting = "6.0"` and `clippy_utils` pinned to git rev
`9fca3bc9fc2bc83c60bde26d18ed68f11564b228` -- copied verbatim from
dylint's own `internal/template/Cargo.toml` /
`internal/template/rust-toolchain.toml` (cloned fresh from
github.com/trailofbits/dylint for this task rather than reconstructed from
memory), since version mismatch between nightly / `dylint_linting` /
`clippy_utils` is the standard failure mode for dylint lint crates.
