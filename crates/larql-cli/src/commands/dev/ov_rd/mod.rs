// `ov_rd` is a research / dev-tooling module — its priorities are
// clarity over micro-optimisation, and consistency with the surrounding
// linear-algebra and program-synthesis idioms. Several clippy lints fire
// on patterns that are intentional here:
//
//  - `needless_range_loop` — indexed for-loops in covariance / Gram
//    matrix / power-iteration code are clearer than zipped iterators.
//  - `for_kv_map` — `for ((head, config), _) in codebooks` reads the
//    (head, config) pattern explicitly; `.keys()` would force a
//    re-destructure.
//  - `branches_sharing_code` / `if_same_then_else` — the explicit
//    duplication in `oracle_pq.rs` makes the per-branch logic
//    auditable.
//  - Mechanical micro-style nudges (`map_identity`, `manual_is_multiple_of`,
//    `useless_format`, `iter_cloned_collect`, `manual_repeat_n`,
//    `map_flatten`, `map_entry`, `needless_deref`, `iter_overeager_cloned`,
//    `iter_nth_zero`, `ptr_arg`, `io_other_error`, `excessive_precision`,
//    `empty_line_after_doc_comments`) — all suppressed module-wide as
//    research-code style preferences rather than real issues.
#![allow(
    clippy::needless_range_loop,
    clippy::for_kv_map,
    clippy::branches_sharing_code,
    clippy::if_same_then_else,
    clippy::map_identity,
    clippy::manual_is_multiple_of,
    clippy::useless_format,
    clippy::iter_cloned_collect,
    clippy::manual_repeat_n,
    clippy::map_flatten,
    clippy::map_entry,
    clippy::needless_borrow,
    clippy::unnecessary_to_owned,
    clippy::manual_contains,
    clippy::iter_nth_zero,
    clippy::ptr_arg,
    clippy::io_other_error,
    clippy::excessive_precision,
    clippy::empty_line_after_doc_comments,
    clippy::explicit_auto_deref
)]

// Per-module `forbid(unsafe_code)`, not crate-wide (pattern 18): the
// modules left unmarked below call `ndarray::s![...]`, whose macro
// expansion embeds its own `#[allow(unsafe_code)] unsafe {...}` --
// `forbid` can't be locally overridden even through a macro expansion,
// so forbid has to stop one level above each such file. Every module
// here IS a leaf CI-confirmed either way (workflow run 31464601274):
// leaving one unmarked is a deliberate exemption, not an oversight.
//
// Per-module `#[cfg(not(target_arch = "wasm32"))]` (wasm32v1-none
// gating, round 1) -- every module below that takes a live
// `&tokenizers::Tokenizer`, a `clap::Args`/`Subcommand` derive, or
// `std::fs`/`std::time::Instant` directly is native-gated. This axis is
// independent of the `forbid(unsafe_code)` one above: both attributes
// stack on the same item where both apply. `eval_program` and
// `induce_program` are each gated as a whole rather than split --
// their own `mod.rs` is the sole external entry point into each
// subtree and is itself wholesale-native, so this also excludes their
// factually-portable children (`eval_program::report`;
// `induce_program::{context,evaluate,guard_synth,localize,proposal}`)
// from the wasm32 build. They have no portable-side caller today (only
// their own native parent and each other), so this is not a loss --
// matches the `compile_cmd` precedent in `commands/extraction/mod.rs`.
#[forbid(unsafe_code)]
mod address;
mod basis;
#[cfg(not(target_arch = "wasm32"))]
mod capture;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod cmd;
#[cfg(not(target_arch = "wasm32"))]
mod edit_catalog;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod eval_program;
#[cfg(not(target_arch = "wasm32"))]
mod gamma_address;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod induce_program;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod input;
#[forbid(unsafe_code)]
mod metal_backend;
#[forbid(unsafe_code)]
mod metrics;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod normalize_program;
#[cfg(not(target_arch = "wasm32"))]
mod oracle;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod oracle_pq;
#[cfg(not(target_arch = "wasm32"))]
mod oracle_pq_address;
#[forbid(unsafe_code)]
mod oracle_pq_eval;
#[cfg(not(target_arch = "wasm32"))]
mod oracle_pq_fit;
mod oracle_pq_forward;
mod oracle_pq_mode_d;
#[forbid(unsafe_code)]
mod oracle_pq_reports;
#[cfg(not(target_arch = "wasm32"))]
mod oracle_pq_stability;
#[forbid(unsafe_code)]
mod pq;
#[cfg(not(target_arch = "wasm32"))]
mod pq_exception;
#[cfg(not(target_arch = "wasm32"))]
mod probe_program_class;
#[forbid(unsafe_code)]
mod program;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod program_cache;
#[forbid(unsafe_code)]
mod reports;
#[forbid(unsafe_code)]
mod runtime;
#[cfg(not(target_arch = "wasm32"))]
mod sanity;
#[cfg(not(target_arch = "wasm32"))]
mod static_replace;
#[forbid(unsafe_code)]
mod stats;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
mod synthesize_program;
#[forbid(unsafe_code)]
mod types;
#[cfg(not(target_arch = "wasm32"))]
mod zero_ablate;
