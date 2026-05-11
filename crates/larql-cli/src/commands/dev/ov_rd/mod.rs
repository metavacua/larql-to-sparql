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

mod address;
mod basis;
mod capture;
pub mod cmd;
mod edit_catalog;
mod eval_program;
mod gamma_address;
mod induce_program;
mod input;
mod metal_backend;
mod metrics;
mod normalize_program;
mod oracle;
mod oracle_pq;
mod oracle_pq_address;
mod oracle_pq_eval;
mod oracle_pq_fit;
mod oracle_pq_forward;
mod oracle_pq_mode_d;
mod oracle_pq_reports;
mod oracle_pq_stability;
mod pq;
mod pq_exception;
mod probe_program_class;
mod program;
mod program_cache;
mod reports;
mod runtime;
mod sanity;
mod static_replace;
mod stats;
mod synthesize_program;
mod types;
mod zero_ablate;
