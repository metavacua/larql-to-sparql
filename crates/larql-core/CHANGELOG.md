# Changelog — larql-core

All notable changes to `larql-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-28] — Hardening findings from the whole-codebase review

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)):

- **P1 — NaN confidence panics a walk.** `partial_cmp().unwrap()` at `graph.rs:278`, `walk.rs:35`, `pagerank.rs:19` panics on NaN confidence loaded from packed/msgpack files. Route through the shared NaN-safe helper (workspace-wide cleanup).

The finding was recorded, not fixed. It remains open — see
[`ROADMAP.md`](ROADMAP.md) §"P0 — correctness and robustness".
