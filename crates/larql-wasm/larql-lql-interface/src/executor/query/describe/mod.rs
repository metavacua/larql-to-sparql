//! `DESCRIBE <entity>` — walk-based edge scan, MoE-aware.
//!
//! Five-phase pipeline:
//!
//!   1. **Build query** — embed the entity, average tokens (`collect`).
//!   2. **Resolve scan layers** — pick from band/layer filter (`collect`).
//!   3. **Walk + collect edges** — coalesce hits per target token (`collect`).
//!   4. **Format & split** — apply relation labels, slot edges into
//!      syntax / knowledge / output buckets (`format`).
//!   5. **Render** — entry-point `exec_describe` orchestrates the
//!      buckets into output lines (`exec`).
//!
//! MoE-router-equipped vindexes get a parallel router-based path
//! (`moe`) that short-circuits before the dense walk.

mod collect;
mod exec;
mod format;
mod moe;
