//! Mutation executor: INSERT, DELETE, UPDATE, MERGE, REBALANCE.
// SPDX-License-Identifier: Apache-2.0

//!
//! All mutations go through the `PatchedVindex` overlay — base vindex
//! files on disk are never modified.

mod delete;
mod insert;
mod merge;
mod rebalance;
mod update;
