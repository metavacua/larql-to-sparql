//! Rewinding a decode step that did not complete.
//!
//! `kv_decode_step_run` appends each layer's K/V before the FFN gets the
//! chance to refuse, so a step that stops partway leaves the cache holding a
//! token that produced no output. These three functions are the same
//! rewind-or-invalidate protocol `StandardEngine` uses, applied to the plain
//! [`KvCache`]: snapshot the lengths, decide whether truncation would be
//! honest, and either restore or say it could not.

use crate::cache::KvCache;

/// Logical row count per layer, `None` for layers that share another's K/V.
pub(super) fn cache_row_counts(cache: &KvCache) -> Vec<Option<usize>> {
    cache
        .layers
        .iter()
        .map(|slot| slot.as_ref().map(|(k, _)| k.shape()[0]))
        .collect()
}

/// Whether truncating back to `entry_rows` would restore the exact cache the
/// step started from.
///
/// Unbounded caches only ever append, so truncation is exact. A windowed cache
/// that reaches its limit drops its oldest row to make room, and that row is
/// gone; row count cannot see it, so the only sound test is whether every
/// layer had room to spare before the step began.
pub(super) fn decode_rewind_is_sound(cache: &KvCache, entry_rows: &[Option<usize>]) -> bool {
    match cache.max_window {
        None => true,
        Some(w) => entry_rows.iter().flatten().all(|&rows| rows < w),
    }
}

/// Truncate every layer back to its recorded length.
pub(super) fn rewind_cache(cache: &mut KvCache, entry_rows: &[Option<usize>]) {
    for (slot, rows) in cache.layers.iter_mut().zip(entry_rows) {
        let (Some((k, v)), Some(rows)) = (slot.as_mut(), rows) else {
            continue;
        };
        if k.shape()[0] > *rows {
            *k = k.slice(ndarray::s![..*rows, ..]).to_owned();
            *v = v.slice(ndarray::s![..*rows, ..]).to_owned();
        }
    }
}
