//! Portable `HashMap`/`HashSet` aliases. See
//! `crates/larql-core/src/collections.rs` for the full rationale
//! (pattern 4: std-collection-needs-no-std-hasher) -- same fix,
//! duplicated per-crate since Rust modules don't cross crate boundaries.
//!
//! `pub` (not `pub(crate)`) from the start -- pattern 14
//! (private-hasher-leaks-into-public-field-type): several structs in
//! this crate may have `pub` fields typed with these aliases, and a `pub`
//! field can't have a type built from a private component without a
//! downstream crate eventually hitting "type is private" once its code
//! needs to resolve a trait bound through it (e.g. `.get()`).

// Unused on native for this crate specifically: every HashMap/HashSet
// site in larql-lql lives inside relations.rs/executor/ (both already
// wholesale-gated `#[cfg(not(target_arch = "wasm32"))]` at their `mod`
// declaration), which reach for `std::collections::` directly rather
// than this alias -- kept here anyway for parity with every other
// gated crate and for any future portable code that needs it.
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
pub use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// BTreeMap/BTreeSet need no hasher (Ord-keyed) -- alloc's own types,
// identical to std's (std::collections::BTreeMap literally re-exports
// alloc's), just needs `extern crate alloc;` in scope, which the
// wasm32 crate root already declares.

#[cfg(target_arch = "wasm32")]
pub type HashMap<K, V> = hashbrown::HashMap<K, V, ::core::hash::BuildHasherDefault<FnvHasher>>;
#[cfg(target_arch = "wasm32")]
pub type HashSet<K> = hashbrown::HashSet<K, ::core::hash::BuildHasherDefault<FnvHasher>>;

#[cfg(target_arch = "wasm32")]
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
#[cfg(target_arch = "wasm32")]
const FNV_PRIME: u64 = 0x100000001b3;

#[cfg(target_arch = "wasm32")]
pub struct FnvHasher(u64);

#[cfg(target_arch = "wasm32")]
impl Default for FnvHasher {
    fn default() -> Self {
        FnvHasher(FNV_OFFSET_BASIS)
    }
}

#[cfg(target_arch = "wasm32")]
impl ::core::hash::Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.0;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        self.0 = hash;
    }
}
