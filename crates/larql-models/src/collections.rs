//! Portable `HashMap`/`HashSet` aliases. See
//! `crates/larql-core/src/collections.rs` for the full rationale
//! (pattern 4: std-collection-needs-no-std-hasher) -- same fix,
//! duplicated per-crate since Rust modules don't cross crate boundaries.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::collections::{HashMap, HashSet};

#[cfg(target_arch = "wasm32")]
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, ::core::hash::BuildHasherDefault<FnvHasher>>;
#[cfg(target_arch = "wasm32")]
pub(crate) type HashSet<K> = hashbrown::HashSet<K, ::core::hash::BuildHasherDefault<FnvHasher>>;

#[cfg(target_arch = "wasm32")]
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
#[cfg(target_arch = "wasm32")]
const FNV_PRIME: u64 = 0x100000001b3;

#[cfg(target_arch = "wasm32")]
pub(crate) struct FnvHasher(u64);

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
