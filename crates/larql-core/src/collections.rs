//! Portable `HashMap`/`HashSet` aliases.
//!
//! `std::collections::HashMap`/`HashSet` have no `core`/`alloc` equivalent
//! -- both need a `BuildHasher`, and `alloc` provides no default one
//! (std's `RandomState` seeds itself from OS randomness, which doesn't
//! exist on `wasm32v1-none`). `hashbrown` is genuinely `no_std`-capable,
//! but its own default hasher (`foldhash`) has the identical OS-randomness
//! requirement -- confirmed via its Cargo.toml, not assumed. The fix used
//! here (a fixed-seed FNV-1a `Hasher`, `default-features = false` on
//! `hashbrown`) matches this exact codebase's prior wasm32 gating attempt
//! (PR #71, "cfg-split DefaultHasher across crates: std native / FNV
//! wasm32"). See docs/superpowers/plans/2026-08-10-larql-cli-wasm-and-safe-gating.md,
//! Task 7, pattern 4: std-collection-needs-no-std-hasher.
//!
//! FNV-1a is not DoS-resistant (unlike std's SipHash-based RandomState) --
//! fine here because this crate's maps are keyed by data the crate itself
//! produces (graph node/edge ids), not attacker-controlled input.

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

/// FNV-1a, the standard 64-bit variant. No external state, no OS
/// randomness -- fully deterministic, which is exactly what `no_std`
/// needs and exactly why it must not be used on attacker-controlled keys.
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
