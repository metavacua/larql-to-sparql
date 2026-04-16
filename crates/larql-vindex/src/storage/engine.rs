use crate::patch::core::PatchedVindex;
use super::epoch::Epoch;
use super::memit_store::MemitStore;
use super::status::CompactStatus;

const MEMIT_MIN_HIDDEN_DIM: usize = 1024;

pub struct StorageEngine {
    patched: PatchedVindex,
    epoch: Epoch,
    mutations_since_minor: usize,
    mutations_since_major: usize,
    memit_store: MemitStore,
}

impl StorageEngine {
    pub fn new(patched: PatchedVindex) -> Self {
        Self {
            patched,
            epoch: Epoch::zero(),
            mutations_since_minor: 0,
            mutations_since_major: 0,
            memit_store: MemitStore::new(),
        }
    }

    pub fn patched(&self) -> &PatchedVindex {
        &self.patched
    }

    pub fn patched_mut(&mut self) -> &mut PatchedVindex {
        &mut self.patched
    }

    pub fn into_patched(self) -> PatchedVindex {
        self.patched
    }

    pub fn epoch(&self) -> u64 {
        self.epoch.value()
    }

    pub fn advance_epoch(&mut self) {
        self.epoch.advance();
        self.mutations_since_minor += 1;
        self.mutations_since_major += 1;
    }

    pub fn memit_store(&self) -> &MemitStore {
        &self.memit_store
    }

    pub fn memit_store_mut(&mut self) -> &mut MemitStore {
        &mut self.memit_store
    }

    pub fn supports_memit(&self) -> bool {
        self.patched.hidden_size() >= MEMIT_MIN_HIDDEN_DIM
    }

    pub fn compact_status(&self) -> CompactStatus {
        let l0_entries = self.patched.knn_store.len();
        let l1_edges = self.patched.num_overrides();
        let l1_layers: std::collections::HashSet<usize> = self
            .patched
            .overrides_gate_iter()
            .map(|(layer, _, _)| layer)
            .collect();

        CompactStatus {
            epoch: self.epoch.value(),
            l0_entries,
            l0_tombstones: 0, // tombstone tracking added in Phase 7
            l1_edges,
            l1_layers_used: l1_layers.len(),
            l2_facts: self.memit_store.total_facts(),
            l2_cycles: self.memit_store.num_cycles(),
            base_layers: self.patched.num_layers(),
            base_features_per_layer: if self.patched.num_layers() > 0 {
                self.patched.num_features(0)
            } else {
                0
            },
            hidden_dim: self.patched.hidden_size(),
            memit_supported: self.supports_memit(),
        }
    }

    pub fn mutations_since_minor(&self) -> usize {
        self.mutations_since_minor
    }

    pub fn mutations_since_major(&self) -> usize {
        self.mutations_since_major
    }

    pub fn reset_minor_counter(&mut self) {
        self.mutations_since_minor = 0;
    }

    pub fn reset_major_counter(&mut self) {
        self.mutations_since_major = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::VectorIndex;

    fn empty_engine() -> StorageEngine {
        let vi = VectorIndex::new(vec![], vec![], 0, 0);
        let pv = PatchedVindex::new(vi);
        StorageEngine::new(pv)
    }

    #[test]
    fn new_engine_epoch_zero() {
        let e = empty_engine();
        assert_eq!(e.epoch(), 0);
    }

    #[test]
    fn advance_epoch_increments() {
        let mut e = empty_engine();
        e.advance_epoch();
        assert_eq!(e.epoch(), 1);
        e.advance_epoch();
        assert_eq!(e.epoch(), 2);
    }

    #[test]
    fn compact_status_empty() {
        let e = empty_engine();
        let s = e.compact_status();
        assert_eq!(s.l0_entries, 0);
        assert_eq!(s.l1_edges, 0);
        assert_eq!(s.l2_facts, 0);
        assert_eq!(s.epoch, 0);
    }

    #[test]
    fn mutations_tracked() {
        let mut e = empty_engine();
        assert_eq!(e.mutations_since_minor(), 0);
        e.advance_epoch();
        e.advance_epoch();
        assert_eq!(e.mutations_since_minor(), 2);
        e.reset_minor_counter();
        assert_eq!(e.mutations_since_minor(), 0);
        assert_eq!(e.mutations_since_major(), 2);
    }

    #[test]
    fn memit_guard_small_model() {
        let e = empty_engine();
        assert!(!e.supports_memit());
    }
}
