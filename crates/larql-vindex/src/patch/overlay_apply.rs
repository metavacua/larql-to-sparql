//! Patch application — `apply_patch`, `remove_patch`,
//! `rebuild_overrides` for `PatchedVindex`.
//!
//! Walks `VindexPatch::operations` and resolves each one into the
//! overlay's override maps (or the L0 KNN store for arch-B ops).
//! Pulled out of `overlay.rs` so the file holding `PatchedVindex`'s
//! query/mutation API stays focused.

use crate::index::FeatureMeta;

use super::format::{decode_gate_vector, PatchOp, VindexPatch};
use super::overlay::PatchedVindex;

impl PatchedVindex {
    /// Apply a patch. Operations are resolved into the override maps.
    pub fn apply_patch(&mut self, patch: VindexPatch) {
        for op in &patch.operations {
            match op {
                PatchOp::InsertKnn { layer, entity, relation, target, target_id, confidence, key_vector_b64 } => {
                    if let Ok(key_vec) = decode_gate_vector(key_vector_b64) {
                        self.knn_store.add(
                            *layer,
                            key_vec,
                            *target_id,
                            target.clone(),
                            entity.clone(),
                            relation.clone(),
                            confidence.unwrap_or(1.0),
                        );
                    }
                    continue;
                }
                PatchOp::DeleteKnn { entity } => {
                    self.knn_store.remove_by_entity(entity);
                    continue;
                }
                _ => {}
            }
            let key = op.key().unwrap(); // safe: only Arch A ops reach here
            match op {
                PatchOp::Insert { target, confidence, gate_vector_b64, down_meta, .. } => {
                    let meta = if let Some(dm) = down_meta {
                        FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        }
                    } else {
                        FeatureMeta {
                            top_token: target.clone(),
                            top_token_id: 0,
                            c_score: confidence.unwrap_or(0.9),
                            top_k: vec![],
                        }
                    };
                    self.overrides_meta.insert(key, Some(meta));
                    self.deleted.remove(&key);
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                }
                PatchOp::Update { gate_vector_b64, down_meta, .. } => {
                    if let Some(dm) = down_meta {
                        let meta = FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        };
                        self.overrides_meta.insert(key, Some(meta));
                    }
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                }
                PatchOp::Delete { .. } => {
                    self.overrides_meta.insert(key, None);
                    self.deleted.insert(key);
                    self.overrides_gate.remove(&key);
                }
                PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } => {
                    unreachable!("KNN ops handled above");
                }
            }
        }
        self.patches.push(patch);
    }

    /// Remove the last applied patch and rebuild overrides.
    pub fn remove_patch(&mut self, index: usize) {
        if index < self.patches.len() {
            self.patches.remove(index);
            self.rebuild_overrides();
        }
    }

    /// Rebuild override maps from scratch (after removing a patch).
    fn rebuild_overrides(&mut self) {
        self.overrides_meta.clear();
        self.overrides_gate.clear();
        self.deleted.clear();
        self.knn_store = super::knn_store::KnnStore::default();
        let patches: Vec<VindexPatch> = self.patches.drain(..).collect();
        for patch in patches {
            self.apply_patch(patch);
        }
    }
}
