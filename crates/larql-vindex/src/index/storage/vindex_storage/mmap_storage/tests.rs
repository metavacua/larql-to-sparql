//! Tests for [`super`].
//!
//! Split out of `mmap_storage.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use crate::index::types::GateLayerSlice;

/// Empty wrapper has every accessor returning `None`.
#[test]
fn empty_storage_returns_none_everywhere() {
    let s = MmapStorage::empty(2560);
    assert!(s.interleaved_kquant_layer_data(0).is_none());
    assert!(s.interleaved_kquant_whole_buffer().is_none());
    assert!(s.interleaved_q4_whole_buffer().is_none());
    assert!(s.down_features_q4k_layer_data(0).is_none());
    assert!(s.gate_q4_layer_data(0).is_none());
    assert!(s.attn_kquant_layer_data(0).is_none());
    assert!(s.attn_q4_whole_buffer().is_none());
    assert!(s.attn_q4_layer_slices(0).is_none());
    assert!(s.attn_q8_layer_data(0).is_none());
    assert!(s.lm_head_q4_bytes().is_none());
    assert!(s.lm_head_f16_bytes().is_none());
    assert!(s.lm_head_f32_bytes().is_none());
    assert!(s.gate_layer_view(0).is_none());
}

/// A `Bytes`-backed `MmapStorage` with a fabricated FFN Q4_K
/// manifest must hand back the same byte ranges the manifest
/// describes.
#[test]
fn ffn_q4k_layer_data_matches_manifest() {
    let mut s = MmapStorage::empty(8);
    // 3 layers × 3 components × 16 bytes = 144 bytes.
    let payload: Vec<u8> = (0u8..144).collect();
    s.interleaved_kquant = Some(Bytes::from(payload.clone()));
    s.interleaved_kquant_manifest = Some(
        (0..3 * FFN_COMPONENTS_PER_LAYER)
            .map(|i| (i * 16, 16, "Q4_K".to_string()))
            .collect(),
    );

    for layer in 0..3 {
        let arr = s
            .interleaved_kquant_layer_data(layer)
            .expect("layer present");
        for (c, (view, fmt)) in arr.iter().enumerate() {
            let global = layer * FFN_COMPONENTS_PER_LAYER + c;
            let expected: &[u8] = &payload[global * 16..(global + 1) * 16];
            assert_eq!(view.as_slice(), expected, "layer {layer} comp {c}");
            assert_eq!(*fmt, "Q4_K");
        }
    }
}

/// A stale FFN Q4_K manifest entry that runs past the buffer
/// must produce `None`, not a slice-bounds panic.
#[test]
fn ffn_q4k_layer_data_rejects_out_of_bounds_manifest() {
    let mut s = MmapStorage::empty(8);
    let payload: Vec<u8> = vec![0u8; 32];
    s.interleaved_kquant = Some(Bytes::from(payload));
    // gate fits, up fits, down points past the end.
    s.interleaved_kquant_manifest = Some(vec![
        (0, 8, "Q4_K".to_string()),
        (8, 8, "Q4_K".to_string()),
        (16, 32, "Q4_K".to_string()), // 16 + 32 = 48 > 32
    ]);
    assert!(s.interleaved_kquant_layer_data(0).is_none());
}

/// Attention Q8 layer data carries vals + scales spans; both must
/// fit before any tuple is formed.
#[test]
fn attn_q8_layer_data_validates_combined_span() {
    let mut s = MmapStorage::empty(8);
    s.attn_q8 = Some(Bytes::from(vec![0u8; 1024]));
    // Q, K, V fit; O's scales run past 1024.
    s.attn_q8_manifest = Some(vec![
        (0, 64, 16),
        (100, 64, 16),
        (200, 64, 16),
        (1000, 64, 16), // 1000 + 64 + 16 = 1080 > 1024
    ]);
    assert!(s.attn_q8_layer_data(0).is_none());
}

/// `GateLayerView<'_>` borrows the dtype + slice + bytes
/// together. The view is `Copy`, so multiple holders share the
/// same borrow without refcount touches.
#[test]
fn gate_layer_view_round_trip() {
    let mut s = MmapStorage::empty(4);
    s.gate_bytes = Some(Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8]));
    s.gate_dtype = StorageDtype::F16;
    s.gate_slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 1,
        },
        GateLayerSlice {
            float_offset: 4,
            num_features: 1,
        },
    ];
    let v0 = s.gate_layer_view(0).expect("layer 0 present");
    assert_eq!(v0.dtype, StorageDtype::F16);
    assert_eq!(v0.slice.num_features, 1);
    let v0_copy = v0; // `Copy`, no clone needed.
    assert_eq!(v0.bytes.as_ptr(), v0_copy.bytes.as_ptr());
}

/// `gate_layer_view` returns `None` when the layer's
/// `num_features` is zero — matches the substore convention for
/// unowned layers in a sharded `--layers` slice.
#[test]
fn gate_layer_view_none_when_layer_unowned() {
    let mut s = MmapStorage::empty(4);
    s.gate_bytes = Some(Bytes::from(vec![0u8; 8]));
    s.gate_slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 0,
    }];
    assert!(s.gate_layer_view(0).is_none());
}

/// `MmapStorage` clones cheaply — every field is `Bytes` /
/// `Vec<...>` / `Copy`, so clone is a refcount bump per
/// whole-file `Bytes`.
#[test]
fn mmap_storage_clones_via_refcount() {
    let mut s = MmapStorage::empty(4);
    s.lm_head_f16 = Some(Bytes::from(vec![1u8, 2, 3, 4]));
    let cloned = s.clone();
    assert_eq!(
        s.lm_head_f16.as_ref().unwrap().as_ptr(),
        cloned.lm_head_f16.as_ref().unwrap().as_ptr(),
    );
}

// ── Setter coverage ──────────────────────────────────────────────
//
// All `set_*` methods take an `Arc<Mmap>` (or `Arc<Vec<u8>>`).
// Building real `Arc<Mmap>` instances from anonymous mmap is the
// closest synthetic analogue of what loaders do; helper below
// produces one with a known byte payload.

fn arc_mmap_from(payload: &[u8]) -> Arc<memmap2::Mmap> {
    let mut anon = memmap2::MmapMut::map_anon(payload.len()).expect("anon mmap");
    anon.copy_from_slice(payload);
    let mmap = anon.make_read_only().expect("freeze");
    Arc::new(mmap)
}

#[test]
fn set_interleaved_q4k_with_manifest_then_layer_data() {
    let payload: Vec<u8> = (0u8..96).collect();
    let mut s = MmapStorage::empty(8);
    s.set_interleaved_kquant(
        arc_mmap_from(&payload),
        Some(vec![
            (0, 16, "Q4_K".to_string()),
            (16, 16, "Q4_K".to_string()),
            (32, 16, "Q4_K".to_string()),
        ]),
    );
    assert!(s.has_interleaved_kquant());
    assert!(s.interleaved_kquant_whole_buffer().is_some());
    assert!(s.interleaved_kquant_whole_buffer_view().is_some());
    let arr = s.interleaved_kquant_layer_data(0).expect("layer 0");
    assert_eq!(arr[0].0.len(), 16);
    assert_eq!(arr[0].1, "Q4_K");
    // Layer 1 is past the 3 manifest entries → None.
    assert!(s.interleaved_kquant_layer_data(1).is_none());
}

#[test]
fn set_interleaved_q4_whole_buffer_round_trip() {
    let payload = vec![7u8; 32];
    let mut s = MmapStorage::empty(8);
    s.set_interleaved_q4(arc_mmap_from(&payload));
    assert!(s.has_interleaved_q4());
    let buf = s.interleaved_q4_whole_buffer().expect("whole buffer");
    assert_eq!(buf.as_ref(), payload.as_slice());
    let view = s.interleaved_q4_whole_buffer_view().expect("view");
    assert_eq!(view.as_ref(), payload.as_slice());
}

#[test]
fn set_down_features_q4k_then_layer_data() {
    let payload = vec![0u8; 64];
    let mut s = MmapStorage::empty(8);
    s.set_down_features_q4k(
        arc_mmap_from(&payload),
        vec![DownFeaturesQ4kEntry {
            offset: 0,
            length: 32,
            format: "Q4_K".to_string(),
            padded_width: 8,
        }],
    );
    assert!(s.has_down_features_kquant());
    let (view, fmt, padded) = s.down_features_q4k_layer_data(0).expect("layer 0");
    assert_eq!(view.len(), 32);
    assert_eq!(fmt, "Q4_K");
    assert_eq!(padded, 8);
}

#[test]
fn set_attn_q4k_q4_q8_round_trips() {
    let payload = vec![0u8; 256];
    let mut s = MmapStorage::empty(8);

    s.set_attn_kquant(
        arc_mmap_from(&payload),
        Some(vec![
            (0, 16, "Q4_K".to_string()),
            (16, 16, "Q4_K".to_string()),
            (32, 16, "Q4_K".to_string()),
            (48, 16, "Q4_K".to_string()),
        ]),
    );
    assert!(s.has_attn_kquant());
    let q4k_arr = s.attn_kquant_layer_data(0).expect("attn q4k");
    assert_eq!(q4k_arr[0].0.len(), 16);

    s.set_attn_q4(
        arc_mmap_from(&payload),
        Some(vec![(0, 16), (16, 16), (32, 16), (48, 16)]),
    );
    assert!(s.has_attn_q4());
    let q4_arr = s.attn_q4_layer_slices(0).expect("attn q4");
    assert_eq!(q4_arr[0].len(), 16);
    assert!(s.attn_q4_whole_buffer().is_some());
    assert!(s.attn_q4_whole_buffer_view().is_some());

    s.set_attn_q8(
        arc_mmap_from(&payload),
        Some(vec![(0, 16, 4), (32, 16, 4), (64, 16, 4), (96, 16, 4)]),
    );
    assert!(s.has_attn_q8());
    let q8_arr = s.attn_q8_layer_data(0).expect("attn q8");
    assert_eq!(q8_arr[0].0.len(), 16);
    assert_eq!(q8_arr[0].1.len(), 4);
}

#[test]
fn set_lm_head_variants_and_views() {
    let payload = vec![0u8; 32];
    let mut s = MmapStorage::empty(8);

    s.set_lm_head_f32(arc_mmap_from(&payload));
    assert!(s.has_lm_head_f32());
    assert!(s.lm_head_f32_bytes().is_some());
    assert!(s.lm_head_f32_view().is_some());

    s.set_lm_head_f16(arc_mmap_from(&payload));
    assert!(s.has_lm_head_f16());
    assert!(s.lm_head_f16_bytes().is_some());
    assert!(s.lm_head_f16_view().is_some());

    s.set_lm_head_kquant_mmap(arc_mmap_from(&payload));
    assert!(s.has_lm_head_kquant());
    assert!(s.lm_head_q4_bytes().is_some());
    assert!(s.lm_head_kquant_view().is_some());
}

#[test]
fn set_lm_head_q4_synth_round_trip() {
    let bytes = Arc::new(vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
    let mut s = MmapStorage::empty(4);
    s.set_lm_head_kquant_synth(bytes.clone());
    assert!(s.has_lm_head_kquant());
    let view = s.lm_head_kquant_view().expect("synth view");
    assert_eq!(view.as_ref(), bytes.as_slice());
}

#[test]
fn set_gate_vectors_then_layer_view() {
    let payload = vec![0u8; 64];
    let mut s = MmapStorage::empty(4);
    s.set_gate_vectors(
        arc_mmap_from(&payload),
        StorageDtype::F16,
        vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 2,
            },
            GateLayerSlice {
                float_offset: 8,
                num_features: 2,
            },
        ],
    );
    assert!(s.has_gate_vectors());
    let view = s.gate_layer_view(0).expect("layer 0");
    assert_eq!(view.dtype, StorageDtype::F16);
    assert_eq!(view.slice.num_features, 2);
}

#[test]
fn set_gate_q4_then_layer_data() {
    let payload = vec![0u8; 64];
    let mut s = MmapStorage::empty(4);
    s.set_gate_q4(
        arc_mmap_from(&payload),
        vec![GateQ4Slice {
            byte_offset: 0,
            byte_len: 32,
            num_features: 4,
        }],
    );
    assert!(s.has_gate_q4());
    let view = s.gate_q4_layer_data(0).expect("layer 0");
    assert_eq!(view.len(), 32);
}

/// Sweep test — exercise every trait method + has_* helper on a
/// fully-populated `MmapStorage` so the trait `impl` block lights
/// up under coverage.
#[test]
fn full_sweep_through_trait_and_helpers() {
    use crate::index::storage::vindex_storage::VindexStorage;
    let payload: Vec<u8> = (0u8..=255).collect();
    let mut s = MmapStorage::empty(8);
    s.set_interleaved_kquant(
        arc_mmap_from(&payload),
        Some(vec![
            (0, 16, "Q4_K".into()),
            (16, 16, "Q4_K".into()),
            (32, 16, "Q4_K".into()),
        ]),
    );
    s.set_interleaved_q4(arc_mmap_from(&payload));
    s.set_down_features_q4k(
        arc_mmap_from(&payload),
        vec![DownFeaturesQ4kEntry {
            offset: 0,
            length: 32,
            format: "Q4_K".into(),
            padded_width: 8,
        }],
    );
    s.set_gate_q4(
        arc_mmap_from(&payload),
        vec![GateQ4Slice {
            byte_offset: 0,
            byte_len: 32,
            num_features: 4,
        }],
    );
    s.set_attn_kquant(
        arc_mmap_from(&payload),
        Some(vec![
            (0, 16, "Q4_K".into()),
            (16, 16, "Q4_K".into()),
            (32, 16, "Q4_K".into()),
            (48, 16, "Q4_K".into()),
        ]),
    );
    s.set_attn_q4(
        arc_mmap_from(&payload),
        Some(vec![(0, 16), (16, 16), (32, 16), (48, 16)]),
    );
    s.set_attn_q8(
        arc_mmap_from(&payload),
        Some(vec![(0, 16, 4), (32, 16, 4), (64, 16, 4), (96, 16, 4)]),
    );
    s.set_lm_head_f32(arc_mmap_from(&payload));
    s.set_lm_head_f16(arc_mmap_from(&payload));
    s.set_lm_head_kquant_mmap(arc_mmap_from(&payload));
    s.set_gate_vectors(
        arc_mmap_from(&payload),
        StorageDtype::F16,
        vec![GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        }],
    );

    // Trait surface — owned-Bytes whole-buffer methods.
    assert!(s.interleaved_kquant_whole_buffer().is_some());
    assert!(s.interleaved_q4_whole_buffer().is_some());
    assert!(s.attn_q4_whole_buffer().is_some());
    assert!(s.lm_head_q4_bytes().is_some());
    assert!(s.lm_head_f16_bytes().is_some());
    assert!(s.lm_head_f32_bytes().is_some());

    // has_* helpers — both the populated and unpopulated
    // branches via a fresh empty.
    assert!(s.has_interleaved_kquant());
    assert!(s.has_interleaved_q4());
    assert!(s.has_down_features_kquant());
    assert!(s.has_gate_q4());
    assert!(s.has_gate_vectors());
    assert!(s.has_attn_kquant());
    assert!(s.has_attn_q4());
    assert!(s.has_attn_q8());
    assert!(s.has_lm_head_kquant());
    assert!(s.has_lm_head_f16());
    assert!(s.has_lm_head_f32());

    let empty = MmapStorage::empty(8);
    assert!(!empty.has_interleaved_kquant());
    assert!(!empty.has_interleaved_q4());
    assert!(!empty.has_down_features_kquant());
    assert!(!empty.has_gate_q4());
    assert!(!empty.has_gate_vectors());
    assert!(!empty.has_attn_kquant());
    assert!(!empty.has_attn_q4());
    assert!(!empty.has_attn_q8());
    assert!(!empty.has_lm_head_kquant());
    assert!(!empty.has_lm_head_f16());
    assert!(!empty.has_lm_head_f32());

    // Trait dispatch via `Arc<dyn VindexStorage>`.
    let dyn_storage: Arc<dyn VindexStorage> = Arc::new(s);
    assert!(dyn_storage.gate_q4_layer_data(0).is_some());
    assert!(dyn_storage.attn_kquant_layer_data(0).is_some());
    assert!(dyn_storage.attn_q4_layer_slices(0).is_some());
    assert!(dyn_storage.attn_q8_layer_data(0).is_some());
    assert!(dyn_storage.gate_layer_view(0).is_some());
    assert!(dyn_storage.down_features_q4k_layer_data(0).is_some());
}

/// `attn_q4_layer_slices` rejects an out-of-bounds manifest slice
/// for the same reason `attn_kquant_layer_data` does — exercising the
/// per-tensor checked_view branch.
#[test]
fn attn_q4_layer_slices_rejects_out_of_bounds() {
    let payload = vec![0u8; 64];
    let mut s = MmapStorage::empty(8);
    s.set_attn_q4(
        arc_mmap_from(&payload),
        Some(vec![(0, 16), (16, 16), (32, 16), (60, 32)]), // last spans past 64
    );
    assert!(s.attn_q4_layer_slices(0).is_none());
}

/// `down_features_q4k_layer_data` rejects an out-of-bounds entry.
#[test]
fn down_features_q4k_layer_data_rejects_out_of_bounds() {
    let payload = vec![0u8; 32];
    let mut s = MmapStorage::empty(8);
    s.set_down_features_q4k(
        arc_mmap_from(&payload),
        vec![DownFeaturesQ4kEntry {
            offset: 16,
            length: 32, // 16+32 = 48 > 32
            format: "Q4_K".into(),
            padded_width: 8,
        }],
    );
    assert!(s.down_features_q4k_layer_data(0).is_none());
}

/// `gate_q4_layer_data` rejects a slice whose `byte_len` is 0
/// (typical for an unowned layer) or whose range overflows.
#[test]
fn gate_q4_layer_data_rejects_zero_or_overflow() {
    let payload = vec![0u8; 32];
    let mut s = MmapStorage::empty(8);
    // Zero-length slice.
    s.set_gate_q4(
        arc_mmap_from(&payload),
        vec![GateQ4Slice {
            byte_offset: 0,
            byte_len: 0,
            num_features: 0,
        }],
    );
    assert!(s.gate_q4_layer_data(0).is_none());

    // Overflow.
    s.set_gate_q4(
        arc_mmap_from(&payload),
        vec![GateQ4Slice {
            byte_offset: 16,
            byte_len: 32, // 16+32 = 48 > 32
            num_features: 4,
        }],
    );
    assert!(s.gate_q4_layer_data(0).is_none());
}

/// `set_interleaved_kquant` is zero-copy from `Arc<Mmap>` — the
/// `Bytes` view points at the same memory the original mmap
/// occupies, no copy.
#[test]
fn set_interleaved_q4k_is_zero_copy_from_arc_mmap() {
    let payload = vec![0u8; 32];
    let mmap_arc = arc_mmap_from(&payload);
    let mmap_ref_ptr = mmap_arc.as_ref().as_ptr();

    let mut s = MmapStorage::empty(8);
    s.set_interleaved_kquant(mmap_arc, None);
    let view = s.interleaved_kquant_whole_buffer_view().expect("buf");
    assert_eq!(view.as_ref().as_ptr(), mmap_ref_ptr);
}

// ── Step 6 helpers: gate layer-slice + dtype + bytes_view +
//                    release_pages ────────────────────────────

#[test]
fn gate_helpers_round_trip() {
    let payload = vec![0u8; 32];
    let mut s = MmapStorage::empty(4);
    let slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        },
        GateLayerSlice {
            float_offset: 8,
            num_features: 2,
        },
    ];
    s.set_gate_vectors(arc_mmap_from(&payload), StorageDtype::F16, slices.clone());

    // Concrete helpers added in step 6.
    assert_eq!(s.gate_dtype(), StorageDtype::F16);
    assert_eq!(s.gate_layer_slices().len(), 2);
    assert_eq!(s.gate_layer_slice(1).map(|s| s.float_offset), Some(8));
    assert!(s.gate_layer_slice(99).is_none());
    assert!(s.gate_bytes_view().is_some());
}

#[test]
fn gate_q4_helpers_round_trip() {
    let payload = vec![0u8; 32];
    let mut s = MmapStorage::empty(4);
    s.set_gate_q4(
        arc_mmap_from(&payload),
        vec![GateQ4Slice {
            byte_offset: 0,
            byte_len: 16,
            num_features: 4,
        }],
    );
    assert_eq!(s.gate_q4_layer_slices().len(), 1);
    assert_eq!(s.gate_q4_layer_slice(0).map(|s| s.byte_len), Some(16));
    assert!(s.gate_q4_layer_slice(99).is_none());
    assert!(s.gate_q4_bytes_view().is_some());
}

/// `release_pages` calls madvise on every tracked file-backed
/// mmap. Best-effort and platform-dependent — this test pins
/// only "doesn't panic, doesn't lose state".
#[test]
fn release_pages_does_not_destroy_storage() {
    let payload = vec![0u8; 64];
    let mut s = MmapStorage::empty(4);
    s.set_interleaved_kquant(arc_mmap_from(&payload), None);
    s.set_attn_kquant(arc_mmap_from(&payload), None);
    s.set_lm_head_f32(arc_mmap_from(&payload));
    s.set_gate_vectors(arc_mmap_from(&payload), StorageDtype::F16, vec![]);
    s.set_gate_q4(arc_mmap_from(&payload), vec![]);
    s.set_lm_head_kquant_synth(Arc::new(vec![1u8, 2, 3, 4]));

    // Five mmap-backed setters → 5 handles. The synth setter does
    // NOT register a handle.
    assert_eq!(s.mmap_handles.len(), 5);

    // No panic; data still readable after.
    s.release_pages();
    assert!(s.has_interleaved_kquant());
    assert!(s.has_attn_kquant());
    assert!(s.has_lm_head_f32());
    assert!(s.has_gate_vectors());
    assert!(s.has_gate_q4());
    assert!(s.has_lm_head_kquant());
}

/// `release_pages` on an empty storage is a no-op.
#[test]
fn release_pages_empty_is_noop() {
    let s = MmapStorage::empty(4);
    s.release_pages();
    assert_eq!(s.mmap_handles.len(), 0);
}

/// `register_mmap` is private but exercised through every
/// `set_*` mmap-taking setter. Verify the count matches the
/// number of mmap-taking setter calls (synth lm_head excluded).
#[test]
fn register_mmap_only_tracks_file_backed_handles() {
    let payload = vec![0u8; 16];
    let mut s = MmapStorage::empty(4);
    assert_eq!(s.mmap_handles.len(), 0);

    s.set_lm_head_kquant_synth(Arc::new(vec![1u8, 2, 3]));
    assert_eq!(s.mmap_handles.len(), 0, "synth (heap) should not register");

    s.set_lm_head_kquant_mmap(arc_mmap_from(&payload));
    assert_eq!(s.mmap_handles.len(), 1);

    s.set_attn_q8(arc_mmap_from(&payload), None);
    assert_eq!(s.mmap_handles.len(), 2);
}

/// `BytesView::is_empty` covers the zero-length path.
#[test]
fn bytes_view_is_empty() {
    let bytes = Bytes::from(vec![0u8; 16]);
    let zero = BytesView::new(&bytes, 0, 0);
    assert!(zero.is_empty());
    assert_eq!(zero.len(), 0);
    assert_eq!(zero.as_slice().len(), 0);
}
