//! Vindex v1 conformance — `interleaved_kquant.bin` + manifest and the
//! `down_features_kquant.bin` sidecar.
//!
//! Contract: a corrupt manifest fails the load with `VindexError`; a
//! manifest whose byte ranges don't fit the slab makes the per-layer
//! accessors *decline* (`None`, callers fall back to another FFN
//! backend) — never a panic, never an out-of-bounds slice. The
//! H1 padded-stride rule (rows padded to the 256-element super-block)
//! is pinned at the writer level. Companion doc:
//! `docs/conformance-v1.md` §interleaved_kquant.

mod common;

use common::{load_dir, save_minimal_vindex};
use larql_vindex::format::filenames::{
    DOWN_FEATURES_KQUANT_BIN, DOWN_FEATURES_KQUANT_MANIFEST_JSON, INTERLEAVED_KQUANT_BIN,
    INTERLEAVED_KQUANT_MANIFEST_JSON,
};
use larql_vindex::format::weights::write_layers::pad_cols_to_256;
use larql_vindex::VectorIndex;

/// Q4_K super-block: 144 bytes encode 256 elements.
const Q4K_BLOCK_BYTES: u64 = 144;
const K_QUANT_BLOCK_ELEMS: usize = 256;
/// gate, up, down.
const FFN_COMPONENTS: usize = 3;
const LAYER_0: usize = 0;

fn manifest_entry(key: &str, offset: u64, length: u64, format: &str) -> String {
    format!(
        r#"{{"key":"{key}","shape":[1,{K_QUANT_BLOCK_ELEMS}],"format":"{format}","offset":{offset},"length":{length}}}"#
    )
}

/// A fixture dir + a slab of `blocks` Q4_K blocks + the given manifest
/// entries, loaded back through the production path.
fn vindex_with_kquant(blocks: u64, entries: &[String]) -> (tempfile::TempDir, VectorIndex) {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let slab = vec![0u8; (blocks * Q4K_BLOCK_BYTES) as usize];
    std::fs::write(tmp.path().join(INTERLEAVED_KQUANT_BIN), &slab).unwrap();
    std::fs::write(
        tmp.path().join(INTERLEAVED_KQUANT_MANIFEST_JSON),
        format!("[{}]", entries.join(",")),
    )
    .unwrap();
    let index = load_dir(tmp.path()).unwrap();
    (tmp, index)
}

fn three_entries(length: u64) -> Vec<String> {
    (0..FFN_COMPONENTS)
        .map(|i| manifest_entry(&format!("t{i}"), i as u64 * length, length, "Q4_K"))
        .collect()
}

/// Positive control: an in-bounds 3-entry manifest yields all three
/// component slices with their format tags.
#[test]
fn valid_manifest_yields_layer_slices() {
    let (_tmp, index) = vindex_with_kquant(FFN_COMPONENTS as u64, &three_entries(Q4K_BLOCK_BYTES));
    let arr = index
        .interleaved_kquant_layer_data(LAYER_0)
        .expect("in-bounds manifest resolves");
    for (slice, fmt) in arr {
        assert_eq!(slice.len(), Q4K_BLOCK_BYTES as usize);
        assert_eq!(fmt, "Q4_K");
    }
}

/// An unknown format tag fails the load loudly — `quant::registry`
/// has no entry, so nothing downstream can decode the bytes.
#[test]
fn unknown_format_tag_fails_load() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::write(tmp.path().join(INTERLEAVED_KQUANT_BIN), [0u8; 16]).unwrap();
    std::fs::write(
        tmp.path().join(INTERLEAVED_KQUANT_MANIFEST_JSON),
        format!("[{}]", manifest_entry("t", 0, 16, "Q9_X")),
    )
    .unwrap();
    let mut index = load_dir(tmp.path()).unwrap(); // best-effort loader declined
    let err = index
        .load_interleaved_kquant(tmp.path())
        .expect_err("unknown tag errors");
    assert!(err.to_string().contains("unknown format tag"), "{err}");
}

/// A manifest whose declared ranges overrun the slab (truncated .bin)
/// is a safe decline: load succeeds, per-layer access returns None.
#[test]
fn truncated_slab_declines_layer_data() {
    // Slab holds one block; manifest claims three.
    let (_tmp, index) = vindex_with_kquant(1, &three_entries(Q4K_BLOCK_BYTES));
    assert!(index.has_interleaved_kquant());
    assert!(index.interleaved_kquant_layer_data(LAYER_0).is_none());
}

/// offset + length overflowing u64/usize must not wrap into a "valid"
/// range.
#[test]
fn offset_overflow_declines_layer_data() {
    let entries = vec![
        manifest_entry("gate", u64::MAX, Q4K_BLOCK_BYTES, "Q4_K"),
        manifest_entry("up", 0, Q4K_BLOCK_BYTES, "Q4_K"),
        manifest_entry("down", 0, Q4K_BLOCK_BYTES, "Q4_K"),
    ];
    let (_tmp, index) = vindex_with_kquant(FFN_COMPONENTS as u64, &entries);
    assert!(index.interleaved_kquant_layer_data(LAYER_0).is_none());
}

/// A manifest with fewer than 3 entries per layer cannot describe a
/// [gate, up, down] triple — decline, don't misindex.
#[test]
fn short_manifest_declines_layer_data() {
    let entries = vec![manifest_entry("gate", 0, Q4K_BLOCK_BYTES, "Q4_K")];
    let (_tmp, index) = vindex_with_kquant(FFN_COMPONENTS as u64, &entries);
    assert!(index.interleaved_kquant_layer_data(LAYER_0).is_none());
}

/// No manifest at all is the documented legacy fallback: the slab
/// loads, per-manifest accessors decline.
#[test]
fn missing_manifest_is_legacy_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::write(tmp.path().join(INTERLEAVED_KQUANT_BIN), [0u8; 16]).unwrap();
    let index = load_dir(tmp.path()).unwrap();
    assert!(index.has_interleaved_kquant());
    assert!(index.interleaved_kquant_layer_data(LAYER_0).is_none());
}

/// The sidecar's .bin without its manifest is an error — there is no
/// legacy stride fallback for feature-major down.
#[test]
fn down_features_bin_without_manifest_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::write(tmp.path().join(DOWN_FEATURES_KQUANT_BIN), [0u8; 16]).unwrap();
    let mut index = load_dir(tmp.path()).unwrap();
    let err = index
        .load_down_features_q4k(tmp.path())
        .expect_err("missing manifest errors");
    assert!(err.to_string().contains("manifest"), "{err}");
}

/// A sidecar manifest entry without `shape[1]` (padded width) errors.
#[test]
fn down_features_manifest_without_padded_width_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::write(tmp.path().join(DOWN_FEATURES_KQUANT_BIN), [0u8; 16]).unwrap();
    std::fs::write(
        tmp.path().join(DOWN_FEATURES_KQUANT_MANIFEST_JSON),
        r#"[{"key":"d","shape":[256],"format":"Q4_K","offset":0,"length":16}]"#,
    )
    .unwrap();
    let mut index = load_dir(tmp.path()).unwrap();
    let err = index
        .load_down_features_q4k(tmp.path())
        .expect_err("1-D shape errors");
    assert!(err.to_string().contains("padded_width"), "{err}");
}

/// Sidecar entries that overrun the slab decline per-layer access.
#[test]
fn down_features_out_of_bounds_declines() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::write(tmp.path().join(DOWN_FEATURES_KQUANT_BIN), [0u8; 16]).unwrap();
    std::fs::write(
        tmp.path().join(DOWN_FEATURES_KQUANT_MANIFEST_JSON),
        format!("[{}]", manifest_entry("d", 0, 1024, "Q4_K")),
    )
    .unwrap();
    let mut index = load_dir(tmp.path()).unwrap();
    index.load_down_features_q4k(tmp.path()).unwrap();
    assert!(index.has_down_features_kquant());
    assert!(index.down_features_q4k_layer_data(LAYER_0).is_none());
}

/// H1 contract pin at the writer: rows are padded to the 256-element
/// super-block, zero-filled. Readers must decode with the *padded*
/// stride (`shape[1]`), never the logical width — the deeper
/// regression suite lives in `kquant_decode.rs` / `kquant_cache.rs`
/// (`q4k_ffn_layer_decodes_row_padded_non_aligned_slabs`).
#[test]
fn writer_pads_rows_to_super_block() {
    const ROWS: usize = 2;
    const NON_ALIGNED_COLS: usize = 320;
    const PADDED_COLS: usize = 512;
    let data: Vec<f32> = (0..ROWS * NON_ALIGNED_COLS).map(|i| i as f32).collect();
    let (padded, padded_cols) = pad_cols_to_256(&data, ROWS, NON_ALIGNED_COLS);
    assert_eq!(padded_cols, PADDED_COLS);
    assert_eq!(padded.len(), ROWS * PADDED_COLS);
    // Row 1's logical data starts at the padded stride, and the pad
    // tail is zero-filled.
    assert_eq!(padded[PADDED_COLS], data[NON_ALIGNED_COLS]);
    assert!(padded[NON_ALIGNED_COLS..PADDED_COLS]
        .iter()
        .all(|&v| v == 0.0));
}
