//! Vindex v1 conformance — `down_meta.bin`.
//!
//! Contract: both readers (`read_binary` legacy heap path,
//! `mmap_binary` production path) treat every header count as
//! untrusted. Corrupt counts fail with `VindexError::Parse` *before*
//! sizing any allocation — the legacy path used to
//! `Vec::with_capacity` straight from attacker-controlled u32 fields
//! (the §3 LOW allocation bomb). Companion doc:
//! `docs/conformance-v1.md` §down_meta.

mod common;

use common::{load_dir, save_minimal_vindex, tokenizer};
use larql_vindex::format::down_meta;
use larql_vindex::format::filenames::DOWN_META_BIN;

const MAGIC: u32 = 0x444D4554; // "DMET"
const VERSION: u32 = 1;
const TOP_K: u32 = 2;

fn header(num_layers: u32, top_k: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC.to_le_bytes());
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&num_layers.to_le_bytes());
    bytes.extend_from_slice(&top_k.to_le_bytes());
    bytes
}

fn read_err(dir: &std::path::Path) -> String {
    down_meta::read_binary(dir, &tokenizer())
        .err()
        .expect("read_binary must error")
        .to_string()
}

fn mmap_err(dir: &std::path::Path) -> String {
    down_meta::mmap_binary(dir, std::sync::Arc::new(tokenizer()))
        .err()
        .expect("mmap_binary must error")
        .to_string()
}

/// The allocation-bomb regression: a header declaring u32::MAX layers
/// must fail fast on the file-size bound in BOTH readers — the legacy
/// path used to `Vec::with_capacity(u32::MAX)` first.
#[test]
fn absurd_layer_count_is_error_not_allocation() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(DOWN_META_BIN), header(u32::MAX, TOP_K)).unwrap();
    assert!(read_err(tmp.path()).contains("declares 4294967295 layers"));
    assert!(mmap_err(tmp.path()).contains("declares 4294967295 layers"));
}

/// A layer declaring more features than the file can hold errors
/// before the per-feature allocation.
#[test]
fn absurd_feature_count_is_error_not_allocation() {
    let tmp = tempfile::tempdir().unwrap();
    let mut bytes = header(1, TOP_K);
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_features
    std::fs::write(tmp.path().join(DOWN_META_BIN), &bytes).unwrap();
    // u32::MAX features × record_size overflows the checked multiply.
    assert!(
        read_err(tmp.path()).contains("overflow") || read_err(tmp.path()).contains("truncated")
    );
    assert!(
        mmap_err(tmp.path()).contains("overflow") || mmap_err(tmp.path()).contains("truncated")
    );
}

/// A plausible-but-unbacked feature count (1000 features, no records)
/// is a truncation error in both readers.
#[test]
fn truncated_records_are_error() {
    const UNBACKED_FEATURES: u32 = 1000;
    let tmp = tempfile::tempdir().unwrap();
    let mut bytes = header(1, TOP_K);
    bytes.extend_from_slice(&UNBACKED_FEATURES.to_le_bytes());
    std::fs::write(tmp.path().join(DOWN_META_BIN), &bytes).unwrap();
    assert!(
        read_err(tmp.path()).contains("truncated"),
        "{}",
        read_err(tmp.path())
    );
    assert!(
        mmap_err(tmp.path()).contains("truncated"),
        "{}",
        mmap_err(tmp.path())
    );
}

/// A huge top_k combined with a feature count overflows the checked
/// record-size/layer-size arithmetic instead of wrapping.
#[test]
fn record_size_overflow_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let mut bytes = header(1, u32::MAX);
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_features
    std::fs::write(tmp.path().join(DOWN_META_BIN), &bytes).unwrap();
    assert!(
        read_err(tmp.path()).contains("overflow"),
        "{}",
        read_err(tmp.path())
    );
    assert!(
        mmap_err(tmp.path()).contains("overflow"),
        "{}",
        mmap_err(tmp.path())
    );
}

/// Truncated mid-header: shorter than the 16-byte header.
#[test]
fn truncated_header_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(DOWN_META_BIN), [0u8; 8]).unwrap();
    assert!(mmap_err(tmp.path()).contains("too small"));
    // The buffered reader hits EOF inside the header read.
    assert!(down_meta::read_binary(tmp.path(), &tokenizer()).is_err());
}

/// A present-but-corrupt down_meta.bin fails the WHOLE vindex load —
/// no silent fallback to the legacy JSONL path.
#[test]
fn corrupt_down_meta_fails_full_vindex_load() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let mut bytes = header(1, TOP_K);
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(tmp.path().join(DOWN_META_BIN), &bytes).unwrap();
    let err = load_dir(tmp.path())
        .err()
        .expect("corrupt down_meta fails load");
    assert!(err.to_string().contains("down_meta.bin"), "{err}");
}

/// Positive control: the legacy heap reader still round-trips a valid
/// file after the bounds hardening.
#[test]
fn valid_file_round_trips_through_read_binary() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let (loaded, total) = down_meta::read_binary(tmp.path(), &tokenizer()).unwrap();
    assert_eq!(loaded.len(), common::FIXTURE_LAYERS);
    assert_eq!(total, common::FIXTURE_LAYERS * common::FIXTURE_FEATURES);
}
