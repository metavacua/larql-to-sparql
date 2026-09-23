//! Vindex v1 conformance — little-endian byte-order golden vectors.
//!
//! Round-trip equality can't catch an endianness regression (a
//! consistently big-endian encoder/decoder pair round-trips fine), so
//! these tests assert the EXACT on-disk bytes of every LE codec a v1
//! vindex ships: `le_floats`, the `.vlp` base64 float embedding,
//! `down_meta.bin`, and `.lknn`. They are the cross-platform guard —
//! no BE CI runner exists, but any target that produces different
//! bytes fails here. Companion doc: `docs/conformance-v1.md`
//! §byte-order.

mod common;

use larql_vindex::format::down_meta;
use larql_vindex::format::filenames::DOWN_META_BIN;
use larql_vindex::format::le_floats::{decode_f32_le, encode_f32_le};
use larql_vindex::patch::format::{decode_gate_vector, encode_gate_vector};
use larql_vindex::{FeatureMeta, KnnStore};

/// 1.0f32 = 0x3F800000, -2.0 = 0xC0000000, 0.5 = 0x3F000000,
/// -0.0 = 0x80000000 — little-endian byte order on disk.
const GOLDEN_FLOATS: [f32; 4] = [1.0, -2.0, 0.5, -0.0];
#[rustfmt::skip]
const GOLDEN_FLOAT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x80, 0x3F,
    0x00, 0x00, 0x00, 0xC0,
    0x00, 0x00, 0x00, 0x3F,
    0x00, 0x00, 0x00, 0x80,
];

#[test]
fn le_floats_encode_matches_golden_bytes() {
    assert_eq!(encode_f32_le(&GOLDEN_FLOATS), GOLDEN_FLOAT_BYTES);
}

#[test]
fn le_floats_decode_matches_golden_values_bit_exact() {
    let decoded = decode_f32_le(&GOLDEN_FLOAT_BYTES);
    assert_eq!(decoded.len(), GOLDEN_FLOATS.len());
    for (d, g) in decoded.iter().zip(GOLDEN_FLOATS.iter()) {
        assert_eq!(d.to_bits(), g.to_bits());
    }
}

/// The `.vlp` embedded-vector codec: base64 over LE f32 bytes.
/// [00 00 80 3F] → "AACAPw==".
#[test]
fn vlp_base64_gate_vector_matches_golden() {
    assert_eq!(encode_gate_vector(&[1.0f32]), "AACAPw==");
    let back = decode_gate_vector("AACAPw==").unwrap();
    assert_eq!(back.len(), 1);
    assert_eq!(back[0].to_bits(), 1.0f32.to_bits());
}

/// down_meta.bin: full-file golden for one layer / one feature /
/// top_k = 1. Header magic 0x444D4554 serialises LE as b"TEMD".
#[test]
fn down_meta_bin_matches_golden_bytes() {
    const TOP_TOKEN_ID: u32 = 3;
    const C_SCORE: f32 = 0.5;
    const TOP_K_TOKEN_ID: u32 = 7;
    const TOP_K_LOGIT: f32 = 0.25;
    let tmp = tempfile::tempdir().unwrap();
    let meta = vec![Some(vec![Some(FeatureMeta {
        top_token: "tok".into(),
        top_token_id: TOP_TOKEN_ID,
        c_score: C_SCORE,
        top_k: vec![larql_models::TopKEntry {
            token: "t".into(),
            token_id: TOP_K_TOKEN_ID,
            logit: TOP_K_LOGIT,
        }],
    })])];
    down_meta::write_binary(tmp.path(), &meta, 1).unwrap();
    let bytes = std::fs::read(tmp.path().join(DOWN_META_BIN)).unwrap();

    #[rustfmt::skip]
    let golden: [u8; 36] = [
        0x54, 0x45, 0x4D, 0x44, // magic 0x444D4554 LE
        0x01, 0x00, 0x00, 0x00, // version 1
        0x01, 0x00, 0x00, 0x00, // num_layers 1
        0x01, 0x00, 0x00, 0x00, // top_k 1
        0x01, 0x00, 0x00, 0x00, // layer 0: num_features 1
        0x03, 0x00, 0x00, 0x00, // top_token_id 3
        0x00, 0x00, 0x00, 0x3F, // c_score 0.5 (0x3F000000)
        0x07, 0x00, 0x00, 0x00, // top_k[0].token_id 7
        0x00, 0x00, 0x80, 0x3E, // top_k[0].logit 0.25 (0x3E800000)
    ];
    assert_eq!(bytes, golden);
}

/// .lknn: header + f16 keys + target ids + JSON meta blob, all LE.
/// `KnnStore::add` L2-normalises the key, so [1, -1] is stored as
/// ±1/√2; f32→f16: 0.7071 → 0x39A8 → [A8 39]; -0.7071 → [A8 B9].
#[test]
fn lknn_matches_golden_bytes() {
    const LAYER: usize = 3;
    const TARGET_ID: u32 = 42;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("knn_store.bin");
    let mut store = KnnStore::default();
    store.add(
        LAYER,
        vec![1.0, -1.0],
        TARGET_ID,
        "Paris".into(),
        "France".into(),
        "capital".into(),
        1.0,
    );
    store.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();

    // The meta blob is JSON; build the writer's exact payload so the
    // golden stays robust to serde_json map ordering.
    let meta = serde_json::to_vec(&serde_json::json!({
        "target_token": "Paris",
        "entity": "France",
        "relation": "capital",
        "confidence": 1.0f32,
    }))
    .unwrap();

    let mut golden = Vec::new();
    golden.extend_from_slice(b"LKNN");
    golden.extend_from_slice(&1u32.to_le_bytes()); // version
    golden.extend_from_slice(&2u32.to_le_bytes()); // dim
    golden.extend_from_slice(&1u32.to_le_bytes()); // n_layers
    golden.extend_from_slice(&(LAYER as u32).to_le_bytes());
    golden.extend_from_slice(&1u32.to_le_bytes()); // n_entries
    golden.extend_from_slice(&[0xA8, 0x39, 0xA8, 0xB9]); // f16 keys (normalised)
    golden.extend_from_slice(&TARGET_ID.to_le_bytes());
    golden.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    golden.extend_from_slice(&meta);
    assert_eq!(bytes, golden);
}
