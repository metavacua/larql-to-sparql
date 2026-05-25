#![cfg(all(test, target_arch = "wasm32"))]

use larql_vindex_wasm32uu::VindexSession;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_experimental);

fn minimal_config_json(num_layers: usize, hidden_size: usize) -> String {
    serde_json::json!({
        "version": 1,
        "model": "test/model",
        "family": "test",
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,
        "vocab_size": 256,
        "embed_scale": 1.0,
        "layers": (0..num_layers).map(|i| serde_json::json!({
            "layer": i,
            "num_features": 0,
            "offset": 0,
            "length": 0
        })).collect::<Vec<_>>(),
        "down_top_k": 5
    })
    .to_string()
}

#[wasm_bindgen_test]
fn smoke_construct() {
    let json = minimal_config_json(2, 64);
    let session = VindexSession::new(&json).expect("construct from valid config");
    assert_eq!(session.num_layers(), 2);
    assert_eq!(session.hidden_size(), 64);
    assert!(!session.has_gate_vectors());
}

#[wasm_bindgen_test]
fn config_round_trips() {
    let json = minimal_config_json(4, 128);
    let session = VindexSession::new(&json).expect("construct");
    let rt: serde_json::Value =
        serde_json::from_str(&session.config()).expect("config() is valid JSON");
    assert_eq!(rt["num_layers"].as_u64().unwrap(), 4);
    assert_eq!(rt["hidden_size"].as_u64().unwrap(), 128);
    assert_eq!(rt["model"].as_str().unwrap(), "test/model");
}

#[wasm_bindgen_test]
fn bad_config_returns_error() {
    assert!(VindexSession::new("not json {{").is_err());
}

#[wasm_bindgen_test]
fn gate_knn_empty_returns_empty_json() {
    let json = minimal_config_json(1, 16);
    let session = VindexSession::new(&json).expect("construct");
    let result = session.gate_knn(0, &vec![0.0f32; 16], 5);
    let parsed: serde_json::Value = serde_json::from_str(&result).expect("valid JSON");
    assert!(parsed.is_array());
    assert_eq!(parsed.as_array().unwrap().len(), 0);
}

#[wasm_bindgen_test]
fn load_gate_bytes_f32() {
    let num_layers = 1usize;
    let hidden = 4usize;
    let num_feat = 3usize;

    // Build a synthetic gate_vectors.bin: 3 features × 4 hidden, f32 LE.
    let data: Vec<f32> = (0..(num_feat * hidden)).map(|i| i as f32).collect();
    let gate_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let config_json = serde_json::json!({
        "version": 1,
        "model": "test/model",
        "family": "test",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": 16,
        "vocab_size": 256,
        "embed_scale": 1.0,
        "layers": [{
            "layer": 0,
            "num_features": num_feat,
            "offset": 0,
            "length": gate_bytes.len()
        }],
        "down_top_k": 5
    })
    .to_string();

    let mut session = VindexSession::new(&config_json).expect("construct");
    session
        .load_gate_bytes(&gate_bytes)
        .expect("load gate bytes");

    assert!(session.has_gate_vectors());
    assert_eq!(session.num_features(0), num_feat as u32);
}
