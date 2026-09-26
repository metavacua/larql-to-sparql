//! Regression for fork bug #151 — `build_vindex_streaming` must produce
//! relation clusters.
//!
//! Before the fix, the streaming extract path never collected gate→down
//! offset directions and never invoked the clustering pipeline, so a
//! streaming-extracted vindex had an EMPTY knowledge graph:
//! `relation_clusters.json` and `feature_clusters.jsonl` were never
//! written (`STATS` → "no relation clusters found"). The in-memory
//! `build_vindex` path wrote them; streaming did not.
//!
//! This test builds a tiny synthetic safetensors model with enough
//! layers to have a knowledge band (llama/32 → layers 13..=25) and a
//! WordLevel tokenizer whose content words live at ids ≥
//! FIRST_CONTENT_TOKEN_ID. It drives `build_vindex_streaming` and
//! asserts the two cluster artifacts exist and are non-empty.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larql_vindex::{
    build_vindex_streaming, ExtractLevel, KquantWriteOptions, QuantFormat, SilentBuildCallbacks,
    StorageDtype, WriteWeightsOptions,
};

static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct TempDir(PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let pid = std::process::id();
        let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("larql_relclusters_{label}_{pid}_{n}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const NUM_LAYERS: usize = 32; // llama/32 → knowledge band layers 13..=25
const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 4;
// vocab: [UNK]=0 [PAD]=1 [BOS]=2 paris=3 france=4 berlin=5 germany=6
const VOCAB: usize = 7;
const CONTENT: [usize; 4] = [3, 4, 5, 6];

/// Write a small synthetic safetensors model + WordLevel tokenizer.
///
/// The embeddings are (near-)standard-basis rows so the down/gate
/// matrices we craft deterministically steer top tokens:
///   - `embed[id, id] = 1.0` for id < HIDDEN (one-hot in hidden space).
///   - down_proj feature `f` outputs content token `CONTENT[f]`.
///   - gate_proj feature `f` is activated by content token
///     `CONTENT[(f + 1) % 4]` — distinct from the output so the
///     offset = embed[out] - embed[in] is non-zero.
fn write_synth_model(model_dir: &Path) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": HIDDEN,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();

    // embed: row-major [VOCAB, HIDDEN], one-hot embed[id, id] for id<HIDDEN.
    let mut embed = vec![0.0f32; VOCAB * HIDDEN];
    for id in 0..VOCAB {
        embed[id * HIDDEN + id] = 1.0;
    }
    tensors.insert("model.embed_tokens.weight".into(), embed);
    metadata.push(("model.embed_tokens.weight".into(), vec![VOCAB, HIDDEN]));

    for layer in 0..NUM_LAYERS {
        // gate_proj: [INTERMEDIATE, HIDDEN]. gate[f, h]=1 → feature f's
        // top whole-word is the token whose embedding is axis h.
        let mut gate = vec![0.0f32; INTERMEDIATE * HIDDEN];
        for f in 0..INTERMEDIATE {
            let in_id = CONTENT[(f + 1) % 4];
            gate[f * HIDDEN + in_id] = 1.0;
        }
        tensors.insert(format!("model.layers.{layer}.mlp.gate_proj.weight"), gate);
        metadata.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        ));

        // down_proj: [HIDDEN, INTERMEDIATE]. down[out_id, f]=10 →
        // embed @ down[:,f] is maximal at vocab row out_id.
        let mut down = vec![0.0f32; HIDDEN * INTERMEDIATE];
        for f in 0..INTERMEDIATE {
            let out_id = CONTENT[f];
            down[out_id * INTERMEDIATE + f] = 10.0;
        }
        tensors.insert(format!("model.layers.{layer}.mlp.down_proj.weight"), down);
        metadata.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
        ));
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    // WordLevel tokenizer with content words at ids ≥ 3 so
    // `compute_offset_direction` (which filters ids < 3) keeps them.
    let tok_json = r#"{
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2,
                      "paris": 3, "france": 4, "berlin": 5, "germany": 6},
            "unk_token": "[UNK]"
        },
        "pre_tokenizer": {"type": "Whitespace"},
        "added_tokens": []
    }"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
}

fn run_extract(model_dir: &Path, output_dir: &Path) {
    let tok_bytes = std::fs::read(model_dir.join("tokenizer.json")).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(&tok_bytes).unwrap();
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        model_dir,
        &tokenizer,
        "test/relation-clusters",
        output_dir,
        5,
        0, // summary_features_per_expert (off)
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();
}

#[test]
fn streaming_extract_writes_relation_clusters() {
    let model = TempDir::new("model");
    write_synth_model(&model.0);

    let out = TempDir::new("out");
    run_extract(&model.0, &out.0);

    let clusters = out.0.join("relation_clusters.json");
    let feature_clusters = out.0.join("feature_clusters.jsonl");

    assert!(
        clusters.exists(),
        "relation_clusters.json must exist after streaming extract (#151)"
    );
    assert!(
        feature_clusters.exists(),
        "feature_clusters.jsonl must exist after streaming extract (#151)"
    );

    // Non-empty: the clustering pipeline early-returns (writing nothing)
    // when zero directions were collected, so a present-but-empty file
    // would mean the collection didn't actually run.
    let clusters_text = std::fs::read_to_string(&clusters).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&clusters_text).unwrap();
    let k = parsed["k"].as_u64().expect("k field present");
    assert!(
        k >= 1,
        "relation_clusters.json must have ≥1 cluster, got k={k}"
    );

    let jsonl = std::fs::read_to_string(&feature_clusters).unwrap();
    let lines: Vec<_> = jsonl.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        !lines.is_empty(),
        "feature_clusters.jsonl must have ≥1 feature assignment"
    );
    for line in &lines {
        let r: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(r.get("l").is_some(), "feature record has layer");
        assert!(r.get("f").is_some(), "feature record has feature index");
        assert!(r.get("c").is_some(), "feature record has cluster id");
    }
}
