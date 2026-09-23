//! Colocated tests for the separate-tensor MoE layer writer.
//!
//! The regression these defend against is specific and was live: extraction of
//! a separate-tensor MoE reported success, verified clean, sliced clean, and
//! then panicked on the first decoded token because no expert store had been
//! written. So the assertions are about *files appearing on disk with the
//! right shape*, not about the quantiser — which `write_layers_parts_tests`
//! covers separately.

use std::collections::HashMap;
use std::path::Path;

use super::super::write_f32::WeightSource;
use super::moe_layers_per_expert::write_per_layer_moe_per_expert;
use crate::format::weights::write_layers::parse_layer_weights_header;

const HIDDEN: usize = 256;
const INTER: usize = 256;
const NUM_LAYERS: usize = 2;
const NUM_EXPERTS: usize = 3;

/// A `WeightSource` backed by an explicit tensor map, so a test can express
/// "this expert is missing" precisely.
struct MapSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    tensors: HashMap<String, (Vec<f32>, usize, usize)>,
}

impl MapSource {
    /// An OLMoE-shaped architecture — the real separate-tensor case.
    fn olmoe(num_experts: usize) -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": "olmoe",
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
        }))
    }

    /// Every expert of every layer present and correctly shaped.
    fn complete() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        for layer in 0..NUM_LAYERS {
            for expert in 0..NUM_EXPERTS {
                insert_expert(&mut tensors, &*arch, layer, expert);
            }
        }
        Self { arch, tensors }
    }

    /// Layer 0 complete; layer 1 missing every expert (a hybrid dense layer).
    fn first_layer_only() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        for expert in 0..NUM_EXPERTS {
            insert_expert(&mut tensors, &*arch, 0, expert);
        }
        Self { arch, tensors }
    }

    /// Layer 0 has expert 0 but not expert 1 — a genuinely malformed layer.
    fn partial_layer() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        insert_expert(&mut tensors, &*arch, 0, 0);
        Self { arch, tensors }
    }
}

fn insert_expert(
    tensors: &mut HashMap<String, (Vec<f32>, usize, usize)>,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    expert: usize,
) {
    // Distinct fill per (layer, expert) so a mixed-up write is detectable.
    let fill = (layer * 10 + expert) as f32;
    if let Some(k) = arch.expert_ffn_gate_key(layer, expert) {
        tensors.insert(k, (vec![fill; INTER * HIDDEN], INTER, HIDDEN));
    }
    if let Some(k) = arch.expert_ffn_up_key(layer, expert) {
        tensors.insert(k, (vec![-fill; INTER * HIDDEN], INTER, HIDDEN));
    }
    if let Some(k) = arch.expert_ffn_down_key(layer, expert) {
        tensors.insert(k, (vec![fill; HIDDEN * INTER], HIDDEN, INTER));
    }
}

impl WeightSource for MapSource {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        self.tensors.get(key).cloned()
    }
    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        NUM_LAYERS
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("moe-per-expert-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn layer_file(dir: &Path, layer: usize) -> std::path::PathBuf {
    dir.join(crate::format::filenames::layer_weights_filename(layer))
}

#[test]
fn a_separate_tensor_moe_gets_an_expert_store() {
    // The headline regression: this used to write nothing at all.
    let dir = temp_dir("complete");
    let written = write_per_layer_moe_per_expert(&MapSource::complete(), &dir, NUM_LAYERS).unwrap();
    assert_eq!(written, NUM_LAYERS);
    for layer in 0..NUM_LAYERS {
        assert!(
            layer_file(&dir, layer).exists(),
            "layer {layer} not written"
        );
    }
}

#[test]
fn each_layer_file_declares_every_expert() {
    let dir = temp_dir("entries");
    write_per_layer_moe_per_expert(&MapSource::complete(), &dir, NUM_LAYERS).unwrap();
    let bytes = std::fs::read(layer_file(&dir, 0)).unwrap();
    let (_, num_entries, inter, hidden, offsets) = parse_layer_weights_header(&bytes).unwrap();
    assert_eq!(num_entries, NUM_EXPERTS);
    assert_eq!(inter, INTER);
    assert_eq!(hidden, HIDDEN);
    assert_eq!(offsets.len(), NUM_EXPERTS);
}

#[test]
fn every_expert_region_is_inside_the_file() {
    // An offset that is merely a plausible number is the failure mode the
    // whole layer format is designed against; check it holds here too.
    let dir = temp_dir("bounds");
    write_per_layer_moe_per_expert(&MapSource::complete(), &dir, NUM_LAYERS).unwrap();
    let bytes = std::fs::read(layer_file(&dir, 1)).unwrap();
    let (_, _, _, _, offsets) = parse_layer_weights_header(&bytes).unwrap();
    for (gu_off, gu_len, dn_off, dn_len) in offsets {
        assert!(gu_off + gu_len <= bytes.len(), "gate_up region overruns");
        assert!(dn_off + dn_len <= bytes.len(), "down region overruns");
    }
}

#[test]
fn a_layer_without_experts_is_skipped_not_written_short() {
    // A hybrid stack's dense layer. Writing a zero-entry file here would look
    // like a valid expert store to every downstream check.
    let dir = temp_dir("hybrid");
    let written =
        write_per_layer_moe_per_expert(&MapSource::first_layer_only(), &dir, NUM_LAYERS).unwrap();
    assert_eq!(written, 1);
    assert!(layer_file(&dir, 0).exists());
    assert!(!layer_file(&dir, 1).exists());
}

#[test]
fn a_partially_present_layer_is_refused() {
    // Expert 0 resolves but expert 1 does not. Writing the short list would
    // silently drop experts that routing will later select.
    let dir = temp_dir("partial");
    let err =
        write_per_layer_moe_per_expert(&MapSource::partial_layer(), &dir, NUM_LAYERS).unwrap_err();
    let s = err.to_string();
    assert!(s.contains("expert 1"), "{s}");
    assert!(s.contains("only 1 are present"), "{s}");
}

#[test]
fn a_packed_model_is_left_to_the_packed_writer() {
    // Gemma 4 is PackedBF16; this writer must decline it rather than race the
    // packed path to the same filenames.
    struct PackedSource(Box<dyn larql_models::ModelArchitecture>);
    impl WeightSource for PackedSource {
        fn get_tensor(&self, _k: &str) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn get_vector(&self, _k: &str) -> Option<Vec<f32>> {
            None
        }
        fn arch(&self) -> &dyn larql_models::ModelArchitecture {
            &*self.0
        }
        fn num_layers(&self) -> usize {
            NUM_LAYERS
        }
        fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn vector_names(&self) -> Vec<String> {
            Vec::new()
        }
        fn get_packed_bf16(&self, _k: &str) -> Option<Vec<u8>> {
            None
        }
    }
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gemma4_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
    }));
    let dir = temp_dir("packed");
    let written = write_per_layer_moe_per_expert(&PackedSource(arch), &dir, NUM_LAYERS).unwrap();
    assert_eq!(written, 0, "packed models must not be handled here");
    assert!(!layer_file(&dir, 0).exists());
}

#[test]
fn a_dense_model_is_declined() {
    struct DenseSource(Box<dyn larql_models::ModelArchitecture>);
    impl WeightSource for DenseSource {
        fn get_tensor(&self, _k: &str) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn get_vector(&self, _k: &str) -> Option<Vec<f32>> {
            None
        }
        fn arch(&self) -> &dyn larql_models::ModelArchitecture {
            &*self.0
        }
        fn num_layers(&self) -> usize {
            NUM_LAYERS
        }
        fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn vector_names(&self) -> Vec<String> {
            Vec::new()
        }
        fn get_packed_bf16(&self, _k: &str) -> Option<Vec<u8>> {
            None
        }
    }
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
    }));
    let dir = temp_dir("dense");
    assert_eq!(
        write_per_layer_moe_per_expert(&DenseSource(arch), &dir, NUM_LAYERS).unwrap(),
        0
    );
}
