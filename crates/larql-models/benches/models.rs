use std::{fs, path::Path};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_models::{
    detect_from_json, detect_from_json_validated, is_ffn_tensor, load_model_dir_validated,
    quant::ggml,
};
use serde_json::json;

const SYNTHETIC_LAYERS: usize = 4;
const SYNTHETIC_HIDDEN: usize = 64;
const SYNTHETIC_INTERMEDIATE: usize = 128;
const SYNTHETIC_VOCAB: usize = 256;
const QUANT_ELEMENTS: usize = 8192;

struct TensorSpec {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn llama_config() -> serde_json::Value {
    json!({
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "rms_norm_eps": 0.000001,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
    })
}

fn gemma4_config() -> serde_json::Value {
    json!({
        "model_type": "gemma4_text",
        "num_hidden_layers": 34,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 256000,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "rope_local_base_freq": 10000.0,
        "rope_global_base_freq": 1000000.0,
        "sliding_window": 1024,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention"
        ],
        "rope_scaling": {
            "rope_type": "default",
            "factor": 1.0
        }
    })
}

fn gpt_oss_config() -> serde_json::Value {
    json!({
        "model_type": "gpt_oss",
        "num_hidden_layers": 24,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 201088,
        "head_dim": 64,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "rope_theta": 150000.0
    })
}

fn bench_config_detection(c: &mut Criterion) {
    let configs = [
        ("llama", llama_config()),
        ("gemma4", gemma4_config()),
        ("gpt_oss", gpt_oss_config()),
    ];
    let mut group = c.benchmark_group("config_detection");

    for (name, config) in configs {
        group.bench_with_input(BenchmarkId::new("detect", name), &config, |b, config| {
            b.iter(|| {
                let arch = detect_from_json(black_box(config));
                black_box(arch.family());
            });
        });
        group.bench_with_input(
            BenchmarkId::new("detect_validated", name),
            &config,
            |b, config| {
                b.iter(|| {
                    let arch = detect_from_json_validated(black_box(config)).unwrap();
                    black_box(arch.family());
                });
            },
        );
    }

    group.finish();
}

fn bench_config_validation(c: &mut Criterion) {
    let configs = [
        ("llama", llama_config()),
        ("gemma4", gemma4_config()),
        ("gpt_oss", gpt_oss_config()),
    ];
    let mut group = c.benchmark_group("config_validation");

    for (name, config) in configs {
        let arch = detect_from_json(&config);
        group.bench_with_input(BenchmarkId::from_parameter(name), &arch, |b, arch| {
            b.iter(|| {
                black_box(arch.validate().is_ok());
            });
        });
    }

    group.finish();
}

fn bench_tensor_key_generation(c: &mut Criterion) {
    let config = gemma4_config();
    let arch = detect_from_json(&config);
    let mut group = c.benchmark_group("tensor_keys");

    group.bench_function("gemma4_all_layer_hot_keys", |b| {
        b.iter(|| {
            let mut bytes = 0usize;
            for layer in 0..arch.config().num_layers {
                bytes += black_box(arch.attn_q_key(layer)).len();
                bytes += black_box(arch.attn_k_key(layer)).len();
                bytes += black_box(arch.attn_v_key(layer)).len();
                bytes += black_box(arch.attn_o_key(layer)).len();
                bytes += black_box(arch.ffn_gate_key(layer)).len();
                bytes += black_box(arch.ffn_up_key(layer)).len();
                bytes += black_box(arch.ffn_down_key(layer)).len();
                if let Some(key) = arch.attn_q_norm_key(layer) {
                    bytes += black_box(key).len();
                }
                if let Some(key) = arch.attn_k_norm_key(layer) {
                    bytes += black_box(key).len();
                }
                if let Some(key) = arch.per_layer_embed_key() {
                    bytes += black_box(key).len();
                }
            }
            black_box(bytes);
        });
    });

    group.finish();
}

fn bench_ffn_tensor_classification(c: &mut Criterion) {
    const KEYS: &[&str] = &[
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.12.block_sparse_moe.experts.7.w1.weight",
        "model.layers.12.block_sparse_moe.experts.7.w2.weight",
        "model.layers.12.block_sparse_moe.gate.weight",
        "model.layers.18.mlp.router.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.2.self_attn.v_proj.weight",
        "model.layers.2.self_attn.o_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];
    let mut group = c.benchmark_group("tensor_classification");

    group.bench_function("is_ffn_tensor_key_set", |b| {
        b.iter(|| {
            let mut ffn_count = 0usize;
            for key in KEYS {
                ffn_count += usize::from(is_ffn_tensor(black_box(key)));
            }
            black_box(ffn_count);
        });
    });

    group.finish();
}

fn bench_quant_decode(c: &mut Criterion) {
    let source: Vec<f32> = (0..QUANT_ELEMENTS)
        .map(|idx| ((idx % 97) as f32 - 48.0) / 17.0)
        .collect();
    let q4_0 = ggml::quantize_q4_0(&source);
    let q8_0 = ggml::quantize_q8_0(&source);
    let q4_1 = vec![0u8; ggml::tensor_data_size(ggml::TYPE_Q4_1, QUANT_ELEMENTS).unwrap()];
    let q5_0 = vec![0u8; ggml::tensor_data_size(ggml::TYPE_Q5_0, QUANT_ELEMENTS).unwrap()];
    let q5_1 = vec![0u8; ggml::tensor_data_size(ggml::TYPE_Q5_1, QUANT_ELEMENTS).unwrap()];
    let q4_k = synth_q4k_data(QUANT_ELEMENTS, 1000);
    let q6_k = synth_q6k_data(QUANT_ELEMENTS, 2000);
    let mut group = c.benchmark_group("quant_decode");
    group.throughput(Throughput::Elements(QUANT_ELEMENTS as u64));

    for (name, tensor_type, data) in [
        ("q4_0", ggml::TYPE_Q4_0, q4_0.as_slice()),
        ("q4_1", ggml::TYPE_Q4_1, q4_1.as_slice()),
        ("q5_0", ggml::TYPE_Q5_0, q5_0.as_slice()),
        ("q5_1", ggml::TYPE_Q5_1, q5_1.as_slice()),
        ("q8_0", ggml::TYPE_Q8_0, q8_0.as_slice()),
        ("q4_k", ggml::TYPE_Q4_K, q4_k.as_slice()),
        ("q6_k", ggml::TYPE_Q6_K, q6_k.as_slice()),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), data, |b, data| {
            b.iter(|| {
                let decoded =
                    ggml::dequantize(black_box(data), tensor_type, QUANT_ELEMENTS).unwrap();
                black_box(decoded);
            });
        });
    }

    group.finish();
}

fn synth_q4k_data(elements: usize, seed: u32) -> Vec<u8> {
    assert!(elements.is_multiple_of(256));
    let mut data = Vec::with_capacity(elements / 256 * 144);
    for block_idx in 0..elements / 256 {
        data.extend_from_slice(&synth_q4k_block(seed + block_idx as u32));
    }
    data
}

fn synth_q4k_block(seed: u32) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    let mut state = seed;
    for byte in &mut block[4..144] {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 16) as u8;
    }
    // d = dmin = 0.0625 as f16. This keeps nonzero synthetic values bounded.
    block[0] = 0x00;
    block[1] = 0x2C;
    block[2] = 0x00;
    block[3] = 0x2C;
    block
}

fn synth_q6k_data(elements: usize, seed: u32) -> Vec<u8> {
    assert!(elements.is_multiple_of(256));
    let mut data = Vec::with_capacity(elements / 256 * 210);
    for block_idx in 0..elements / 256 {
        data.extend_from_slice(&synth_q6k_block(seed + block_idx as u32));
    }
    data
}

fn synth_q6k_block(seed: u32) -> Vec<u8> {
    let mut block = vec![0u8; 210];
    let mut state = seed;
    for byte in &mut block[..208] {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 16) as u8;
    }
    // d = 0.0625 as f16.
    block[208] = 0x00;
    block[209] = 0x2C;
    block
}

fn bench_synthetic_safetensors_loading(c: &mut Criterion) {
    let tempdir = tempfile::tempdir().unwrap();
    write_synthetic_model(tempdir.path());
    let mut group = c.benchmark_group("weight_loading");
    group.sample_size(10);
    group.throughput(Throughput::Elements((SYNTHETIC_LAYERS * 7 + 3) as u64));

    group.bench_function("load_synthetic_safetensors_validated", |b| {
        b.iter(|| {
            let weights = load_model_dir_validated(black_box(tempdir.path())).unwrap();
            black_box(weights.tensors.len());
        });
    });

    group.finish();
}

fn write_synthetic_model(dir: &Path) {
    let config = json!({
        "model_type": "llama",
        "num_hidden_layers": SYNTHETIC_LAYERS,
        "hidden_size": SYNTHETIC_HIDDEN,
        "intermediate_size": SYNTHETIC_INTERMEDIATE,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": SYNTHETIC_VOCAB,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0
    });
    fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    let mut tensors = vec![
        tensor(
            "model.embed_tokens.weight",
            &[SYNTHETIC_VOCAB, SYNTHETIC_HIDDEN],
            1,
        ),
        tensor("model.norm.weight", &[SYNTHETIC_HIDDEN], 2),
        tensor("lm_head.weight", &[SYNTHETIC_VOCAB, SYNTHETIC_HIDDEN], 3),
    ];

    for layer in 0..SYNTHETIC_LAYERS {
        let prefix = format!("model.layers.{layer}");
        tensors.push(tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[SYNTHETIC_HIDDEN, SYNTHETIC_HIDDEN],
            layer as u32 + 10,
        ));
        tensors.push(tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[SYNTHETIC_HIDDEN, SYNTHETIC_HIDDEN],
            layer as u32 + 20,
        ));
        tensors.push(tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[SYNTHETIC_HIDDEN, SYNTHETIC_HIDDEN],
            layer as u32 + 30,
        ));
        tensors.push(tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[SYNTHETIC_HIDDEN, SYNTHETIC_HIDDEN],
            layer as u32 + 40,
        ));
        tensors.push(tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[SYNTHETIC_INTERMEDIATE, SYNTHETIC_HIDDEN],
            layer as u32 + 50,
        ));
        tensors.push(tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[SYNTHETIC_INTERMEDIATE, SYNTHETIC_HIDDEN],
            layer as u32 + 60,
        ));
        tensors.push(tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[SYNTHETIC_HIDDEN, SYNTHETIC_INTERMEDIATE],
            layer as u32 + 70,
        ));
    }

    fs::write(dir.join("model.safetensors"), encode_safetensors(&tensors)).unwrap();
}

fn tensor(name: &str, shape: &[usize], seed: u32) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        dtype: "F32",
        shape: shape.to_vec(),
        bytes: f32_bytes(shape.iter().product(), seed),
    }
}

fn f32_bytes(elements: usize, seed: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elements * 4);
    for idx in 0..elements {
        let bits = (idx as u32)
            .wrapping_mul(1_664_525)
            .wrapping_add(seed.wrapping_mul(1_013_904_223));
        let value = (bits % 4096) as f32 / 4096.0;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_safetensors(tensors: &[TensorSpec]) -> Vec<u8> {
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();

    for tensor in tensors {
        let end = offset + tensor.bytes.len();
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut output = Vec::with_capacity(8 + header_bytes.len() + offset);
    output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(&header_bytes);
    for tensor in tensors {
        output.extend_from_slice(&tensor.bytes);
    }
    output
}

criterion_group!(
    benches,
    bench_config_detection,
    bench_config_validation,
    bench_tensor_key_generation,
    bench_ffn_tensor_classification,
    bench_quant_decode,
    bench_synthetic_safetensors_loading
);
criterion_main!(benches);
