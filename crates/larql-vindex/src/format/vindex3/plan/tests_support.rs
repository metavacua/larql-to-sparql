//! Shared fixtures: Glimmer-shaped inventories built through the real
//! inventory pipeline (no hand-written report structs), so plan tests
//! exercise the same shapes `inspect-hf` produces.

use std::io::Write;
use std::path::Path;

use larql_models::inventory::{build_inventory, ArchitectureInventory};

/// Number of layers in the fixture target model.
pub const FIXTURE_LAYERS: usize = 8;

/// Write a config + one shard (with real payload bytes) and build its
/// inventory. Payloads are a deterministic per-offset pattern so encode
/// tests can compare bytes end to end, not just counts.
fn inventory_from(
    dir: &Path,
    config: &serde_json::Value,
    header: &serde_json::Value,
) -> ArchitectureInventory {
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let header_bytes = serde_json::to_vec(header).unwrap();
    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    let payload_len = header
        .as_object()
        .unwrap()
        .values()
        .filter_map(|d| d["data_offsets"].as_array())
        .filter_map(|offs| offs.get(1)?.as_u64())
        .max()
        .unwrap_or(0);
    file.write_all(&payload_pattern(payload_len)).unwrap();
    build_inventory(dir).unwrap()
}

/// Deterministic payload bytes for fixture shards: `f(i) = (i * 31 + 7) mod 251`.
pub fn payload_pattern(len: u64) -> Vec<u8> {
    (0..len)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) % 251) as u8)
        .collect()
}

/// Build an artifact from an explicit tensor list `(name, shape)`,
/// offsets sequential — for closure-variant fixtures whose whole point
/// is a precise operand estate.
pub fn custom_artifact(
    dir: &Path,
    config: &serde_json::Value,
    tensors: &[(&str, &[usize])],
) -> ArchitectureInventory {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in tensors {
        push_tensor(&mut header, &mut offset, name, shape);
    }
    inventory_from(dir, config, &serde_json::Value::Object(header))
}

/// Append one BF16 tensor at the running offset.
fn push_tensor(
    header: &mut serde_json::Map<String, serde_json::Value>,
    offset: &mut u64,
    name: &str,
    shape: &[usize],
) {
    let len = 2 * shape.iter().product::<usize>() as u64;
    header.insert(
        name.to_string(),
        serde_json::json!({
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [*offset, *offset + len],
        }),
    );
    *offset += len;
}

/// A Glimmer-shaped target: unknown family, hybrid attention, per-layer
/// rope array with NoPE zeros on the global layers, dangerous unconsumed
/// scalars, a vision subtree, mixed tensor stack.
pub fn glimmer_shaped_target(dir: &Path) -> ArchitectureInventory {
    let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let layer_rope_theta: Vec<f64> = (0..FIXTURE_LAYERS)
        .map(|i| if i % 4 == 3 { 0.0 } else { 500000.0 })
        .collect();
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 64,
            "num_hidden_layers": FIXTURE_LAYERS,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "sliding_window": 16,
            "rms_norm_eps": 1e-5,
            "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
            "layer_types": layer_types,
            "layer_rope_theta": layer_rope_theta,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8
        },
        "vision_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-6
        }
    });
    // Full operand estate, mirroring the real four-norm Glimmer layer
    // anatomy, attention gate included (its semantics are judged on the
    // registered family): 12 tensors per layer.
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    push_tensor(
        &mut header,
        &mut offset,
        "model.language_model.embed_tokens.weight",
        &[128, 64],
    );
    push_tensor(
        &mut header,
        &mut offset,
        "model.vision_tower.layers.0.attn.qkv.weight",
        &[32, 32],
    );
    push_tensor(&mut header, &mut offset, "lm_head.weight", &[128, 64]);
    push_tensor(
        &mut header,
        &mut offset,
        "model.language_model.norm.weight",
        &[64],
    );
    for layer in 0..FIXTURE_LAYERS {
        let stack = format!("model.language_model.layers.{layer}");
        // 8 q-heads * head_dim 8 = 64 rows; 2 kv-heads * 8 = 16 rows.
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.q_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.k_proj.weight"),
            &[16, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.v_proj.weight"),
            &[16, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.o_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.gate_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.mlp.gate_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.mlp.up_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.mlp.down_proj.weight"),
            &[64, 256],
        );
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            push_tensor(
                &mut header,
                &mut offset,
                &format!("{stack}.{norm}.weight"),
                &[64],
            );
        }
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

/// A drafter-shaped artifact declaring `target_layer_ids` taps into a
/// deeper producer.
pub fn drafter_shaped(dir: &Path) -> ArchitectureInventory {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerAssistantModel"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "sliding_window": 16,
        "block_size": 4,
        "mask_token_id": 99,
        "target_layer_ids": [1, 3, 5]
    });
    // The first four tensors keep their historical offsets — encode tests
    // slice the source pattern at [8320, 32896) for the projector. The
    // appended estate mirrors the real assistant's two-norm + QK-norm
    // layer anatomy (11 tensors per layer).
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    push_tensor(
        &mut header,
        &mut offset,
        "layers.0.self_attn.q_proj.weight",
        &[64, 64],
    );
    push_tensor(&mut header, &mut offset, "norm.weight", &[64]);
    push_tensor(&mut header, &mut offset, "encoder.fc.weight", &[192, 64]);
    push_tensor(
        &mut header,
        &mut offset,
        "encoder.output_norm_enc.weight",
        &[64],
    );
    for layer in 0..2 {
        if layer > 0 {
            push_tensor(
                &mut header,
                &mut offset,
                &format!("layers.{layer}.self_attn.q_proj.weight"),
                &[64, 64],
            );
        }
        // 8 q-heads * head_dim 8 = 64 rows; 4 kv-heads * 8 = 32 rows.
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.k_proj.weight"),
            &[32, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.v_proj.weight"),
            &[32, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.o_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.q_norm.weight"),
            &[8],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.k_norm.weight"),
            &[8],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.input_layernorm.weight"),
            &[64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.post_attention_layernorm.weight"),
            &[64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.gate_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.up_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.down_proj.weight"),
            &[64, 256],
        );
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

/// A fully-known dense model: recognised family, no unconsumed keys beyond
/// metadata, uniform attention.
pub fn known_dense(dir: &Path) -> ArchitectureInventory {
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    let header = serde_json::json!({
        "model.embed_tokens.weight":
            {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 16384]}
    });
    inventory_from(dir, &config, &header)
}
