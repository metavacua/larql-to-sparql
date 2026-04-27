//! FFN tensor naming conventions and helpers for cloning tensors on demand.

use std::collections::HashMap;

use ndarray::ArcArray2;

pub fn detect_ffn_pattern(
    tensors: &HashMap<String, ArcArray2<f32>>,
    component: &str,
) -> String {
    let patterns: &[&str] = match component {
        "gate" => &[
            "model.layers.{}.mlp.gate_proj.weight",
            "layers.{}.ffn.gate.weight",
            "model.layers.{}.feed_forward.gate_proj.weight",
        ],
        "up" => &[
            "model.layers.{}.mlp.up_proj.weight",
            "layers.{}.ffn.up.weight",
            "model.layers.{}.feed_forward.up_proj.weight",
        ],
        "down" => &[
            "model.layers.{}.mlp.down_proj.weight",
            "layers.{}.ffn.down.weight",
            "model.layers.{}.feed_forward.down_proj.weight",
        ],
        _ => &[],
    };

    for pat in patterns {
        let test = pat.replace("{}", "0");
        if tensors.contains_key(&test) {
            return pat.to_string();
        }
    }

    let search = match component {
        "gate" => "gate",
        "up" => "up",
        "down" => "down",
        _ => "",
    };
    for key in tensors.keys() {
        if key.contains(search) && key.contains(".0.") {
            return key.replace(".0.", ".{}.");
        }
    }

    format!("model.layers.{{}}.mlp.{}_proj.weight", component)
}

pub fn ensure_cloned(
    modified: &mut HashMap<String, ArcArray2<f32>>,
    originals: &HashMap<String, ArcArray2<f32>>,
    key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !modified.contains_key(key) {
        let original = originals
            .get(key)
            .ok_or_else(|| format!("tensor not found: {}", key))?;
        modified.insert(key.to_string(), original.to_owned().into());
    }
    Ok(())
}

pub fn decode_f32_b64(b64: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
