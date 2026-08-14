//! Side-load the auxiliary weights from the checkpoint's safetensors
//! directory, mirroring the vision tower's convention: the vindex carries
//! backbone weights only; everything else lives beside `config.json`.
//!
//! Two checkpoint properties are asserted here rather than assumed:
//!
//! - **The text input table duplicates the backbone embedding** byte for
//!   byte. The aux load skips it on that basis, so if a future revision
//!   breaks the duplication the load fails instead of serving a wrong
//!   text embedding.
//! - **The pad row of every audio input table is exactly zero**, which is
//!   what makes summing pad channels a no-op. Downstream embedding code
//!   may rely on it, so it is checked where the tensors enter the engine.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use ndarray::Array2;

use crate::architectures::moss_tts_realtime::BACKBONE_KEY_PREFIX;
use crate::config::ModelArchitecture;
use crate::detect::ModelError;
use crate::loading::safetensors::tensor_to_f32;

use super::config::MossTtsRealtimeConfig;
use super::keys::{
    local_embed_table, local_final_norm, local_layer_prefix, local_lm_head, top_embed_table,
    LayerMemberSuffixes,
};
use super::weights::{LocalLayerWeights, LocalTransformerWeights, MossTtsRealtimeAuxWeights};

const SAFETENSORS_EXTENSION: &str = "safetensors";

/// Load the audio-side weights (per-codebook embedding tables + depth
/// transformer) from a directory of safetensors files.
///
/// `arch` supplies the layer-member key spelling and the backbone embed
/// key for the duplication assertion; `config` supplies every count.
pub fn load_moss_tts_aux_from_safetensors(
    dir: impl AsRef<Path>,
    arch: &dyn ModelArchitecture,
    config: MossTtsRealtimeConfig,
) -> Result<MossTtsRealtimeAuxWeights, ModelError> {
    let dir = dir.as_ref();
    let members = LayerMemberSuffixes::from_arch(arch)?;

    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    // (file, key) locations of the two tables the duplication assertion
    // compares raw, so nothing is copied for the check.
    let mut dup_check: HashMap<String, PathBuf> = HashMap::new();

    let backbone_embed_key = format!("{BACKBONE_KEY_PREFIX}{}", arch.embed_key());
    let text_table_key = top_embed_table(0);
    let wanted = wanted_keys(&config, &members);

    for path in safetensors_files(dir)? {
        let mmap = map_file(&path)?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        for (name, view) in st.tensors() {
            if name == backbone_embed_key || name == text_table_key {
                dup_check.insert(name.to_string(), path.clone());
            }
            if !wanted.contains(name.as_str()) {
                continue;
            }
            let shape = view.shape().to_vec();
            let data = tensor_to_f32(&view)?;
            match shape.len() {
                2 => {
                    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                        .map_err(|e| ModelError::Parse(e.to_string()))?;
                    tensors.insert(name.to_string(), arr);
                }
                1 => {
                    vectors.insert(name.to_string(), data);
                }
                other => {
                    return Err(ModelError::Parse(format!(
                        "aux tensor {name} has unexpected rank {other}"
                    )));
                }
            }
        }
    }

    assert_text_table_duplicates_backbone(&dup_check, &backbone_embed_key, &text_table_key)?;

    let audio_embed_tables = take_tables(
        &mut tensors,
        &config,
        arch.config().hidden_size,
        (1..=config.backbone_audio_tables()).map(top_embed_table),
    )?;
    for (index, table) in audio_embed_tables.iter().enumerate() {
        assert_pad_row_zero(table, &config, &top_embed_table(index + 1))?;
    }

    let local_embed_tables = take_tables(
        &mut tensors,
        &config,
        config.local.hidden_size,
        (0..config.local_embed_tables()).map(local_embed_table),
    )?;
    let lm_heads = take_tables(
        &mut tensors,
        &config,
        config.local.hidden_size,
        (0..config.lm_heads()).map(local_lm_head),
    )?;

    let mut layers = Vec::with_capacity(config.local.num_layers);
    for layer in 0..config.local.num_layers {
        layers.push(take_layer(layer, &members, &mut tensors, &mut vectors)?);
    }
    let final_norm = take_vec(&mut vectors, &local_final_norm())?;

    Ok(MossTtsRealtimeAuxWeights {
        audio_embed_tables,
        local: LocalTransformerWeights {
            embed_tables: local_embed_tables,
            layers,
            final_norm,
            lm_heads,
        },
        config,
    })
}

fn wanted_keys(
    config: &MossTtsRealtimeConfig,
    members: &LayerMemberSuffixes,
) -> std::collections::HashSet<String> {
    let mut wanted = std::collections::HashSet::new();
    for table in 1..=config.backbone_audio_tables() {
        wanted.insert(top_embed_table(table));
    }
    for table in 0..config.local_embed_tables() {
        wanted.insert(local_embed_table(table));
    }
    for layer in 0..config.local.num_layers {
        let prefix = local_layer_prefix(layer);
        for suffix in members.all() {
            wanted.insert(format!("{prefix}{suffix}"));
        }
    }
    wanted.insert(local_final_norm());
    for head in 0..config.lm_heads() {
        wanted.insert(local_lm_head(head));
    }
    wanted
}

fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>, ModelError> {
    let entries = std::fs::read_dir(dir).map_err(|e| ModelError::Parse(e.to_string()))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some(SAFETENSORS_EXTENSION))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(ModelError::NoSafetensors(dir.to_path_buf()));
    }
    Ok(files)
}

fn map_file(path: &Path) -> Result<Mmap, ModelError> {
    let file = std::fs::File::open(path).map_err(|e| ModelError::Parse(e.to_string()))?;
    unsafe { Mmap::map(&file) }.map_err(|e| ModelError::Parse(e.to_string()))
}

/// Compare the text input table against the backbone embedding raw, in
/// their on-disk dtype, without materialising either.
fn assert_text_table_duplicates_backbone(
    locations: &HashMap<String, PathBuf>,
    backbone_embed_key: &str,
    text_table_key: &str,
) -> Result<(), ModelError> {
    let backbone_path = locations
        .get(backbone_embed_key)
        .ok_or_else(|| ModelError::MissingTensor(backbone_embed_key.to_string()))?;
    let text_path = locations
        .get(text_table_key)
        .ok_or_else(|| ModelError::MissingTensor(text_table_key.to_string()))?;

    let backbone_mmap = map_file(backbone_path)?;
    let backbone_st = safetensors::SafeTensors::deserialize(&backbone_mmap)
        .map_err(|e| ModelError::Parse(e.to_string()))?;
    let text_mmap;
    let text_st = if text_path == backbone_path {
        None
    } else {
        text_mmap = map_file(text_path)?;
        Some(
            safetensors::SafeTensors::deserialize(&text_mmap)
                .map_err(|e| ModelError::Parse(e.to_string()))?,
        )
    };

    let backbone_view = backbone_st
        .tensor(backbone_embed_key)
        .map_err(|e| ModelError::Parse(e.to_string()))?;
    let text_view = text_st
        .as_ref()
        .unwrap_or(&backbone_st)
        .tensor(text_table_key)
        .map_err(|e| ModelError::Parse(e.to_string()))?;

    if backbone_view.data() != text_view.data() || backbone_view.shape() != text_view.shape() {
        return Err(ModelError::Parse(format!(
            "{text_table_key} is not byte-identical to {backbone_embed_key}; the aux \
             loader skips the text table on the assumption that the backbone copy \
             serves both roles, and this checkpoint violates it"
        )));
    }
    Ok(())
}

/// The pad row must be exactly zero: it is what makes summing pad
/// channels a mathematical no-op. A checkpoint property, not an
/// architectural guarantee — hence checked, not assumed.
fn assert_pad_row_zero(
    table: &Array2<f32>,
    config: &MossTtsRealtimeConfig,
    key: &str,
) -> Result<(), ModelError> {
    let row = config.audio_pad_token;
    if row >= table.nrows() {
        return Err(ModelError::Parse(format!(
            "audio_pad_token {row} out of range for {key} with {} rows",
            table.nrows()
        )));
    }
    if table.row(row).iter().any(|&v| v != 0.0) {
        return Err(ModelError::Parse(format!(
            "{key} pad row {row} is not all-zero; the summed-embedding no-op \
             property does not hold for this checkpoint"
        )));
    }
    Ok(())
}

fn take_tables(
    tensors: &mut HashMap<String, Array2<f32>>,
    config: &MossTtsRealtimeConfig,
    hidden_size: usize,
    keys: impl Iterator<Item = String>,
) -> Result<Vec<Array2<f32>>, ModelError> {
    keys.map(|key| {
        let table = tensors
            .remove(&key)
            .ok_or_else(|| ModelError::MissingTensor(key.clone()))?;
        let expected = (config.audio_vocab_size, hidden_size);
        if table.dim() != expected {
            return Err(ModelError::Parse(format!(
                "{key} has shape {:?}, expected {expected:?}",
                table.dim()
            )));
        }
        Ok(table)
    })
    .collect()
}

fn take_layer(
    layer: usize,
    members: &LayerMemberSuffixes,
    tensors: &mut HashMap<String, Array2<f32>>,
    vectors: &mut HashMap<String, Vec<f32>>,
) -> Result<LocalLayerWeights, ModelError> {
    let prefix = local_layer_prefix(layer);
    let mut tensor = |suffix: &str| -> Result<Array2<f32>, ModelError> {
        let key = format!("{prefix}{suffix}");
        tensors.remove(&key).ok_or(ModelError::MissingTensor(key))
    };
    let mut vector = |suffix: &str| -> Result<Vec<f32>, ModelError> {
        let key = format!("{prefix}{suffix}");
        vectors.remove(&key).ok_or(ModelError::MissingTensor(key))
    };
    Ok(LocalLayerWeights {
        q_proj: tensor(&members.q_proj)?,
        k_proj: tensor(&members.k_proj)?,
        v_proj: tensor(&members.v_proj)?,
        o_proj: tensor(&members.o_proj)?,
        q_norm: vector(&members.q_norm)?,
        k_norm: vector(&members.k_norm)?,
        gate_proj: tensor(&members.gate_proj)?,
        up_proj: tensor(&members.up_proj)?,
        down_proj: tensor(&members.down_proj)?,
        input_layernorm: vector(&members.input_layernorm)?,
        post_attention_layernorm: vector(&members.post_attention_layernorm)?,
    })
}

fn take_vec(vectors: &mut HashMap<String, Vec<f32>>, key: &str) -> Result<Vec<f32>, ModelError> {
    vectors
        .remove(key)
        .ok_or_else(|| ModelError::MissingTensor(key.to_string()))
}
