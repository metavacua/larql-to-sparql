//! Read DSv4 per-layer tensors from a GGUF file.
//!
//! Layered on:
//! - [`larql_models::loading::gguf::GgufFile`] — open + parse headers
//! - [`larql_models::quant::ggml::dequantize`] — convert raw bytes → f32
//! - [`larql_models::architectures::deepseek_v4_tensors`] — name schema
//!
//! Produces the `HashMap<DsV4TensorKind, Vec<f32>>` that
//! [`super::dsv4_storage_build::build_layer_storage`] consumes.
//!
//! Two entry points:
//! - [`rekey_dsv4_layer_tensors`] — pure: re-key a name→vec map by
//!   `DsV4TensorKind`. Synthetic testable.
//! - [`read_dsv4_layer_tensors_from_gguf`] — open-shard, byte-read,
//!   dequantize, then re-key.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use larql_models::architectures::deepseek_v4_tensors::{
    all_kinds, tensor_name_of, try_parse_name, DsV4TensorKind,
};
use larql_models::detect::ModelError;
use larql_models::loading::gguf::GgufFile;
use larql_models::quant::ggml::{dequantize, is_integer_type, tensor_data_size};

/// Re-key a generic `(raw_gguf_name → Vec<f32>)` map by
/// `DsV4TensorKind` for a given `layer_index`.
///
/// Pure-logic helper: takes the output of any byte-reading step and
/// drops everything that doesn't belong to the requested layer (or
/// can't be parsed by the DSv4 schema).
pub fn rekey_dsv4_layer_tensors(
    raw_map: HashMap<String, Vec<f32>>,
    layer_index: usize,
) -> HashMap<DsV4TensorKind, Vec<f32>> {
    let mut out = HashMap::new();
    for (name, vec) in raw_map {
        let Some((kind, layer)) = try_parse_name(&name) else {
            continue;
        };
        match (kind.is_per_layer(), layer) {
            (true, Some(l)) if l == layer_index => {
                out.insert(kind, vec);
            }
            (false, None) => {
                // Global tensor — DSv4 forward loop pulls these
                // separately, but caller may want to receive them. Skip
                // here to keep the per-layer map clean.
            }
            _ => {}
        }
    }
    out
}

/// Read per-layer DSv4 tensors from a `GgufFile`, dequantize each to
/// `f32`, and re-key by `DsV4TensorKind`.
///
/// Returns a `HashMap` with one entry per DSv4 tensor kind that's
/// actually present in the GGUF for layer `layer_index`. Missing
/// tensors are simply absent from the map — the downstream
/// [`super::dsv4_storage_build::build_layer_storage`] will reject any
/// that the variant requires.
pub fn read_dsv4_layer_tensors_from_gguf(
    gguf: &GgufFile,
    layer_index: usize,
) -> Result<HashMap<DsV4TensorKind, Vec<f32>>, ModelError> {
    read_dsv4_layer_tensors_from_gguf_excluding(gguf, layer_index, &[])
}

/// Like [`read_dsv4_layer_tensors_from_gguf`] but skips the tensor
/// kinds in `exclude` — they are neither read nor dequantized.
///
/// The resident-quant loader (`dsv4-quant-residency`) uses this to read
/// the small f32 tensors while **excluding** the routed-MoE experts
/// (`ffn_{gate,up,down}_exps`), which it reads raw via
/// [`read_dsv4_layer_raw_expert_tensors_from_gguf`] instead — so the
/// ~26 GB/layer f32 expansion of those tensors is never allocated.
pub fn read_dsv4_layer_tensors_from_gguf_excluding(
    gguf: &GgufFile,
    layer_index: usize,
    exclude: &[DsV4TensorKind],
) -> Result<HashMap<DsV4TensorKind, Vec<f32>>, ModelError> {
    // Build the per-layer name → kind index up front.
    let mut want: HashMap<String, DsV4TensorKind> = HashMap::new();
    for &kind in all_kinds() {
        if !kind.is_per_layer() || exclude.contains(&kind) {
            continue;
        }
        want.insert(tensor_name_of(kind, layer_index), kind);
    }

    // For each matching tensor in the GGUF, read its bytes + dequantize.
    // Integer-typed tensors (e.g. `ffn_gate_tid2eid` is I32) have no
    // meaningful f32 dequantization — they're routing tables consumed
    // elsewhere. Skip them here; future code can read them separately.
    let mut out: HashMap<DsV4TensorKind, Vec<f32>> = HashMap::new();
    let mut shard_files: HashMap<usize, File> = HashMap::new();
    for info in &gguf.tensor_infos {
        let Some(&kind) = want.get(info.name()) else {
            continue;
        };
        if is_integer_type(info.tensor_type()) {
            continue;
        }
        let shard_idx = info.shard_idx();
        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "DSv4 layer {layer_index}: {} offset overflow",
                    info.name()
                ))
            })?;
        let n_elements: u64 = info.dims().iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;

        let path: PathBuf = gguf.shard_path(info).to_path_buf();
        let f = shard_files.entry(shard_idx).or_insert_with_key(|_| {
            File::open(&path)
                .unwrap_or_else(|e| panic!("DSv4 layer {layer_index}: open shard {path:?}: {e}"))
        });
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; data_size];
        f.read_exact(&mut buf)?;

        let vec = dequantize(&buf, info.tensor_type(), n_elements as usize)?;
        out.insert(kind, vec);
    }
    Ok(out)
}

/// A routed-MoE expert tensor read straight from the GGUF as its raw
/// quantized bytes, plus the metadata `QuantTensor::from_raw` needs.
///
/// `rows`/`cols` are the flat 2D `[n_expert * out_dim, in_dim]` shape
/// (GGUF stores the tensor 3D fastest-first as `[in_dim, out_dim,
/// n_expert]`). This is exactly the packing `QuantTensor::expert_slice`
/// expects, verified against the real GGUF by
/// [`tests::real_gguf_audit_expert_slice_packing`].
#[derive(Clone)]
pub struct RawExpertTensor {
    pub bytes: Vec<u8>,
    pub tensor_type: u32,
    pub rows: usize,
    pub cols: usize,
}

/// Read the routed-MoE expert tensors (`ffn_gate_exps`, `ffn_up_exps`,
/// `ffn_down_exps`) for `layer_index` as **raw quantized bytes** —
/// without dequantizing.
///
/// This is the resident-quant (`dsv4-quant-residency`) counterpart to
/// [`read_dsv4_layer_tensors_from_gguf`]: the bytes are handed to
/// `QuantTensor::from_raw`, so the ~26 GB/layer f32 expansion of these
/// tensors is never allocated. Only 3D expert tensors are returned; any
/// kind absent for the layer is simply not in the map (e.g. a dense
/// layer with no routed experts).
pub fn read_dsv4_layer_raw_expert_tensors_from_gguf(
    gguf: &GgufFile,
    layer_index: usize,
    want_kinds: &[DsV4TensorKind],
) -> Result<HashMap<DsV4TensorKind, RawExpertTensor>, ModelError> {
    let want: HashMap<String, DsV4TensorKind> = want_kinds
        .iter()
        .map(|&k| (tensor_name_of(k, layer_index), k))
        .collect();

    let mut out: HashMap<DsV4TensorKind, RawExpertTensor> = HashMap::new();
    let mut shard_files: HashMap<usize, File> = HashMap::new();
    for info in &gguf.tensor_infos {
        let Some(&kind) = want.get(info.name()) else {
            continue;
        };
        let dims = info.dims();
        // GGUF stores weights fastest-first. The flat `from_raw` shape is
        // `[rows, cols]` row-major:
        // - 3D expert tensor `[in, out, n_expert]` → `[n_expert*out, in]`
        //   (per-expert slice via `QuantTensor::expert_slice`); also used
        //   for the grouped o-proj A `[group_dim, o_lora, n_groups]`.
        // - 2D linear weight `[in, out]` → `[out, in]` (matches the f32
        //   `take_2d(out, in)` layout: `x @ W^T`).
        let (rows, cols) = match dims.len() {
            3 => {
                let in_dim = dims[0] as usize;
                let out_dim = dims[1] as usize;
                let n_expert = dims[2] as usize;
                (n_expert * out_dim, in_dim)
            }
            2 => {
                let in_dim = dims[0] as usize;
                let out_dim = dims[1] as usize;
                (out_dim, in_dim)
            }
            other => {
                return Err(ModelError::Parse(format!(
                    "DSv4 layer {layer_index}: {} expected a 2D/3D tensor, got {other}D dims {dims:?}",
                    info.name(),
                )));
            }
        };

        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "DSv4 layer {layer_index}: {} offset overflow",
                    info.name()
                ))
            })?;
        let n_elements: u64 = dims.iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;

        let path: PathBuf = gguf.shard_path(info).to_path_buf();
        let f = shard_files.entry(info.shard_idx()).or_insert_with_key(|_| {
            File::open(&path)
                .unwrap_or_else(|e| panic!("DSv4 layer {layer_index}: open shard {path:?}: {e}"))
        });
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut bytes = vec![0u8; data_size];
        f.read_exact(&mut bytes)?;

        out.insert(
            kind,
            RawExpertTensor {
                bytes,
                tensor_type: info.tensor_type(),
                rows,
                cols,
            },
        );
    }
    Ok(out)
}

/// Read a single **global** (non-layer) tensor by exact GGUF name as raw
/// quantized bytes (e.g. `token_embd.weight`, `output.weight`), returning
/// `None` if the model doesn't carry it (e.g. tied embeddings omit
/// `output.weight`).
///
/// Like [`read_dsv4_layer_raw_expert_tensors_from_gguf`] but for the
/// global head tensors the per-layer (`blk.N.*`) name schema doesn't
/// cover. 2D `[in, out]` → flat `[out, in]` (matches the f32 head
/// loader's `(n_vocab, n_embd)` reshape).
pub fn read_dsv4_named_raw_tensor(
    gguf: &GgufFile,
    name: &str,
) -> Result<Option<RawExpertTensor>, ModelError> {
    let Some(info) = gguf.tensor_infos.iter().find(|i| i.name() == name) else {
        return Ok(None);
    };
    let dims = info.dims();
    let (rows, cols) = match dims.len() {
        2 => (dims[1] as usize, dims[0] as usize),
        1 => (dims[0] as usize, 1usize),
        other => {
            return Err(ModelError::Parse(format!(
                "DSv4 global {name}: expected 1D/2D tensor, got {other}D dims {dims:?}"
            )));
        }
    };
    let abs_offset = gguf
        .shard_data_offset(info)
        .checked_add(info.offset())
        .ok_or_else(|| ModelError::Parse(format!("DSv4 global {name}: offset overflow")))?;
    let n_elements: u64 = dims.iter().product();
    let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;
    let path = gguf.shard_path(info).to_path_buf();
    let mut f = File::open(&path)?;
    f.seek(SeekFrom::Start(abs_offset))?;
    let mut bytes = vec![0u8; data_size];
    f.read_exact(&mut bytes)?;
    Ok(Some(RawExpertTensor {
        bytes,
        tensor_type: info.tensor_type(),
        rows,
        cols,
    }))
}

/// Read per-layer DSv4 **integer** tensors (e.g. `ffn_gate_tid2eid`),
/// keyed by [`DsV4TensorKind`]. Returns `Vec<i32>` per kind so the
/// downstream loader can shape it into the right routing table.
///
/// Skips non-integer types — callers should pair this with
/// [`read_dsv4_layer_tensors_from_gguf`] for the f32 weights.
pub fn read_dsv4_layer_int_tensors_from_gguf(
    gguf: &GgufFile,
    layer_index: usize,
) -> Result<HashMap<DsV4TensorKind, Vec<i32>>, ModelError> {
    let mut want: HashMap<String, DsV4TensorKind> = HashMap::new();
    for &kind in all_kinds() {
        if !kind.is_per_layer() {
            continue;
        }
        want.insert(tensor_name_of(kind, layer_index), kind);
    }

    let mut out: HashMap<DsV4TensorKind, Vec<i32>> = HashMap::new();
    let mut shard_files: HashMap<usize, File> = HashMap::new();
    for info in &gguf.tensor_infos {
        let Some(&kind) = want.get(info.name()) else {
            continue;
        };
        if !is_integer_type(info.tensor_type()) {
            continue;
        }
        let shard_idx = info.shard_idx();
        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "DSv4 layer {layer_index}: {} offset overflow",
                    info.name()
                ))
            })?;
        let n_elements: u64 = info.dims().iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;
        // DSv4 integer tensors observed so far are all I32 (type 26).
        // tensor_data_size returns the right byte count for any int width.
        let path: PathBuf = gguf.shard_path(info).to_path_buf();
        let f = shard_files.entry(shard_idx).or_insert_with_key(|_| {
            File::open(&path)
                .unwrap_or_else(|e| panic!("DSv4 layer {layer_index}: open shard {path:?}: {e}"))
        });
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; data_size];
        f.read_exact(&mut buf)?;
        // Decode bytes as i32 little-endian. Only I32 (type 26) handled
        // here — other integer widths are rare in DSv4 and can extend
        // this later if needed.
        if info.tensor_type() != larql_models::quant::ggml::TYPE_I32 {
            continue;
        }
        let elems = n_elements as usize;
        if buf.len() < elems * 4 {
            return Err(ModelError::Parse(format!(
                "DSv4 layer {layer_index}: {} I32 buffer too short",
                info.name()
            )));
        }
        let mut vec = Vec::with_capacity(elems);
        for chunk in buf[..elems * 4].chunks_exact(4) {
            vec.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out.insert(kind, vec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rekey_picks_only_requested_layer() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.3.attn_norm.weight".to_string(), vec![1.0; 4]);
        raw.insert("blk.3.attn_q_a.weight".to_string(), vec![2.0; 8]);
        // Different layer — should be dropped.
        raw.insert("blk.4.attn_norm.weight".to_string(), vec![9.0; 4]);
        // Global tensor — dropped by the per-layer view.
        raw.insert("token_embd.weight".to_string(), vec![0.0; 16]);
        // Non-DSv4 — dropped.
        raw.insert("blk.3.foo_bar.weight".to_string(), vec![5.0; 4]);

        let out = rekey_dsv4_layer_tensors(raw, 3);
        assert_eq!(out.len(), 2);
        assert_eq!(out.get(&DsV4TensorKind::AttnNorm).unwrap()[0], 1.0);
        assert_eq!(out.get(&DsV4TensorKind::AttnQA).unwrap()[0], 2.0);
    }

    #[test]
    fn rekey_handles_indexer_dotted_names() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.5.indexer.attn_q_b.weight".to_string(), vec![1.0; 8]);
        raw.insert("blk.5.indexer.proj.weight".to_string(), vec![2.0; 4]);
        let out = rekey_dsv4_layer_tensors(raw, 5);
        assert_eq!(out.len(), 2);
        assert!(out.contains_key(&DsV4TensorKind::IndexerAttnQB));
        assert!(out.contains_key(&DsV4TensorKind::IndexerProj));
    }

    /// `exp_probs_b` carries no suffix in the real DSv4 GGUF (the
    /// schema was speculative in initial draft — calibrated 2026-05-23).
    #[test]
    fn rekey_handles_no_suffix_exp_probs_b() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.2.exp_probs_b".to_string(), vec![1.0, 2.0, 3.0]);
        let out = rekey_dsv4_layer_tensors(raw, 2);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out.get(&DsV4TensorKind::FfnExpProbsB).unwrap(),
            &vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn rekey_layer_mismatch_drops_everything() {
        let mut raw: HashMap<String, Vec<f32>> = HashMap::new();
        raw.insert("blk.0.attn_norm.weight".to_string(), vec![1.0; 4]);
        raw.insert("blk.1.attn_norm.weight".to_string(), vec![2.0; 4]);
        let out = rekey_dsv4_layer_tensors(raw, 7);
        assert!(out.is_empty());
    }

    #[test]
    fn rekey_empty_input_yields_empty_output() {
        let out = rekey_dsv4_layer_tensors(HashMap::new(), 0);
        assert!(out.is_empty());
    }

    /// Real-GGUF: layer 0's `ffn_gate_tid2eid` is I32 with shape
    /// `[6, 129280]` = (n_expert_used, n_vocab). Verify the int reader
    /// loads it correctly.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_loads_ffn_gate_tid2eid() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let map = read_dsv4_layer_int_tensors_from_gguf(&gguf, 0).expect("load int tensors");
        let table = map
            .get(&DsV4TensorKind::FfnGateTid2Eid)
            .expect("ffn_gate_tid2eid present on layer 0");
        // n_expert_used=6, n_vocab=129280 → 775680 entries.
        assert_eq!(table.len(), 6 * 129280);
        // Indices should be in [0, n_expert) = [0, 256).
        let max_idx = *table.iter().max().unwrap();
        let min_idx = *table.iter().min().unwrap();
        assert!(
            min_idx >= 0 && max_idx < 256,
            "tid2eid out of expected expert range: min={min_idx} max={max_idx}"
        );

        // Layers 3+ should NOT have tid2eid (verified by the empty map).
        let layer_3_map =
            read_dsv4_layer_int_tensors_from_gguf(&gguf, 3).expect("load int tensors for layer 3");
        assert!(
            !layer_3_map.contains_key(&DsV4TensorKind::FfnGateTid2Eid),
            "layer 3 should not have ffn_gate_tid2eid (hash routing is first 3 layers)"
        );
    }

    /// P0 audit (dsv4-quant-residency change): enumerate every GGUF
    /// tensor type DSv4-Flash actually uses and assert that every
    /// matmul weight is in a format the `QuantTensor` lazy-quant path
    /// can run resident — Q4_K/Q5_K/Q6_K/Q8_0 (with F32 as the
    /// small-tensor fallback). Integer tensors (`*.weight` routing
    /// tables like `ffn_gate_tid2eid`) are read by the separate int
    /// reader and are excluded from the matmul invariant.
    ///
    /// This is the load-bearing precondition for P1/P2: if any large
    /// weight uses a format `QuantTensor::matvec` cannot dispatch, the
    /// dual-storage port must route it to the f32 fallback. The test
    /// prints a full type histogram so any future GGUF re-quant is
    /// visible, and fails loudly if a new unsupported format appears.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_audit_tensor_types() {
        use larql_models::quant::ggml::{
            is_integer_type, type_name, TYPE_F32, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0,
        };

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");

        // Histogram: tensor_type id → (count, example name).
        let mut histogram: HashMap<u32, (usize, String)> = HashMap::new();
        // Matmul weights whose format the lazy-quant path cannot run.
        let mut unsupported: Vec<(String, u32)> = Vec::new();

        // Formats `QuantTensor::matvec` dispatches (lazy.rs). F32 is the
        // small-tensor fallback (norms, biases); the K-quants are the
        // large matmul weights we want resident.
        let supported = [TYPE_F32, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0];

        for info in &gguf.tensor_infos {
            let ty = info.tensor_type();
            let entry = histogram
                .entry(ty)
                .or_insert_with(|| (0, info.name().to_string()));
            entry.0 += 1;

            // Integer routing tables (ffn_gate_tid2eid, exp_probs_b ids)
            // are not matmul weights — the int reader handles them.
            if is_integer_type(ty) {
                continue;
            }
            if !supported.contains(&ty) {
                unsupported.push((info.name().to_string(), ty));
            }
        }

        // Print the histogram sorted by type id for a stable audit log.
        let mut types: Vec<u32> = histogram.keys().copied().collect();
        types.sort_unstable();
        eprintln!(
            "DSv4-Flash GGUF tensor-type histogram ({} tensors):",
            gguf.tensor_infos.len()
        );
        for ty in types {
            let (count, example) = &histogram[&ty];
            eprintln!(
                "  {:>10} (id {:>3}): {:>5} tensors  e.g. {example}",
                type_name(ty),
                ty,
                count,
            );
        }

        if !unsupported.is_empty() {
            eprintln!(
                "{} weight tensor(s) use a format the lazy-quant path cannot run:",
                unsupported.len()
            );
            for (name, ty) in &unsupported {
                eprintln!("  {name} → {} (id {ty})", type_name(*ty));
            }
        }

        assert!(
            unsupported.is_empty(),
            "DSv4-Flash uses {} matmul-weight format(s) outside Q4_K/Q5_K/Q6_K/Q8_0/F32 \
             — these need the f32 fallback in dual storage (see first entry: {:?})",
            unsupported.len(),
            unsupported.first(),
        );
    }

    /// P0 audit task 1.2 (dsv4-quant-residency): confirm
    /// `QuantTensor::from_raw` accepts DSv4's MoE expert tensor bytes
    /// and that `expert_slice` packing matches the GGUF expert layout.
    ///
    /// DSv4 packs `ffn_*_exps` 3D as GGUF dims `[in_dim, out_dim,
    /// n_expert]` (fastest-first), which `from_raw` treats as a flat
    /// 2D `[n_expert * out_dim, in_dim]`. `expert_slice(e, n_expert)`
    /// must then yield the per-expert `[out_dim, in_dim]` matrix, and a
    /// `matvec` against it must run through the lazy-quant kernel
    /// (no full dequant). This is the precondition for P2 task 3.4
    /// (MoE dispatch via `expert_slice` instead of per-expert dequant).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_audit_expert_slice_packing() {
        use larql_models::quant::lazy::QuantTensor;
        use ndarray::Array1;

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");

        // Find a routed-MoE gate-experts tensor (any layer that has one).
        let info = gguf
            .tensor_infos
            .iter()
            .find(|i| i.name().ends_with(".ffn_gate_exps.weight"))
            .expect("DSv4 GGUF has an ffn_gate_exps.weight tensor");
        let dims = info.dims().to_vec();
        eprintln!(
            "{}: dims {:?} type {}",
            info.name(),
            dims,
            larql_models::quant::ggml::type_name(info.tensor_type())
        );
        // GGUF stores fastest-first: [in_dim, out_dim, n_expert].
        assert_eq!(dims.len(), 3, "ffn_gate_exps should be 3D (got {dims:?})");
        let in_dim = dims[0] as usize;
        let out_dim = dims[1] as usize;
        let n_expert = dims[2] as usize;
        // Flat 2D for from_raw: rows = n_expert * out_dim, cols = in_dim.
        let rows = n_expert * out_dim;
        let cols = in_dim;

        // Read the raw quantized bytes (no dequant).
        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .expect("offset");
        let n_elements: u64 = info.dims().iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize).expect("size");
        let mut f = File::open(gguf.shard_path(info)).expect("open shard");
        f.seek(SeekFrom::Start(abs_offset)).expect("seek");
        let mut buf = vec![0u8; data_size];
        f.read_exact(&mut buf).expect("read tensor bytes");

        // Build resident QuantTensor directly from GGUF bytes.
        let qt = QuantTensor::from_raw(buf, info.tensor_type(), rows, cols)
            .expect("from_raw accepts DSv4 expert tensor");
        assert_eq!(qt.shape(), [rows, cols]);

        // expert_slice must give the per-expert [out_dim, in_dim] matrix.
        for e in [0usize, 1, n_expert / 2, n_expert - 1] {
            let slice = qt.expert_slice(e, n_expert).expect("expert_slice");
            assert_eq!(
                slice.shape(),
                [out_dim, in_dim],
                "expert {e} slice shape mismatch"
            );
        }

        // The lazy-quant matvec must run against a sliced expert without
        // materializing the full f32 weight.
        let expert0 = qt.expert_slice(0, n_expert).expect("expert_slice 0");
        let x = Array1::<f32>::from_elem(in_dim, 0.01);
        let y = expert0.matvec(&x).expect("matvec on quant expert slice");
        assert_eq!(y.len(), out_dim);
        assert!(
            y.iter().all(|v| v.is_finite()),
            "matvec produced non-finite output"
        );
        eprintln!(
            "expert_slice OK: n_expert={n_expert} out_dim={out_dim} in_dim={in_dim}; \
             matvec y[0]={:.5}",
            y[0]
        );
    }

    /// P1 (dsv4-quant-residency): prove the resident-quant footprint of
    /// one layer's routed MoE experts is the Q4_K size, not the f32
    /// expansion. This is the memory win that lets the model fit in RAM
    /// (~161 GB resident vs ~1.1 TB f32) and removes the per-token
    /// streaming reload. Builds the three expert `QuantTensor`s
    /// (gate/up/down) directly from GGUF bytes via `from_raw` — the
    /// `FfnStorage::{gate,up,down}_exps_quant` dual-storage fields hold
    /// exactly these — and asserts the total resident bytes are a small
    /// fraction of the f32 size.
    ///
    /// Backs spec scenarios "Loader keeps Q4_K bytes quantized" and
    /// "Resident footprint fits the quantized size".
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_resident_expert_footprint() {
        use larql_models::quant::lazy::QuantTensor;

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");

        // Read raw bytes for one named tensor (no dequant).
        let read_raw = |suffix: &str| -> (Vec<u8>, u32, Vec<u64>) {
            let info = gguf
                .tensor_infos
                .iter()
                .find(|i| i.name().ends_with(suffix))
                .unwrap_or_else(|| panic!("no tensor ending in {suffix}"));
            let abs_offset = gguf
                .shard_data_offset(info)
                .checked_add(info.offset())
                .expect("offset");
            let n_elements: u64 = info.dims().iter().product();
            let data_size =
                tensor_data_size(info.tensor_type(), n_elements as usize).expect("size");
            let mut f = File::open(gguf.shard_path(info)).expect("open shard");
            f.seek(SeekFrom::Start(abs_offset)).expect("seek");
            let mut buf = vec![0u8; data_size];
            f.read_exact(&mut buf).expect("read");
            (buf, info.tensor_type(), info.dims().to_vec())
        };

        let mut quant_bytes = 0usize;
        let mut f32_bytes = 0usize;
        for suffix in [
            ".ffn_gate_exps.weight",
            ".ffn_up_exps.weight",
            ".ffn_down_exps.weight",
        ] {
            let (buf, ty, dims) = read_raw(suffix);
            assert_eq!(dims.len(), 3, "{suffix} should be 3D");
            // GGUF fastest-first [in_dim, out_dim, n_expert] → flat
            // [n_expert*out_dim, in_dim] for from_raw.
            let in_dim = dims[0] as usize;
            let out_dim = dims[1] as usize;
            let n_expert = dims[2] as usize;
            let n_elem = in_dim * out_dim * n_expert;
            quant_bytes += buf.len();
            f32_bytes += n_elem * std::mem::size_of::<f32>();
            // The resident dual-storage field holds exactly this tensor.
            let qt = QuantTensor::from_raw(buf, ty, n_expert * out_dim, in_dim)
                .unwrap_or_else(|e| panic!("from_raw {suffix}: {e:?}"));
            assert_eq!(qt.shape(), [n_expert * out_dim, in_dim]);
        }

        let ratio = quant_bytes as f64 / f32_bytes as f64;
        eprintln!(
            "resident MoE experts (1 layer): quant {:.2} GB vs f32 {:.2} GB ({:.1}× smaller, ratio {:.3})",
            quant_bytes as f64 / 1e9,
            f32_bytes as f64 / 1e9,
            1.0 / ratio,
            ratio,
        );
        // Q4_K is ~4.5 bits/weight vs f32's 32 → resident should be well
        // under a quarter of the f32 expansion. Generous bound (0.25) so
        // a future re-quant (Q5_K/Q6_K mix) still passes while still
        // proving the win is real.
        assert!(
            ratio < 0.25,
            "resident MoE footprint {ratio:.3} not materially smaller than f32 — \
             expected the Q4_K bytes (~0.14×), got {:.2} GB quant vs {:.2} GB f32",
            quant_bytes as f64 / 1e9,
            f32_bytes as f64 / 1e9,
        );
    }

    /// P1 (dsv4-quant-residency): `read_dsv4_layer_raw_expert_tensors_from_gguf`
    /// returns the three routed-expert tensors as raw quantized bytes
    /// with the `from_raw`-ready `[n_expert*out_dim, in_dim]` shape, and
    /// those bytes round-trip through `QuantTensor::from_raw`. This is
    /// the resident loader's input — the f32 expansion is never built.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_raw_expert_reader_round_trips_to_quant_tensor() {
        use larql_models::quant::ggml::type_name;
        use larql_models::quant::lazy::QuantTensor;

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");

        // Pick a routed-MoE layer (hash routing is layers 0-2; layer 4
        // is a regular routed-expert layer).
        let want = [
            DsV4TensorKind::FfnGateExps,
            DsV4TensorKind::FfnUpExps,
            DsV4TensorKind::FfnDownExps,
        ];
        let raw = read_dsv4_layer_raw_expert_tensors_from_gguf(&gguf, 4, &want)
            .expect("read raw expert tensors");

        for kind in want {
            let t = raw
                .get(&kind)
                .unwrap_or_else(|| panic!("layer 4 should have {kind}"));
            // Byte length must match the declared quant shape exactly.
            let expected = tensor_data_size(t.tensor_type, t.rows * t.cols).expect("size");
            assert_eq!(
                t.bytes.len(),
                expected,
                "{kind}: raw byte length {} != expected {expected} for {} {}×{}",
                t.bytes.len(),
                type_name(t.tensor_type),
                t.rows,
                t.cols,
            );
            // And the bytes build a QuantTensor of that shape (no dequant).
            let qt = QuantTensor::from_raw(t.bytes.clone(), t.tensor_type, t.rows, t.cols)
                .unwrap_or_else(|e| panic!("from_raw {kind}: {e:?}"));
            assert_eq!(qt.shape(), [t.rows, t.cols]);
        }

        // A dense / hash-routed-only layer query still succeeds (the map
        // just contains whatever expert tensors exist).
        let _ = read_dsv4_layer_raw_expert_tensors_from_gguf(&gguf, 0, &want)
            .expect("layer 0 raw expert read");
    }
}
