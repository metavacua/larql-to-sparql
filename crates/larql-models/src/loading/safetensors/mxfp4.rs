//! MXFP4 packed-expert detection and per-expert dequantisation.
//!
//! Split out of `safetensors.rs` (ROADMAP H5b). GPT-OSS ships its MoE
//! experts as packed `*_blocks` / `*_scales` pairs rather than one tensor
//! per expert; this module recognises that layout and expands it into the
//! per-expert Mixtral-style keys the rest of the loader expects.

use std::collections::HashMap;

use ndarray::Array2;

use crate::detect::ModelError;

use super::{
    normalize_key, tensor_to_f32, BLOCK_SPARSE_ROUTER_WEIGHT, MIXTRAL_DOWN_PROJ, MIXTRAL_GATE_PROJ,
    MIXTRAL_UP_PROJ, MXFP4_ROUTER_WEIGHT,
};

// Packed-expert tensor-name suffixes. These name the MXFP4 layout, so
// they live with the code that understands it rather than in the shard
// walker that merely passes names through.
pub(super) const MXFP4_GATE_UP_BLOCKS_SUFFIX: &str = ".gate_up_proj_blocks";
pub(super) const MXFP4_BLOCKS_SUFFIX: &str = "_blocks";
pub(super) const MXFP4_SCALES_SUFFIX: &str = "_scales";
pub(super) const MXFP4_GATE_UP_BLOCKS: &str = "gate_up_proj_blocks";
pub(super) const MXFP4_EXPERTS_GATE_UP_BLOCKS: &str = "experts.gate_up_proj_blocks";
pub(super) const MXFP4_DOWN_BLOCKS: &str = "down_proj_blocks";
pub(super) const MXFP4_DOWN_SCALES: &str = "down_proj_scales";

/// Load GPT-OSS MXFP4 packed expert tensors from a safetensors file into the
/// weights map, using per-expert Mixtral-style key names.
///
/// GPT-OSS stores experts as:
///   layers.{L}.mlp.experts.gate_up_proj_blocks: [experts, 2*hidden, groups, 16] U8
///   layers.{L}.mlp.experts.gate_up_proj_scales: [experts, 2*hidden, groups] U8
///   layers.{L}.mlp.experts.down_proj_blocks: [experts, hidden, groups, 16] U8
///   layers.{L}.mlp.experts.down_proj_scales: [experts, hidden, groups] U8
///
/// Dequantization and gate/up splitting are handled by `quant::mxfp4`.
/// Output keys follow Mixtral conventions:
///   layers.{L}.block_sparse_moe.experts.{E}.w1.weight (gate)
///   layers.{L}.block_sparse_moe.experts.{E}.w3.weight (up)
///   layers.{L}.block_sparse_moe.experts.{E}.w2.weight (down)
pub(super) fn load_mxfp4_expert_tensors(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    skip_key: &impl Fn(&str) -> bool,
    tensors: &mut HashMap<String, crate::WeightArray>,
) -> Result<(), ModelError> {
    for name in tensor_names {
        if !name.ends_with(MXFP4_GATE_UP_BLOCKS_SUFFIX) {
            continue;
        }

        let scales_name = name.replace(MXFP4_BLOCKS_SUFFIX, MXFP4_SCALES_SUFFIX);
        let down_blocks_name = name.replace(MXFP4_GATE_UP_BLOCKS, MXFP4_DOWN_BLOCKS);
        let down_scales_name = name.replace(MXFP4_GATE_UP_BLOCKS, MXFP4_DOWN_SCALES);

        let blocks_view = st
            .tensor(name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 blocks: {e}")))?;
        let scales_view = st
            .tensor(&scales_name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 scales: {e}")))?;

        let shape = blocks_view.shape();
        if shape.len() != 4 {
            continue;
        }

        let num_experts = shape[0];
        let out_features = shape[1]; // = 2 * hidden (gate + up fused)
        let groups = shape[2];
        let in_features = groups * 32;
        let half = out_features / 2;

        let base_key = normalize_key(name, prefixes);
        let layer_prefix = base_key.split(".mlp.").next().unwrap_or("");
        let should_load_gate_up = (0..num_experts).any(|e| {
            !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_GATE_PROJ))
                || !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_UP_PROJ))
        });

        // Dequantize and split fused gate_up → separate gate (w1) and up (w3).
        if should_load_gate_up {
            let (gate_experts, up_experts) = crate::quant::mxfp4::split_gate_up_experts(
                blocks_view.data(),
                scales_view.data(),
                num_experts,
                out_features,
                groups,
            )?;

            for (e, (gate_data, up_data)) in gate_experts.into_iter().zip(up_experts).enumerate() {
                let gate_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_GATE_PROJ);
                if !skip_key(&gate_key) {
                    tensors.insert(
                        gate_key,
                        Array2::from_shape_vec((half, in_features), gate_data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
                let up_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_UP_PROJ);
                if !skip_key(&up_key) {
                    tensors.insert(
                        up_key,
                        Array2::from_shape_vec((half, in_features), up_data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
            }
        }

        // Dequantize down projection.
        if let (Ok(db), Ok(ds)) = (st.tensor(&down_blocks_name), st.tensor(&down_scales_name)) {
            let down_shape = db.shape();
            if down_shape.len() == 4 {
                let down_out = down_shape[1];
                let down_groups = down_shape[2];
                let down_in = down_groups * 32;
                let should_load_down = (0..num_experts)
                    .any(|e| !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_DOWN_PROJ)));
                if should_load_down {
                    let down_experts = crate::quant::mxfp4::dequantize_all_experts(
                        db.data(),
                        ds.data(),
                        num_experts,
                        down_out,
                        down_groups,
                    )?;
                    for (e, data) in down_experts.into_iter().enumerate() {
                        let down_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_DOWN_PROJ);
                        if !skip_key(&down_key) {
                            tensors.insert(
                                down_key,
                                Array2::from_shape_vec((down_out, down_in), data)
                                    .map_err(|e| ModelError::Parse(e.to_string()))?
                                    .into_shared(),
                            );
                        }
                    }
                }
            }
        }

        // Remap router: mlp.router.weight → block_sparse_moe.gate.weight
        let router_name = name.replace(MXFP4_EXPERTS_GATE_UP_BLOCKS, MXFP4_ROUTER_WEIGHT);
        if let Ok(router_view) = st.tensor(&router_name) {
            if let Ok(data) = tensor_to_f32(&router_view) {
                let s = router_view.shape();
                if s.len() == 2 {
                    let router_key = format!("{layer_prefix}.{BLOCK_SPARSE_ROUTER_WEIGHT}");
                    if !skip_key(&router_key) {
                        tensors.insert(
                            router_key,
                            Array2::from_shape_vec((s[0], s[1]), data)
                                .map_err(|e| ModelError::Parse(e.to_string()))?
                                .into_shared(),
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

/// Key for one dequantised MXFP4 expert projection.
///
/// `layer_prefix` here comes from splitting a packed tensor name at `.mlp.`,
/// so it has no trailing separator; the shared builder expects one, the way
/// [`ModelArchitecture::layer_prefix`](crate::ModelArchitecture::layer_prefix)
/// supplies it. The convention itself lives in
/// [`crate::tensor_keys::mxfp4_dequantised`] so the architecture that
/// advertises these keys and the loader that writes them cannot drift.
fn mxfp4_expert_key(layer_prefix: &str, expert_id: usize, projection: &str) -> String {
    crate::tensor_keys::mxfp4_dequantised::projection(
        &format!("{layer_prefix}."),
        expert_id,
        projection,
    )
}

/// Per-expert MXFP4 dequantization (DeepSeek-V4 family).
///
/// DeepSeek-V4 stores expert weights one (.weight, .scale) pair per
/// (expert, projection) — `layers.X.ffn.experts.E.w1.weight` (I8 packed FP4) +
/// `layers.X.ffn.experts.E.w1.scale` (F8_E8M0 scales), ditto w2/w3. This is
/// distinct from GPT-OSS's fused `experts.gate_up_proj_blocks` layout that
/// `load_mxfp4_expert_tensors` handles.
///
/// Detects the format by scanning for `*.experts.<digit>.w[123].weight` tensors
/// with `I8` dtype. For each match, looks up the companion `.scale` (`F8_E8M0`)
/// and dequantizes via `quant::mxfp4::dequantize_expert`.
///
/// Returns the set of tensor names that were consumed (both `.weight` and
/// `.scale`) so the main loading loop can skip them.
pub(super) fn dequantize_per_expert_mxfp4(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    tensors: &mut HashMap<String, crate::WeightArray>,
) -> Result<std::collections::HashSet<String>, ModelError> {
    use std::collections::HashSet;
    let mut consumed: HashSet<String> = HashSet::new();

    // Match V4-style per-expert weights: any tensor name containing
    // ".experts.<int>.w<1|2|3>.weight" — broad enough to catch both the
    // full `model.layers.X.ffn.experts.E.wY.weight` (HF default) and any
    // shortened variant (`layers.X.ffn.experts.E.wY.weight`).
    let is_v4_expert_weight = |name: &str| -> bool {
        if !name.ends_with(".w1.weight")
            && !name.ends_with(".w2.weight")
            && !name.ends_with(".w3.weight")
        {
            return false;
        }
        // Must have ".experts.<digit>" before the .wN.weight suffix
        if let Some(idx) = name.rfind(".experts.") {
            let after = &name[idx + ".experts.".len()..];
            if let Some(dot) = after.find('.') {
                return after[..dot].chars().all(|c| c.is_ascii_digit());
            }
        }
        false
    };

    for name in tensor_names {
        if !is_v4_expert_weight(name) {
            continue;
        }

        let weight_view = match st.tensor(name) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // V4 packed FP4 weights are stored as I8 (signed) per the safetensors header.
        if weight_view.dtype() != safetensors::Dtype::I8 {
            continue;
        }

        let scale_name = name.replacen(".weight", ".scale", 1);
        let scale_view = match st.tensor(&scale_name) {
            Ok(v) => v,
            Err(_) => continue, // No scale companion → not MXFP4, leave to main loop.
        };
        if scale_view.dtype() != safetensors::Dtype::F8_E8M0 {
            continue;
        }

        // Shape sanity. weight: (out_features, packed_in/2). scale: (out_features, groups).
        let w_shape = weight_view.shape();
        let s_shape = scale_view.shape();
        if w_shape.len() != 2 || s_shape.len() != 2 {
            continue;
        }
        if w_shape[0] != s_shape[0] {
            continue;
        }

        let out_features = w_shape[0];
        let groups = s_shape[1];
        let in_features = groups * 32;

        // Assert layout consistency: weight cols × 2 (nibbles per byte) == groups × 32.
        if w_shape[1] * 2 != in_features {
            continue;
        }

        let unpacked = crate::quant::mxfp4::dequantize_expert(
            weight_view.data(),
            scale_view.data(),
            out_features,
            groups,
        )?;

        let key = normalize_key(name, prefixes);
        let arr = Array2::from_shape_vec((out_features, in_features), unpacked)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        tensors.insert(key, arr.into_shared());

        consumed.insert(name.clone());
        consumed.insert(scale_name);
    }

    Ok(consumed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{tensor::TensorView, Dtype};

    /// Build an in-memory safetensors buffer. Returned as bytes because
    /// `SafeTensors::deserialize` borrows them, so the caller must own the
    /// buffer for as long as the parsed view lives.
    fn st_bytes(tensors: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>) -> Vec<u8> {
        let owned: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = tensors
            .into_iter()
            .map(|(n, d, s, b)| (n.to_string(), d, s, b))
            .collect();
        let views: Vec<(String, TensorView)> = owned
            .iter()
            .map(|(n, d, s, b)| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap()))
            .collect();
        safetensors::serialize(views, None).unwrap()
    }

    fn never_skip() -> impl Fn(&str) -> bool {
        |_: &str| false
    }

    #[test]
    fn expert_key_follows_the_shared_mixtral_convention() {
        // The doc contract: `layer_prefix` arrives with no trailing dot and
        // the helper supplies it. A drift here silently produces keys the
        // architecture never advertises.
        let k = mxfp4_expert_key("layers.3", 7, MIXTRAL_GATE_PROJ);
        assert_eq!(
            k,
            crate::tensor_keys::mxfp4_dequantised::projection("layers.3.", 7, MIXTRAL_GATE_PROJ)
        );
        assert!(
            k.contains("layers.3."),
            "prefix must gain its separator: {k}"
        );
        assert!(!k.contains("layers.3.."), "and must not double it: {k}");
    }

    // ── dequantize_per_expert_mxfp4: the rejection ladder ────────────────
    //
    // Each guard returns "not mine, leave it to the main loop" rather than
    // erroring, so the observable is an empty `consumed` set plus an
    // untouched tensor map. Tested one guard at a time.

    #[test]
    fn v4_ignores_names_without_an_experts_index() {
        // `.w1.weight` alone is not enough — the `.experts.<digit>.` segment
        // is what distinguishes a V4 expert from an ordinary projection.
        let buf = st_bytes(vec![(
            "layers.0.ffn.w1.weight",
            Dtype::I8,
            vec![2, 16],
            vec![0u8; 32],
        )]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let names = vec!["layers.0.ffn.w1.weight".to_string()];
        let mut tensors = HashMap::new();
        let consumed = dequantize_per_expert_mxfp4(&st, &names, &[], &mut tensors).unwrap();
        assert!(consumed.is_empty());
        assert!(tensors.is_empty());
    }

    #[test]
    fn v4_ignores_a_non_numeric_expert_segment() {
        let name = "layers.0.ffn.experts.shared.w1.weight";
        let buf = st_bytes(vec![(name, Dtype::I8, vec![2, 16], vec![0u8; 32])]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[name.to_string()], &[], &mut tensors).unwrap();
        assert!(consumed.is_empty(), "`experts.shared` is not an index");
    }

    #[test]
    fn v4_ignores_a_correctly_named_tensor_of_the_wrong_dtype() {
        // Right name, wrong dtype: an F32 expert is a dense weight the main
        // loop must still see, not an MXFP4 one to unpack here.
        let name = "layers.0.ffn.experts.0.w1.weight";
        let buf = st_bytes(vec![(name, Dtype::F32, vec![2, 4], vec![0u8; 32])]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[name.to_string()], &[], &mut tensors).unwrap();
        assert!(consumed.is_empty());
        assert!(tensors.is_empty(), "must be left for the main loop");
    }

    #[test]
    fn v4_ignores_an_i8_expert_with_no_scale_companion() {
        let name = "layers.0.ffn.experts.0.w1.weight";
        let buf = st_bytes(vec![(name, Dtype::I8, vec![2, 16], vec![0u8; 32])]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[name.to_string()], &[], &mut tensors).unwrap();
        assert!(consumed.is_empty(), "no `.scale` → not MXFP4");
    }

    #[test]
    fn v4_ignores_a_scale_of_the_wrong_dtype() {
        let w = "layers.0.ffn.experts.0.w1.weight";
        let s = "layers.0.ffn.experts.0.w1.scale";
        let buf = st_bytes(vec![
            (w, Dtype::I8, vec![2, 16], vec![0u8; 32]),
            (s, Dtype::U8, vec![2, 1], vec![127u8; 2]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[w.to_string()], &[], &mut tensors).unwrap();
        assert!(consumed.is_empty(), "scale must be F8_E8M0");
    }

    #[test]
    fn v4_ignores_a_row_count_mismatch_between_weight_and_scale() {
        let w = "layers.0.ffn.experts.0.w1.weight";
        let s = "layers.0.ffn.experts.0.w1.scale";
        let buf = st_bytes(vec![
            (w, Dtype::I8, vec![2, 16], vec![0u8; 32]),
            // 3 rows of scale against 2 rows of weight.
            (s, Dtype::F8_E8M0, vec![3, 1], vec![127u8; 3]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[w.to_string()], &[], &mut tensors).unwrap();
        assert!(consumed.is_empty(), "out_features must agree");
    }

    #[test]
    fn v4_ignores_a_packed_width_that_contradicts_the_group_count() {
        // The layout assertion: weight cols × 2 nibbles must equal
        // groups × 32. Here 16 cols → 32 values, but 2 groups claim 64.
        let w = "layers.0.ffn.experts.0.w1.weight";
        let s = "layers.0.ffn.experts.0.w1.scale";
        let buf = st_bytes(vec![
            (w, Dtype::I8, vec![2, 16], vec![0u8; 32]),
            (s, Dtype::F8_E8M0, vec![2, 2], vec![127u8; 4]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[w.to_string()], &[], &mut tensors).unwrap();
        assert!(consumed.is_empty(), "packed width must match groups*32");
    }

    #[test]
    fn v4_dequantises_a_well_formed_expert_and_reports_both_names_consumed() {
        // out_features = 2, one group of 32 → 16 packed bytes per row.
        let w = "layers.0.ffn.experts.0.w1.weight";
        let s = "layers.0.ffn.experts.0.w1.scale";
        let buf = st_bytes(vec![
            (w, Dtype::I8, vec![2, 16], vec![0x21u8; 32]),
            // 127 → 2^0 = 1.0, so the dequantised values are the raw table.
            (s, Dtype::F8_E8M0, vec![2, 1], vec![127u8; 2]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let consumed =
            dequantize_per_expert_mxfp4(&st, &[w.to_string()], &[], &mut tensors).unwrap();

        assert_eq!(consumed.len(), 2, "both weight and scale are consumed");
        assert!(consumed.contains(w) && consumed.contains(s));
        let arr = tensors.get(w).expect("dequantised expert must be inserted");
        assert_eq!(arr.shape(), &[2, 32], "groups*32 columns");
    }

    // ── load_mxfp4_expert_tensors (GPT-OSS fused layout) ─────────────────

    #[test]
    fn fused_loader_ignores_tensors_without_the_gate_up_blocks_suffix() {
        let name = "layers.0.mlp.experts.down_proj_blocks";
        let buf = st_bytes(vec![(name, Dtype::U8, vec![1, 2, 1, 16], vec![0u8; 32])]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        load_mxfp4_expert_tensors(&st, &[name.to_string()], &[], &never_skip(), &mut tensors)
            .unwrap();
        assert!(
            tensors.is_empty(),
            "only *.gate_up_proj_blocks is a trigger"
        );
    }

    #[test]
    fn fused_loader_errors_when_the_scales_companion_is_absent() {
        // Unlike the V4 path's silent skips, a fused blocks tensor with no
        // scales is malformed rather than foreign — the pair is written
        // together by the exporter.
        let name = "layers.0.mlp.experts.gate_up_proj_blocks";
        let buf = st_bytes(vec![(name, Dtype::U8, vec![1, 2, 1, 16], vec![0u8; 32])]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let err =
            load_mxfp4_expert_tensors(&st, &[name.to_string()], &[], &never_skip(), &mut tensors)
                .unwrap_err();
        assert!(
            format!("{err}").contains("MXFP4 scales"),
            "error should name the missing side: {err}"
        );
    }

    #[test]
    fn fused_loader_skips_a_blocks_tensor_that_is_not_four_dimensional() {
        // A 3-D blocks tensor cannot carry [experts, out, groups, 16];
        // treated as not-this-layout rather than an error.
        let b = "layers.0.mlp.experts.gate_up_proj_blocks";
        let s = "layers.0.mlp.experts.gate_up_proj_scales";
        let buf = st_bytes(vec![
            (b, Dtype::U8, vec![2, 1, 16], vec![0u8; 32]),
            (s, Dtype::U8, vec![2, 1], vec![127u8; 2]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        load_mxfp4_expert_tensors(&st, &[b.to_string()], &[], &never_skip(), &mut tensors).unwrap();
        assert!(tensors.is_empty());
    }

    #[test]
    fn fused_loader_honours_skip_key_and_does_no_work_when_everything_is_filtered() {
        // `should_load_gate_up` is the guard that makes a fully-skipped
        // layer cheap: with every key filtered, nothing is dequantised.
        let b = "layers.0.mlp.experts.gate_up_proj_blocks";
        let s = "layers.0.mlp.experts.gate_up_proj_scales";
        let buf = st_bytes(vec![
            (b, Dtype::U8, vec![1, 2, 1, 16], vec![0x21u8; 32]),
            (s, Dtype::U8, vec![1, 2, 1], vec![127u8; 2]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        let skip_all = |_: &str| true;
        load_mxfp4_expert_tensors(&st, &[b.to_string()], &[], &skip_all, &mut tensors).unwrap();
        assert!(tensors.is_empty(), "skip_key must suppress every insert");
    }

    #[test]
    fn fused_loader_splits_gate_and_up_into_separate_mixtral_keys() {
        // out_features = 2 → half = 1: one gate row (w1) and one up row (w3).
        let b = "layers.0.mlp.experts.gate_up_proj_blocks";
        let s = "layers.0.mlp.experts.gate_up_proj_scales";
        let buf = st_bytes(vec![
            (b, Dtype::U8, vec![1, 2, 1, 16], vec![0x21u8; 32]),
            (s, Dtype::U8, vec![1, 2, 1], vec![127u8; 2]),
        ]);
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        let mut tensors = HashMap::new();
        load_mxfp4_expert_tensors(&st, &[b.to_string()], &[], &never_skip(), &mut tensors).unwrap();

        let gate = mxfp4_expert_key("layers.0", 0, MIXTRAL_GATE_PROJ);
        let up = mxfp4_expert_key("layers.0", 0, MIXTRAL_UP_PROJ);
        assert!(tensors.contains_key(&gate), "missing gate key {gate}");
        assert!(tensors.contains_key(&up), "missing up key {up}");
        // in_features = groups * 32.
        assert_eq!(tensors[&gate].shape(), &[1, 32]);
        assert_eq!(tensors[&up].shape(), &[1, 32]);
    }
}
