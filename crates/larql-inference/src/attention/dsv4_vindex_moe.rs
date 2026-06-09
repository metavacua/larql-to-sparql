//! DeepSeek V4 MoE-FFN vindex storage (`dsv4-vindex-extraction` V3,
//! task 4.2).
//!
//! The DSv4 FFN block, present on every layer:
//!
//! - **routed experts** — `ffn_{gate,up,down}_exps` (raw Q4_K, 3D packed
//!   `[n_expert, *, *]`),
//! - **shared expert** — `ffn_{gate,up}_shexp` (Q4_K) + `ffn_down_shexp`
//!   (Q6_K), all raw passthrough,
//! - **router** — `ffn_gate_inp` (f32 `[n_expert, n_embd]`) + the optional
//!   routing bias `exp_probs_b` (f32 `[n_expert]`),
//! - **hash routing** — `ffn_gate_tid2eid` (i32 `[n_vocab, n_expert_used]`),
//!   present only on the first 3 (hash-routed) layers,
//! - `ffn_norm` (f32 `[n_embd]`).
//!
//! Quantized tensors are stored as raw GGUF bytes (no dequant/recompress,
//! preserving Q4_K *and* the shared-expert down's Q6_K). Built on the
//! shared [`super::dsv4_vindex_wire`] primitives.

use super::dsv4_gguf_reader::RawExpertTensor;
use super::dsv4_vindex_wire::{
    put_f32_vec, put_header, put_opt_f32_vec, put_opt_i32_vec, put_raw, Cursor,
};

/// Error decoding a `dsv4_moe.bin` blob (the shared vindex wire error).
pub type DsV4MoePersistError = super::dsv4_vindex_wire::DsV4VindexWireError;

/// 4-byte magic: "D4MO" (DSv4 vindex MoE).
const MAGIC: [u8; 4] = *b"D4MO";
const VERSION: u16 = 1;

/// One DSv4 layer's MoE-FFN weights in extraction form.
#[derive(Clone)]
pub struct DsV4MoeWeights {
    /// `[n_expert, n_ff_exp, n_embd]` routed gate experts (Q4_K).
    pub gate_exps: RawExpertTensor,
    /// `[n_expert, n_ff_exp, n_embd]` routed up experts (Q4_K).
    pub up_exps: RawExpertTensor,
    /// `[n_expert, n_embd, n_ff_exp]` routed down experts (Q4_K).
    pub down_exps: RawExpertTensor,
    /// `[n_ff_exp*n_expert_shared, n_embd]` shared gate (Q4_K).
    pub gate_shexp: RawExpertTensor,
    /// `[n_ff_exp*n_expert_shared, n_embd]` shared up (Q4_K).
    pub up_shexp: RawExpertTensor,
    /// `[n_embd, n_ff_exp*n_expert_shared]` shared down (Q6_K).
    pub down_shexp: RawExpertTensor,
    /// `[n_expert, n_embd]` router gate (f32).
    pub gate_inp: Vec<f32>,
    /// `[n_embd]` FFN input RMSNorm gain (f32).
    pub ffn_norm: Vec<f32>,
    /// `[n_expert]` routing bias (f32) — absent on hash-routed layers.
    pub exp_probs_b: Option<Vec<f32>>,
    /// `[n_vocab*n_expert_used]` token→expert hash table (i32) — present
    /// only on the first 3 (hash-routed) layers.
    pub gate_tid2eid: Option<Vec<i32>>,
}

/// Serialize one layer's MoE-FFN weights to the `dsv4_moe.bin` format.
pub fn serialize_dsv4_moe(w: &DsV4MoeWeights) -> Vec<u8> {
    let mut out = Vec::new();
    put_header(&mut out, MAGIC, VERSION);
    for t in [
        &w.gate_exps,
        &w.up_exps,
        &w.down_exps,
        &w.gate_shexp,
        &w.up_shexp,
        &w.down_shexp,
    ] {
        put_raw(&mut out, t);
    }
    put_f32_vec(&mut out, &w.gate_inp);
    put_f32_vec(&mut out, &w.ffn_norm);
    put_opt_f32_vec(&mut out, &w.exp_probs_b);
    put_opt_i32_vec(&mut out, &w.gate_tid2eid);
    out
}

/// Deserialize one layer's MoE-FFN weights from the `dsv4_moe.bin` format.
pub fn deserialize_dsv4_moe(bytes: &[u8]) -> Result<DsV4MoeWeights, DsV4MoePersistError> {
    let mut c = Cursor::new(bytes);
    c.read_header(MAGIC, VERSION)?;
    Ok(DsV4MoeWeights {
        gate_exps: c.read_raw("gate_exps")?,
        up_exps: c.read_raw("up_exps")?,
        down_exps: c.read_raw("down_exps")?,
        gate_shexp: c.read_raw("gate_shexp")?,
        up_shexp: c.read_raw("up_shexp")?,
        down_shexp: c.read_raw("down_shexp")?,
        gate_inp: c.read_f32_vec("gate_inp")?,
        ffn_norm: c.read_f32_vec("ffn_norm")?,
        exp_probs_b: c.read_opt_f32_vec("exp_probs_b")?,
        gate_tid2eid: c.read_opt_i32_vec("gate_tid2eid")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::ggml::TYPE_F32;

    fn raw(rows: usize, cols: usize, seed: u8) -> RawExpertTensor {
        RawExpertTensor {
            bytes: (0..rows * cols)
                .flat_map(|i| ((i as f32) * 0.01 + seed as f32).to_le_bytes())
                .collect(),
            tensor_type: TYPE_F32,
            rows,
            cols,
        }
    }

    fn sample(hash: bool, bias: bool) -> DsV4MoeWeights {
        DsV4MoeWeights {
            gate_exps: raw(8 * 4, 16, 1),
            up_exps: raw(8 * 4, 16, 2),
            down_exps: raw(8 * 16, 4, 3),
            gate_shexp: raw(4, 16, 4),
            up_shexp: raw(4, 16, 5),
            down_shexp: raw(16, 4, 6),
            gate_inp: vec![0.5; 8 * 16],
            ffn_norm: vec![1.0; 16],
            exp_probs_b: bias.then(|| vec![0.1; 8]),
            gate_tid2eid: hash.then(|| (0..32 * 6).map(|i| i as i32 % 8).collect()),
        }
    }

    fn assert_raw_eq(a: &RawExpertTensor, b: &RawExpertTensor) {
        assert_eq!(a.tensor_type, b.tensor_type);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(a.bytes, b.bytes, "raw bytes differ");
    }

    fn assert_moe_eq(a: &DsV4MoeWeights, b: &DsV4MoeWeights) {
        assert_raw_eq(&a.gate_exps, &b.gate_exps);
        assert_raw_eq(&a.up_exps, &b.up_exps);
        assert_raw_eq(&a.down_exps, &b.down_exps);
        assert_raw_eq(&a.gate_shexp, &b.gate_shexp);
        assert_raw_eq(&a.up_shexp, &b.up_shexp);
        assert_raw_eq(&a.down_shexp, &b.down_shexp);
        assert_eq!(a.gate_inp, b.gate_inp);
        assert_eq!(a.ffn_norm, b.ffn_norm);
        assert_eq!(a.exp_probs_b, b.exp_probs_b);
        assert_eq!(a.gate_tid2eid, b.gate_tid2eid);
    }

    /// A gated MoE layer (routing bias, no hash table) round-trips.
    #[test]
    fn gated_layer_round_trips() {
        let w = sample(false, true);
        let got = deserialize_dsv4_moe(&serialize_dsv4_moe(&w)).unwrap();
        assert_moe_eq(&got, &w);
        assert!(got.gate_tid2eid.is_none());
        assert!(got.exp_probs_b.is_some());
    }

    /// A hash-routed MoE layer (token→expert table) round-trips, hash
    /// values preserved exactly.
    #[test]
    fn hash_layer_round_trips() {
        let w = sample(true, false);
        let got = deserialize_dsv4_moe(&serialize_dsv4_moe(&w)).unwrap();
        assert_moe_eq(&got, &w);
        assert_eq!(got.gate_tid2eid, w.gate_tid2eid);
        assert!(got.exp_probs_b.is_none());
    }

    /// Bad magic / version / truncation → typed error, never a panic.
    #[test]
    fn malformed_blobs_are_typed_errors() {
        let good = serialize_dsv4_moe(&sample(true, true));

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            deserialize_dsv4_moe(&bad_magic),
            Err(DsV4MoePersistError::BadMagic(_))
        ));

        let mut bad_ver = good.clone();
        bad_ver[4] = 0xFF;
        bad_ver[5] = 0xFF;
        assert!(matches!(
            deserialize_dsv4_moe(&bad_ver),
            Err(DsV4MoePersistError::UnsupportedVersion(0xFFFF))
        ));

        for cut in 0..good.len() {
            assert!(
                deserialize_dsv4_moe(&good[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    /// **V3 MoE round-trip gate** (dsv4-vindex-extraction task 4.2).
    ///
    /// Scan the real DSv4-Flash GGUF for a hash-routed layer (first 3) and
    /// a gated layer, read their MoE weights via the *same* readers the
    /// resident loader feeds, round-trip through `dsv4_moe.bin`, and assert
    /// byte/type/shape equality on every expert + shared-expert tensor
    /// (incl. the shared down's Q6_K), the router gate + bias, the FFN
    /// norm, and the hash table — verifying the table + bias survive.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_moe_round_trips_to_storage() {
        use super::super::dsv4_gguf_reader::{
            read_dsv4_layer_int_tensors_from_gguf, read_dsv4_layer_raw_expert_tensors_from_gguf,
            read_dsv4_layer_tensors_from_gguf,
        };
        use super::super::dsv4_layer_variants::detect_layer_variant;
        use super::super::dsv4_storage_build::DsV4Hyperparams;
        use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;
        use larql_models::loading::gguf::GgufFile;
        use std::collections::HashMap;

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");

        const N_LAYER: usize = 43;
        let mut hash_layer = None;
        let mut gated_layer = None;
        for l in 0..N_LAYER {
            let v = detect_layer_variant(&gguf, l, hp.head_dim);
            if v.uses_hash_routing && hash_layer.is_none() {
                hash_layer = Some(l);
            } else if !v.uses_hash_routing && gated_layer.is_none() {
                gated_layer = Some(l);
            }
            if hash_layer.is_some() && gated_layer.is_some() {
                break;
            }
        }
        let hash_layer = hash_layer.expect("a hash-routed layer exists");
        let gated_layer = gated_layer.expect("a gated layer exists");
        eprintln!("hash layer {hash_layer}, gated layer {gated_layer}");

        let expert_kinds = [
            DsV4TensorKind::FfnGateExps,
            DsV4TensorKind::FfnUpExps,
            DsV4TensorKind::FfnDownExps,
            DsV4TensorKind::FfnGateShexp,
            DsV4TensorKind::FfnUpShexp,
            DsV4TensorKind::FfnDownShexp,
        ];
        let take = |raw: &mut HashMap<DsV4TensorKind, RawExpertTensor>, k| {
            raw.remove(&k).unwrap_or_else(|| panic!("missing {k:?}"))
        };

        for layer in [hash_layer, gated_layer] {
            let mut raw = read_dsv4_layer_raw_expert_tensors_from_gguf(&gguf, layer, &expert_kinds)
                .expect("read expert raw");
            let f32s = read_dsv4_layer_tensors_from_gguf(&gguf, layer).expect("read f32");
            let ints = read_dsv4_layer_int_tensors_from_gguf(&gguf, layer).expect("read int");

            let w = DsV4MoeWeights {
                gate_exps: take(&mut raw, DsV4TensorKind::FfnGateExps),
                up_exps: take(&mut raw, DsV4TensorKind::FfnUpExps),
                down_exps: take(&mut raw, DsV4TensorKind::FfnDownExps),
                gate_shexp: take(&mut raw, DsV4TensorKind::FfnGateShexp),
                up_shexp: take(&mut raw, DsV4TensorKind::FfnUpShexp),
                down_shexp: take(&mut raw, DsV4TensorKind::FfnDownShexp),
                gate_inp: f32s[&DsV4TensorKind::FfnGateInp].clone(),
                ffn_norm: f32s[&DsV4TensorKind::FfnNorm].clone(),
                exp_probs_b: f32s.get(&DsV4TensorKind::FfnExpProbsB).cloned(),
                gate_tid2eid: ints.get(&DsV4TensorKind::FfnGateTid2Eid).cloned(),
            };

            // The shared-expert down is Q6_K, the rest Q4_K — neither must
            // degrade to f32 in the passthrough.
            assert_ne!(w.down_shexp.tensor_type, TYPE_F32, "down_shexp quantized");
            assert_ne!(w.gate_exps.tensor_type, TYPE_F32, "gate_exps quantized");

            let got = deserialize_dsv4_moe(&serialize_dsv4_moe(&w)).unwrap();
            assert_moe_eq(&got, &w);
            eprintln!(
                "layer {layer}: hash={} bias={} down_shexp_type={}",
                w.gate_tid2eid.is_some(),
                w.exp_probs_b.is_some(),
                w.down_shexp.tensor_type,
            );
        }
    }
}
