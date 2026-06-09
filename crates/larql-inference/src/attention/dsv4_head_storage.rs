//! Global DSv4 head storage + GGUF loader.
//!
//! Owns the model-level (non-per-layer) tensors:
//! - `token_embd`     — `(n_vocab, n_embd)`
//! - `output_norm`    — `(n_embd,)` final RMSNorm gain
//! - `output`         — `(n_vocab, n_embd)` lm_head projection
//! - `hc_head_fn`     — `(n_hc, n_hc * n_embd)` output mHC projection
//! - `hc_head_scale`  — scalar (stored as length-1 array)
//! - `hc_head_base`   — `(n_hc,)`
//!
//! Mirrors the per-layer [`super::dsv4_storage::DsV4LayerWeightStorage`]
//! pattern: owns f32 buffers + exposes a borrowed-view method that
//! produces [`super::dsv4_model_forward::DsV4HeadWeights`].

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use ndarray::{Array2, ArrayView2};

use larql_models::detect::ModelError;
use larql_models::loading::gguf::GgufFile;
use larql_models::quant::ggml::{dequantize, is_integer_type, tensor_data_size};

use super::dsv4_model_forward::{DsV4HeadParams, DsV4HeadWeights};
use super::dsv4_storage_build::DsV4Hyperparams;

/// Owned global head weights. Borrow as [`DsV4HeadWeights`] via
/// [`Self::as_weights`]. The token embedding lives here too — it's
/// global (one table for the whole model) and the natural place to
/// own its f32 buffer.
#[derive(Clone)]
pub struct DsV4HeadStorage {
    pub token_embd: Array2<f32>,
    pub output_norm: Vec<f32>,
    pub output_lm_head: Array2<f32>,
    pub output_hc_fn: Array2<f32>,
    pub output_hc_scale: f32,
    pub output_hc_base: Vec<f32>,
    pub n_vocab: usize,
}

impl DsV4HeadStorage {
    /// Borrow as a [`DsV4HeadWeights`]. `token_embd` is exposed via a
    /// separate accessor (it's an input to `dsv4_model_forward`, not
    /// part of the output head).
    pub fn as_weights(&self) -> DsV4HeadWeights<'_> {
        DsV4HeadWeights {
            output_hc_fn: self.output_hc_fn.view(),
            output_hc_scale: self.output_hc_scale,
            output_hc_base: &self.output_hc_base,
            output_norm: &self.output_norm,
            output_lm_head: self.output_lm_head.view(),
        }
    }

    /// Borrow the token embedding for [`dsv4_model_forward`] input.
    pub fn token_embd_view(&self) -> ArrayView2<'_, f32> {
        self.token_embd.view()
    }

    /// Build the corresponding [`DsV4HeadParams`] from a
    /// [`DsV4Hyperparams`] + this storage's own vocab size.
    pub fn params(&self, hp: &DsV4Hyperparams) -> DsV4HeadParams {
        DsV4HeadParams {
            n_embd: hp.n_embd,
            n_hc: self.output_hc_base.len(),
            n_vocab: self.n_vocab,
            norm_eps: hp.norm_eps,
            // Same eps used elsewhere — the model carries no separate
            // head-eps so we reuse the per-layer one.
            hc_eps: hp.norm_eps,
        }
    }
}

/// Load DSv4 global head tensors from a `GgufFile`.
///
/// Requires `hp` (for n_embd) to validate shapes. `n_vocab` and `n_hc`
/// are inferred from the loaded tensor shapes.
pub fn load_dsv4_head(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
) -> Result<DsV4HeadStorage, ModelError> {
    let mut raw: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut raw_dims: HashMap<&'static str, Vec<u64>> = HashMap::new();
    let want: &[&'static str] = &[
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "hc_head_fn",
        "hc_head_scale",
        "hc_head_base",
    ];
    let mut shard_files: HashMap<usize, File> = HashMap::new();
    for info in &gguf.tensor_infos {
        let want_key = want.iter().copied().find(|k| *k == info.name());
        let Some(key) = want_key else { continue };
        if is_integer_type(info.tensor_type()) {
            continue;
        }
        let abs_offset = gguf
            .shard_data_offset(info)
            .checked_add(info.offset())
            .ok_or_else(|| ModelError::Parse(format!("head {key}: offset overflow")))?;
        let n_elements: u64 = info.dims().iter().product();
        let data_size = tensor_data_size(info.tensor_type(), n_elements as usize)?;
        let path = gguf.shard_path(info).to_path_buf();
        let f = shard_files.entry(info.shard_idx()).or_insert_with_key(|_| {
            File::open(&path).unwrap_or_else(|e| panic!("open shard {path:?}: {e}"))
        });
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; data_size];
        f.read_exact(&mut buf)?;
        let vec = dequantize(&buf, info.tensor_type(), n_elements as usize)?;
        raw_dims.insert(key, info.dims().to_vec());
        raw.insert(key, vec);
    }

    // ── Shape + populate the owned struct ──
    let token_embd_vec = take(&mut raw, "token_embd.weight")?;
    let td = need_dims(&raw_dims, "token_embd.weight")?;
    // ggml stores token_embd as [n_embd, n_vocab] → ndarray (n_vocab, n_embd).
    if td.len() != 2 || td[0] as usize != hp.n_embd {
        return Err(ModelError::Parse(format!(
            "token_embd.weight unexpected dims {td:?} (n_embd={})",
            hp.n_embd
        )));
    }
    let n_vocab = td[1] as usize;
    let token_embd = Array2::from_shape_vec((n_vocab, hp.n_embd), token_embd_vec)
        .map_err(|e| ModelError::Parse(format!("token_embd reshape: {e}")))?;

    let output_norm = take(&mut raw, "output_norm.weight")?;
    if output_norm.len() != hp.n_embd {
        return Err(ModelError::Parse(format!(
            "output_norm.weight length {} != n_embd {}",
            output_norm.len(),
            hp.n_embd
        )));
    }

    let output_lm_head_vec = take(&mut raw, "output.weight")?;
    let od = need_dims(&raw_dims, "output.weight")?;
    if od.len() != 2 || od[0] as usize != hp.n_embd || od[1] as usize != n_vocab {
        return Err(ModelError::Parse(format!(
            "output.weight unexpected dims {od:?} (n_embd={}, n_vocab={})",
            hp.n_embd, n_vocab
        )));
    }
    let output_lm_head = Array2::from_shape_vec((n_vocab, hp.n_embd), output_lm_head_vec)
        .map_err(|e| ModelError::Parse(format!("output reshape: {e}")))?;

    let hc_fn_vec = take(&mut raw, "hc_head_fn")?;
    let hd = need_dims(&raw_dims, "hc_head_fn")?;
    // ggml [n_hc * n_embd, n_hc] → ndarray (n_hc, n_hc * n_embd).
    if hd.len() != 2 {
        return Err(ModelError::Parse(format!(
            "hc_head_fn unexpected dims {hd:?}"
        )));
    }
    let hc_dim = hd[0] as usize;
    let n_hc = hd[1] as usize;
    if hc_dim == 0 || hp.n_embd == 0 || hc_dim % hp.n_embd != 0 || hc_dim / hp.n_embd != n_hc {
        return Err(ModelError::Parse(format!(
            "hc_head_fn dims {hd:?} inconsistent: n_embd={}, hc_dim={hc_dim}, n_hc={n_hc}",
            hp.n_embd
        )));
    }
    let output_hc_fn = Array2::from_shape_vec((n_hc, hc_dim), hc_fn_vec)
        .map_err(|e| ModelError::Parse(format!("hc_head_fn reshape: {e}")))?;

    let scale_vec = take(&mut raw, "hc_head_scale")?;
    if scale_vec.len() != 1 {
        return Err(ModelError::Parse(format!(
            "hc_head_scale length {} != 1",
            scale_vec.len()
        )));
    }
    let output_hc_scale = scale_vec[0];

    let output_hc_base = take(&mut raw, "hc_head_base")?;
    if output_hc_base.len() != n_hc {
        return Err(ModelError::Parse(format!(
            "hc_head_base length {} != n_hc {}",
            output_hc_base.len(),
            n_hc
        )));
    }

    Ok(DsV4HeadStorage {
        token_embd,
        output_norm,
        output_lm_head,
        output_hc_fn,
        output_hc_scale,
        output_hc_base,
        n_vocab,
    })
}

fn take(
    raw: &mut HashMap<&'static str, Vec<f32>>,
    key: &'static str,
) -> Result<Vec<f32>, ModelError> {
    raw.remove(key)
        .ok_or_else(|| ModelError::Parse(format!("missing head tensor {key}")))
}

fn need_dims<'a>(
    raw_dims: &'a HashMap<&'static str, Vec<u64>>,
    key: &'static str,
) -> Result<&'a [u64], ModelError> {
    raw_dims
        .get(key)
        .map(|v| v.as_slice())
        .ok_or_else(|| ModelError::Parse(format!("missing head tensor dims for {key}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end real-GGUF load of the global head. Verifies shapes
    /// against the calibration baseline: n_vocab=129280, n_embd=4096,
    /// n_hc=4.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn load_head_from_real_gguf() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("load head");

        assert_eq!(head.n_vocab, 129280);
        assert_eq!(head.token_embd.shape(), &[129280, 4096]);
        assert_eq!(head.output_lm_head.shape(), &[129280, 4096]);
        assert_eq!(head.output_norm.len(), 4096);
        assert_eq!(head.output_hc_base.len(), 4);
        assert_eq!(head.output_hc_fn.shape(), &[4, 4 * 4096]);
        assert!(head.output_hc_scale.is_finite());

        // View → DsV4HeadWeights round-trip and params construction.
        let w = head.as_weights();
        assert_eq!(w.output_norm.len(), 4096);
        assert_eq!(w.output_lm_head.shape(), &[129280, 4096]);
        let p = head.params(&hp);
        assert_eq!(p.n_embd, 4096);
        assert_eq!(p.n_vocab, 129280);
        assert_eq!(p.n_hc, 4);
    }
}
