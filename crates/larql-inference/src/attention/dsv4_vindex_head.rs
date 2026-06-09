//! DeepSeek V4 head/embedding vindex storage (`dsv4-vindex-extraction`
//! V3, task 4.3).
//!
//! The model-global head tensors the per-layer blobs don't cover:
//!
//! - **token embedding** — `token_embd.weight` (raw, quantized in the
//!   GGUF; `[n_vocab, n_embd]`),
//! - **lm head** — `output.weight` (raw, quantized; `[n_vocab, n_embd]`).
//!   `Option`: tied-embedding models omit it and reuse `token_embd`,
//! - **final norm** — `output_norm.weight` (f32 `[n_embd]`).
//!
//! The global **head mHC** (`hc_head_*`) is carried by the mHC format's
//! `head` field ([`super::dsv4_vindex_mhc`]), not here. Quantized tensors
//! are stored as raw GGUF bytes (no dequant/recompress). Built on the
//! shared [`super::dsv4_vindex_wire`] primitives.

use super::dsv4_gguf_reader::RawExpertTensor;
use super::dsv4_vindex_wire::{put_f32_vec, put_header, put_raw, Cursor};

/// Error decoding a `dsv4_head.bin` blob (the shared vindex wire error).
pub type DsV4HeadPersistError = super::dsv4_vindex_wire::DsV4VindexWireError;

/// 4-byte magic: "D4HD" (DSv4 vindex head).
const MAGIC: [u8; 4] = *b"D4HD";
const VERSION: u16 = 1;

/// The DSv4 global head/embedding weights in extraction form.
#[derive(Clone)]
pub struct DsV4HeadVindex {
    /// `[n_vocab, n_embd]` token embedding (raw, quantized).
    pub token_embd: RawExpertTensor,
    /// `[n_vocab, n_embd]` lm head (raw, quantized) — `None` for
    /// tied-embedding models that reuse `token_embd`.
    pub lm_head: Option<RawExpertTensor>,
    /// `[n_embd]` final RMSNorm gain (f32).
    pub output_norm: Vec<f32>,
}

/// Serialize the global head weights to the `dsv4_head.bin` format.
pub fn serialize_dsv4_head(w: &DsV4HeadVindex) -> Vec<u8> {
    let mut out = Vec::new();
    put_header(&mut out, MAGIC, VERSION);
    put_raw(&mut out, &w.token_embd);
    match &w.lm_head {
        Some(t) => {
            out.push(1);
            put_raw(&mut out, t);
        }
        None => out.push(0),
    }
    put_f32_vec(&mut out, &w.output_norm);
    out
}

/// Deserialize the global head weights from the `dsv4_head.bin` format.
pub fn deserialize_dsv4_head(bytes: &[u8]) -> Result<DsV4HeadVindex, DsV4HeadPersistError> {
    let mut c = Cursor::new(bytes);
    c.read_header(MAGIC, VERSION)?;
    let token_embd = c.read_raw("token_embd")?;
    let lm_head = if c.read_u8("has_lm_head")? != 0 {
        Some(c.read_raw("lm_head")?)
    } else {
        None
    };
    let output_norm = c.read_f32_vec("output_norm")?;
    Ok(DsV4HeadVindex {
        token_embd,
        lm_head,
        output_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::ggml::TYPE_F32;

    fn raw(rows: usize, cols: usize, seed: u8) -> RawExpertTensor {
        RawExpertTensor {
            bytes: (0..rows * cols)
                .flat_map(|i| ((i as f32) * 0.001 + seed as f32).to_le_bytes())
                .collect(),
            tensor_type: TYPE_F32,
            rows,
            cols,
        }
    }

    fn assert_raw_eq(a: &RawExpertTensor, b: &RawExpertTensor) {
        assert_eq!(a.tensor_type, b.tensor_type);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(a.bytes, b.bytes, "raw bytes differ");
    }

    /// Untied head (separate lm_head) round-trips.
    #[test]
    fn untied_head_round_trips() {
        let w = DsV4HeadVindex {
            token_embd: raw(128, 32, 1),
            lm_head: Some(raw(128, 32, 2)),
            output_norm: vec![1.0; 32],
        };
        let got = deserialize_dsv4_head(&serialize_dsv4_head(&w)).unwrap();
        assert_raw_eq(&got.token_embd, &w.token_embd);
        assert_raw_eq(got.lm_head.as_ref().unwrap(), w.lm_head.as_ref().unwrap());
        assert_eq!(got.output_norm, w.output_norm);
    }

    /// Tied-embedding head (no lm_head) round-trips to `None`.
    #[test]
    fn tied_head_round_trips() {
        let w = DsV4HeadVindex {
            token_embd: raw(128, 32, 1),
            lm_head: None,
            output_norm: vec![0.5; 32],
        };
        let got = deserialize_dsv4_head(&serialize_dsv4_head(&w)).unwrap();
        assert!(got.lm_head.is_none());
        assert_raw_eq(&got.token_embd, &w.token_embd);
        assert_eq!(got.output_norm, w.output_norm);
    }

    /// Bad magic / version / truncation → typed error, never a panic.
    #[test]
    fn malformed_blobs_are_typed_errors() {
        let w = DsV4HeadVindex {
            token_embd: raw(64, 16, 1),
            lm_head: Some(raw(64, 16, 2)),
            output_norm: vec![1.0; 16],
        };
        let good = serialize_dsv4_head(&w);

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            deserialize_dsv4_head(&bad_magic),
            Err(DsV4HeadPersistError::BadMagic(_))
        ));

        let mut bad_ver = good.clone();
        bad_ver[4] = 0xFF;
        bad_ver[5] = 0xFF;
        assert!(matches!(
            deserialize_dsv4_head(&bad_ver),
            Err(DsV4HeadPersistError::UnsupportedVersion(0xFFFF))
        ));

        for cut in 0..good.len() {
            assert!(
                deserialize_dsv4_head(&good[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    /// **V3 head round-trip gate** (dsv4-vindex-extraction task 4.3).
    ///
    /// Read `token_embd.weight`, `output.weight`, and `output_norm.weight`
    /// from the real DSv4-Flash GGUF — the embed/lm-head as raw quantized
    /// bytes via the global raw reader, the norm via the head loader —
    /// round-trip through `dsv4_head.bin`, and assert byte/type/shape
    /// equality. The quantized embed/lm-head must not degrade to f32.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_head_round_trips_to_storage() {
        use super::super::dsv4_gguf_reader::read_dsv4_named_raw_tensor;
        use super::super::dsv4_head_storage::load_dsv4_head;
        use super::super::dsv4_storage_build::DsV4Hyperparams;
        use larql_models::loading::gguf::GgufFile;

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let token_embd = read_dsv4_named_raw_tensor(&gguf, "token_embd.weight")
            .expect("read token_embd")
            .expect("token_embd present");
        let lm_head = read_dsv4_named_raw_tensor(&gguf, "output.weight").expect("read output");

        let w = DsV4HeadVindex {
            token_embd,
            lm_head,
            output_norm: head.output_norm.clone(),
        };

        // Embed + lm-head are quantized in the GGUF; passthrough preserves
        // that (no f32 expansion).
        assert_ne!(w.token_embd.tensor_type, TYPE_F32, "token_embd quantized");
        assert_eq!(w.token_embd.rows, head.n_vocab);
        assert_eq!(w.token_embd.cols, hp.n_embd);
        assert_eq!(w.output_norm.len(), hp.n_embd);

        let got = deserialize_dsv4_head(&serialize_dsv4_head(&w)).unwrap();
        assert_raw_eq(&got.token_embd, &w.token_embd);
        assert_eq!(got.lm_head.is_some(), w.lm_head.is_some());
        if let (Some(a), Some(b)) = (&got.lm_head, &w.lm_head) {
            assert_raw_eq(a, b);
        }
        assert_eq!(got.output_norm, w.output_norm);
        eprintln!(
            "head round-trip OK: token_embd {}×{} type {}, lm_head present={}",
            w.token_embd.rows,
            w.token_embd.cols,
            w.token_embd.tensor_type,
            w.lm_head.is_some(),
        );
    }
}
