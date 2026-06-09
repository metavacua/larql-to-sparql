//! DeepSeek V4 mHC-bookend vindex storage (`dsv4-vindex-extraction` V3,
//! task 4.1).
//!
//! mHC ("multi-head compression" residual bookends) wraps each compute
//! block with a learned `fn`/`scale`/`base` triple. DSv4 has three:
//!
//! - per-layer **attn** bookend (`hc_attn_{fn,scale,base}`) — every layer,
//! - per-layer **ffn** bookend (`hc_ffn_{fn,scale,base}`) — every layer,
//! - the global **head** bookend (`hc_head_{fn,scale,base}`) — once.
//!
//! All three are f32 in the GGUF, so the blob stores plain f32 vectors
//! (no quantized passthrough); `hc_fn`'s 2D shape `[(2+n_hc)*n_hc,
//! n_hc*n_embd]` and `hc_base`'s `[(2+n_hc)*n_hc]` are recoverable from
//! the hyperparameters on read, so only the flat values are stored.
//!
//! The three bookends are independently optional: per-layer blobs set
//! `attn` + `ffn` (head `None`); the model-level head blob sets `head`
//! only. Built on the shared [`super::dsv4_vindex_wire`] primitives.

use super::dsv4_vindex_wire::{put_f32_vec, put_header, Cursor};

/// Error decoding a `dsv4_mhc.bin` blob (the shared vindex wire error).
pub type DsV4MhcPersistError = super::dsv4_vindex_wire::DsV4VindexWireError;

/// 4-byte magic: "D4MH" (DSv4 vindex mHC).
const MAGIC: [u8; 4] = *b"D4MH";
const VERSION: u16 = 1;

/// One mHC bookend: the gate-fn matrix (flattened), the scale array
/// (`[3]` per-layer, `[1]` for the head), and the base vector. All f32.
#[derive(Clone, PartialEq, Debug)]
pub struct MhcBookend {
    /// Flattened `hc_fn` (`[(2+n_hc)*n_hc, n_hc*n_embd]` row-major).
    pub fn_weight: Vec<f32>,
    /// `hc_scale` — 3 elements per-layer, 1 for the head.
    pub scale: Vec<f32>,
    /// `hc_base` — `[(2+n_hc)*n_hc]`.
    pub base: Vec<f32>,
}

/// One DSv4 mHC blob: a per-layer blob sets `attn` + `ffn` (head `None`);
/// the model-level head blob sets `head` only.
#[derive(Clone, Default)]
pub struct DsV4MhcWeights {
    pub attn: Option<MhcBookend>,
    pub ffn: Option<MhcBookend>,
    pub head: Option<MhcBookend>,
}

/// Serialize one mHC blob to the `dsv4_mhc.bin` wire format.
pub fn serialize_dsv4_mhc(w: &DsV4MhcWeights) -> Vec<u8> {
    let mut out = Vec::new();
    put_header(&mut out, MAGIC, VERSION);
    for b in [&w.attn, &w.ffn, &w.head] {
        match b {
            Some(b) => {
                out.push(1);
                put_bookend(&mut out, b);
            }
            None => out.push(0),
        }
    }
    out
}

/// Deserialize an mHC blob from the `dsv4_mhc.bin` wire format.
pub fn deserialize_dsv4_mhc(bytes: &[u8]) -> Result<DsV4MhcWeights, DsV4MhcPersistError> {
    let mut c = Cursor::new(bytes);
    c.read_header(MAGIC, VERSION)?;
    Ok(DsV4MhcWeights {
        attn: read_opt_bookend(&mut c, "attn")?,
        ffn: read_opt_bookend(&mut c, "ffn")?,
        head: read_opt_bookend(&mut c, "head")?,
    })
}

fn put_bookend(out: &mut Vec<u8>, b: &MhcBookend) {
    put_f32_vec(out, &b.fn_weight);
    put_f32_vec(out, &b.scale);
    put_f32_vec(out, &b.base);
}

fn read_opt_bookend(
    c: &mut Cursor,
    what: &'static str,
) -> Result<Option<MhcBookend>, DsV4MhcPersistError> {
    if c.read_u8(what)? == 0 {
        return Ok(None);
    }
    Ok(Some(MhcBookend {
        fn_weight: c.read_f32_vec(what)?,
        scale: c.read_f32_vec(what)?,
        base: c.read_f32_vec(what)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bookend(seed: f32, scale_len: usize) -> MhcBookend {
        MhcBookend {
            fn_weight: (0..24).map(|i| i as f32 * 0.1 + seed).collect(),
            scale: (0..scale_len).map(|i| i as f32 + seed).collect(),
            base: (0..6).map(|i| i as f32 - seed).collect(),
        }
    }

    /// A per-layer blob (attn + ffn, no head) round-trips.
    #[test]
    fn layer_mhc_round_trips() {
        let w = DsV4MhcWeights {
            attn: Some(bookend(1.0, 3)),
            ffn: Some(bookend(2.0, 3)),
            head: None,
        };
        let got = deserialize_dsv4_mhc(&serialize_dsv4_mhc(&w)).unwrap();
        assert_eq!(got.attn, w.attn);
        assert_eq!(got.ffn, w.ffn);
        assert!(got.head.is_none());
    }

    /// The model-level head blob (head only, scalar scale) round-trips.
    #[test]
    fn head_mhc_round_trips() {
        let w = DsV4MhcWeights {
            attn: None,
            ffn: None,
            head: Some(bookend(3.0, 1)),
        };
        let got = deserialize_dsv4_mhc(&serialize_dsv4_mhc(&w)).unwrap();
        assert!(got.attn.is_none());
        assert!(got.ffn.is_none());
        assert_eq!(got.head, w.head);
    }

    /// All-absent round-trips to a 7-byte blob (header + 3 zero flags).
    #[test]
    fn empty_mhc_round_trips() {
        let blob = serialize_dsv4_mhc(&DsV4MhcWeights::default());
        assert_eq!(blob.len(), 4 + 2 + 3);
        let got = deserialize_dsv4_mhc(&blob).unwrap();
        assert!(got.attn.is_none() && got.ffn.is_none() && got.head.is_none());
    }

    /// Bad magic / version / truncation → typed error, never a panic.
    #[test]
    fn malformed_blobs_are_typed_errors() {
        let w = DsV4MhcWeights {
            attn: Some(bookend(1.0, 3)),
            ffn: Some(bookend(2.0, 3)),
            head: Some(bookend(3.0, 1)),
        };
        let good = serialize_dsv4_mhc(&w);

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            deserialize_dsv4_mhc(&bad_magic),
            Err(DsV4MhcPersistError::BadMagic(_))
        ));

        let mut bad_ver = good.clone();
        bad_ver[4] = 0xFF;
        bad_ver[5] = 0xFF;
        assert!(matches!(
            deserialize_dsv4_mhc(&bad_ver),
            Err(DsV4MhcPersistError::UnsupportedVersion(0xFFFF))
        ));

        for cut in 0..good.len() {
            assert!(
                deserialize_dsv4_mhc(&good[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    /// **V3 mHC round-trip gate** (dsv4-vindex-extraction task 4.1).
    ///
    /// Read layer 0's attn + ffn mHC bookends (per-layer f32 reader) and
    /// the global head bookend (via `load_dsv4_head`) from the real
    /// DSv4-Flash GGUF, round-trip through `dsv4_mhc.bin`, and assert
    /// every triple reloads bit-exactly with the hp-derived shapes.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_mhc_round_trips_to_storage() {
        use super::super::dsv4_gguf_reader::read_dsv4_layer_tensors_from_gguf;
        use super::super::dsv4_head_storage::load_dsv4_head;
        use super::super::dsv4_storage_build::DsV4Hyperparams;
        use larql_models::architectures::deepseek_v4_tensors::DsV4TensorKind;
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

        // Per-layer attn + ffn bookends from layer 0.
        let f32s = read_dsv4_layer_tensors_from_gguf(&gguf, 0).expect("read layer 0 f32");
        let bookend = |fn_k, scale_k, base_k| MhcBookend {
            fn_weight: f32s[&fn_k].clone(),
            scale: f32s[&scale_k].clone(),
            base: f32s[&base_k].clone(),
        };
        // Global head bookend from the head loader.
        let head = load_dsv4_head(&gguf, &hp).expect("head");
        let w = DsV4MhcWeights {
            attn: Some(bookend(
                DsV4TensorKind::HcAttnFn,
                DsV4TensorKind::HcAttnScale,
                DsV4TensorKind::HcAttnBase,
            )),
            ffn: Some(bookend(
                DsV4TensorKind::HcFfnFn,
                DsV4TensorKind::HcFfnScale,
                DsV4TensorKind::HcFfnBase,
            )),
            head: Some(MhcBookend {
                fn_weight: head.output_hc_fn.iter().copied().collect(),
                scale: vec![head.output_hc_scale],
                base: head.output_hc_base.clone(),
            }),
        };

        // mHC scale is 3 per-layer, 1 for the head; base is (2+n_hc)*n_hc.
        let hc_mix = (2 + hp.n_hc) * hp.n_hc;
        assert_eq!(w.attn.as_ref().unwrap().scale.len(), 3);
        assert_eq!(w.attn.as_ref().unwrap().base.len(), hc_mix);
        assert_eq!(w.head.as_ref().unwrap().scale.len(), 1);

        let got = deserialize_dsv4_mhc(&serialize_dsv4_mhc(&w)).unwrap();
        assert_eq!(got.attn, w.attn);
        assert_eq!(got.ffn, w.ffn);
        assert_eq!(got.head, w.head);
        eprintln!(
            "mHC round-trip OK: hc_mix={hc_mix}, attn_fn {} elems, head_fn {} elems",
            w.attn.as_ref().unwrap().fn_weight.len(),
            w.head.as_ref().unwrap().fn_weight.len(),
        );
    }
}
