//! DeepSeek V4 HCA-compressor + lightning-indexer vindex storage
//! (`dsv4-vindex-extraction` V2).
//!
//! The variant-gated structural attention weights neither the generic
//! Q/K/V/O writer nor V1's [`super::dsv4_vindex_attn`] covers:
//!
//! - **HCA compressor** — present when `compress_ratio > 0` (Compress +
//!   Indexer layers): `attn_compress_kv`/`gate` (raw Q4_K passthrough) +
//!   `attn_compress_ape`/`norm` (f32, small).
//! - **lightning indexer** — present only when `compress_ratio == 4`:
//!   its own sub-compressor (`indexer.compress_kv`/`gate`/`ape`/`norm`),
//!   `indexer.attn_q_b` (raw Q4_K, the large indexer Q-up), and
//!   `indexer.proj` (Q4_K in the GGUF — stored raw to stay byte-faithful,
//!   even though the resident loader currently dequantizes it).
//!
//! `compress_ratio == 4` implies the main compressor is also present, so a
//! valid Indexer layer carries both `compressor` *and* `indexer`. A
//! NoCompress layer (`compress_ratio == 0`; layers 0–1) carries neither.
//! Top-k masks are **not** stored — selection stays runtime-dynamic.
//!
//! Built on the shared [`super::dsv4_vindex_wire`] primitives.

use super::dsv4_gguf_reader::RawExpertTensor;
use super::dsv4_vindex_wire::{put_f32_vec, put_header, put_raw, Cursor};

/// Error decoding a `dsv4_hca.bin` blob (the shared vindex wire error).
pub type DsV4HcaPersistError = super::dsv4_vindex_wire::DsV4VindexWireError;

/// 4-byte magic: "D4HC" (DSv4 vindex HCA/indexer).
const MAGIC: [u8; 4] = *b"D4HC";
const VERSION: u16 = 1;

/// One HCA compressor (the main one, or the indexer's sub-compressor) in
/// extraction form: `kv`/`gate` raw quantized, `ape`/`norm` f32.
#[derive(Clone)]
pub struct CompressorVindex {
    /// `(coff*head_dim, n_embd)` compressed-KV down-projection.
    pub kv: RawExpertTensor,
    /// `(coff*head_dim, n_embd)` gate.
    pub gate: RawExpertTensor,
    /// `(compress_ratio, coff*head_dim)` absolute-position embedding.
    pub ape: Vec<f32>,
    /// `(head_dim,)` RMSNorm gain (or `(indexer_head_size,)` for the
    /// indexer's sub-compressor).
    pub norm: Vec<f32>,
}

/// The lightning indexer's weights (Indexer layers, `compress_ratio == 4`).
#[derive(Clone)]
pub struct IndexerVindex {
    /// The indexer's own sub-compressor.
    pub compressor: CompressorVindex,
    /// `(n_index_head*indexer_head_size, q_lora_rank)` indexer Q-up.
    pub attn_q_b: RawExpertTensor,
    /// `(n_index_head, n_embd)` indexer projection (Q4_K, stored raw).
    pub proj: RawExpertTensor,
}

/// One DSv4 layer's HCA/indexer weights — both `None` for a NoCompress
/// layer; `compressor` only for a Compress layer; both for an Indexer
/// layer.
#[derive(Clone, Default)]
pub struct DsV4HcaWeights {
    pub compressor: Option<CompressorVindex>,
    pub indexer: Option<IndexerVindex>,
}

/// Serialize one layer's HCA/indexer weights to a `dsv4_hca.bin` blob.
pub fn serialize_dsv4_hca(w: &DsV4HcaWeights) -> Vec<u8> {
    let mut out = Vec::new();
    put_header(&mut out, MAGIC, VERSION);
    match &w.compressor {
        Some(c) => {
            out.push(1);
            put_compressor(&mut out, c);
        }
        None => out.push(0),
    }
    match &w.indexer {
        Some(i) => {
            out.push(1);
            put_compressor(&mut out, &i.compressor);
            put_raw(&mut out, &i.attn_q_b);
            put_raw(&mut out, &i.proj);
        }
        None => out.push(0),
    }
    out
}

/// Deserialize one layer's HCA/indexer weights from a `dsv4_hca.bin` blob.
pub fn deserialize_dsv4_hca(bytes: &[u8]) -> Result<DsV4HcaWeights, DsV4HcaPersistError> {
    let mut c = Cursor::new(bytes);
    c.read_header(MAGIC, VERSION)?;
    let compressor = if c.read_u8("has_compressor")? != 0 {
        Some(read_compressor(&mut c)?)
    } else {
        None
    };
    let indexer = if c.read_u8("has_indexer")? != 0 {
        Some(IndexerVindex {
            compressor: read_compressor(&mut c)?,
            attn_q_b: c.read_raw("idx_attn_q_b")?,
            proj: c.read_raw("idx_proj")?,
        })
    } else {
        None
    };
    Ok(DsV4HcaWeights {
        compressor,
        indexer,
    })
}

fn put_compressor(out: &mut Vec<u8>, c: &CompressorVindex) {
    put_raw(out, &c.kv);
    put_raw(out, &c.gate);
    put_f32_vec(out, &c.ape);
    put_f32_vec(out, &c.norm);
}

fn read_compressor(c: &mut Cursor) -> Result<CompressorVindex, DsV4HcaPersistError> {
    Ok(CompressorVindex {
        kv: c.read_raw("compress_kv")?,
        gate: c.read_raw("compress_gate")?,
        ape: c.read_f32_vec("compress_ape")?,
        norm: c.read_f32_vec("compress_norm")?,
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

    fn compressor(seed: u8) -> CompressorVindex {
        CompressorVindex {
            kv: raw(8, 32, seed),
            gate: raw(8, 32, seed + 1),
            ape: vec![0.5; 4 * 8],
            norm: vec![1.25; 8],
        }
    }

    fn assert_raw_eq(a: &RawExpertTensor, b: &RawExpertTensor) {
        assert_eq!(a.tensor_type, b.tensor_type);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(a.bytes, b.bytes, "raw bytes differ");
    }

    fn assert_compressor_eq(a: &CompressorVindex, b: &CompressorVindex) {
        assert_raw_eq(&a.kv, &b.kv);
        assert_raw_eq(&a.gate, &b.gate);
        assert_eq!(a.ape, b.ape);
        assert_eq!(a.norm, b.norm);
    }

    /// A Compress layer (compressor, no indexer) round-trips.
    #[test]
    fn compressor_only_round_trips() {
        let w = DsV4HcaWeights {
            compressor: Some(compressor(1)),
            indexer: None,
        };
        let got = deserialize_dsv4_hca(&serialize_dsv4_hca(&w)).unwrap();
        assert_compressor_eq(
            got.compressor.as_ref().unwrap(),
            w.compressor.as_ref().unwrap(),
        );
        assert!(got.indexer.is_none());
    }

    /// An Indexer layer (both compressor + indexer) round-trips.
    #[test]
    fn indexer_layer_round_trips() {
        let w = DsV4HcaWeights {
            compressor: Some(compressor(1)),
            indexer: Some(IndexerVindex {
                compressor: compressor(10),
                attn_q_b: raw(16, 24, 20),
                proj: raw(4, 32, 30),
            }),
        };
        let got = deserialize_dsv4_hca(&serialize_dsv4_hca(&w)).unwrap();
        assert_compressor_eq(
            got.compressor.as_ref().unwrap(),
            w.compressor.as_ref().unwrap(),
        );
        let gi = got.indexer.as_ref().unwrap();
        let wi = w.indexer.as_ref().unwrap();
        assert_compressor_eq(&gi.compressor, &wi.compressor);
        assert_raw_eq(&gi.attn_q_b, &wi.attn_q_b);
        assert_raw_eq(&gi.proj, &wi.proj);
    }

    /// A NoCompress layer (neither) round-trips to a 7-byte blob
    /// (header + two zero flags).
    #[test]
    fn nocompress_layer_round_trips() {
        let w = DsV4HcaWeights::default();
        let blob = serialize_dsv4_hca(&w);
        assert_eq!(blob.len(), 4 + 2 + 1 + 1, "header + 2 absent flags");
        let got = deserialize_dsv4_hca(&blob).unwrap();
        assert!(got.compressor.is_none());
        assert!(got.indexer.is_none());
    }

    /// Bad magic / version / truncation → typed error, never a panic.
    #[test]
    fn malformed_blobs_are_typed_errors() {
        let w = DsV4HcaWeights {
            compressor: Some(compressor(1)),
            indexer: Some(IndexerVindex {
                compressor: compressor(10),
                attn_q_b: raw(16, 24, 20),
                proj: raw(4, 32, 30),
            }),
        };
        let good = serialize_dsv4_hca(&w);

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            deserialize_dsv4_hca(&bad_magic),
            Err(DsV4HcaPersistError::BadMagic(_))
        ));

        let mut bad_ver = good.clone();
        bad_ver[4] = 0xFF;
        bad_ver[5] = 0xFF;
        assert!(matches!(
            deserialize_dsv4_hca(&bad_ver),
            Err(DsV4HcaPersistError::UnsupportedVersion(0xFFFF))
        ));

        for cut in 0..good.len() {
            assert!(
                deserialize_dsv4_hca(&good[..cut]).is_err(),
                "truncation at {cut} must error"
            );
        }
    }

    /// **V2 round-trip gate** (dsv4-vindex-extraction task 3.2).
    ///
    /// Scan the real DSv4-Flash GGUF for a Compress layer and an Indexer
    /// layer, read their HCA/indexer weights via the *same* readers the
    /// resident loader feeds, round-trip through `dsv4_hca.bin`, and
    /// assert byte/type/shape equality on every raw tensor plus f32
    /// equality on the ape/norm — variant-gated (compressor only for the
    /// Compress layer; compressor + indexer for the Indexer layer).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn real_gguf_hca_round_trips_to_storage() {
        use super::super::dsv4_gguf_reader::{
            read_dsv4_layer_raw_expert_tensors_from_gguf, read_dsv4_layer_tensors_from_gguf,
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

        // Find one Compress (cr>0, !=4) layer and one Indexer (cr==4) layer.
        // DSv4-Flash has 43 blocks (matches the sibling real-GGUF tests).
        const N_LAYER: usize = 43;
        let mut compress_layer = None;
        let mut indexer_layer = None;
        for l in 0..N_LAYER {
            let v = detect_layer_variant(&gguf, l, hp.head_dim);
            match v.compress_ratio {
                Some(4) if indexer_layer.is_none() => indexer_layer = Some(l),
                Some(cr) if cr > 0 && compress_layer.is_none() => compress_layer = Some(l),
                _ => {}
            }
            if compress_layer.is_some() && indexer_layer.is_some() {
                break;
            }
        }
        let compress_layer = compress_layer.expect("a Compress layer exists");
        let indexer_layer = indexer_layer.expect("an Indexer layer exists");
        eprintln!("Compress layer {compress_layer}, Indexer layer {indexer_layer}");

        let take = |raw: &mut HashMap<DsV4TensorKind, RawExpertTensor>, k| {
            raw.remove(&k).unwrap_or_else(|| panic!("missing {k:?}"))
        };
        let build_compressor = |raw: &mut HashMap<DsV4TensorKind, RawExpertTensor>,
                                f32s: &HashMap<DsV4TensorKind, Vec<f32>>,
                                kv,
                                gate,
                                ape,
                                norm| CompressorVindex {
            kv: take(raw, kv),
            gate: take(raw, gate),
            ape: f32s[&ape].clone(),
            norm: f32s[&norm].clone(),
        };

        // ── Compress layer: compressor only ──
        {
            let mut raw = read_dsv4_layer_raw_expert_tensors_from_gguf(
                &gguf,
                compress_layer,
                &[
                    DsV4TensorKind::AttnCompressorKv,
                    DsV4TensorKind::AttnCompressorGate,
                ],
            )
            .expect("read compress raw");
            let f32s = read_dsv4_layer_tensors_from_gguf(&gguf, compress_layer)
                .expect("read compress f32");
            let w = DsV4HcaWeights {
                compressor: Some(build_compressor(
                    &mut raw,
                    &f32s,
                    DsV4TensorKind::AttnCompressorKv,
                    DsV4TensorKind::AttnCompressorGate,
                    DsV4TensorKind::AttnCompressorApe,
                    DsV4TensorKind::AttnCompressorNorm,
                )),
                indexer: None,
            };
            let got = deserialize_dsv4_hca(&serialize_dsv4_hca(&w)).unwrap();
            assert!(got.indexer.is_none(), "Compress layer has no indexer");
            assert_compressor_eq(
                got.compressor.as_ref().unwrap(),
                w.compressor.as_ref().unwrap(),
            );
            assert_ne!(
                got.compressor.as_ref().unwrap().kv.tensor_type,
                larql_models::quant::ggml::TYPE_F32,
                "compress_kv should be quantized"
            );
        }

        // ── Indexer layer: compressor + indexer ──
        {
            let mut raw = read_dsv4_layer_raw_expert_tensors_from_gguf(
                &gguf,
                indexer_layer,
                &[
                    DsV4TensorKind::AttnCompressorKv,
                    DsV4TensorKind::AttnCompressorGate,
                    DsV4TensorKind::IndexerCompressorKv,
                    DsV4TensorKind::IndexerCompressorGate,
                    DsV4TensorKind::IndexerAttnQB,
                    DsV4TensorKind::IndexerProj,
                ],
            )
            .expect("read indexer raw");
            let f32s =
                read_dsv4_layer_tensors_from_gguf(&gguf, indexer_layer).expect("read indexer f32");
            let w = DsV4HcaWeights {
                compressor: Some(build_compressor(
                    &mut raw,
                    &f32s,
                    DsV4TensorKind::AttnCompressorKv,
                    DsV4TensorKind::AttnCompressorGate,
                    DsV4TensorKind::AttnCompressorApe,
                    DsV4TensorKind::AttnCompressorNorm,
                )),
                indexer: Some(IndexerVindex {
                    compressor: build_compressor(
                        &mut raw,
                        &f32s,
                        DsV4TensorKind::IndexerCompressorKv,
                        DsV4TensorKind::IndexerCompressorGate,
                        DsV4TensorKind::IndexerCompressorApe,
                        DsV4TensorKind::IndexerCompressorNorm,
                    ),
                    attn_q_b: take(&mut raw, DsV4TensorKind::IndexerAttnQB),
                    proj: take(&mut raw, DsV4TensorKind::IndexerProj),
                }),
            };
            let got = deserialize_dsv4_hca(&serialize_dsv4_hca(&w)).unwrap();
            assert_compressor_eq(
                got.compressor.as_ref().unwrap(),
                w.compressor.as_ref().unwrap(),
            );
            let gi = got.indexer.as_ref().unwrap();
            let wi = w.indexer.as_ref().unwrap();
            assert_compressor_eq(&gi.compressor, &wi.compressor);
            assert_raw_eq(&gi.attn_q_b, &wi.attn_q_b);
            assert_raw_eq(&gi.proj, &wi.proj);
            eprintln!(
                "indexer round-trip OK: attn_q_b type {} ({} bytes), proj type {} ({} bytes)",
                wi.attn_q_b.tensor_type,
                wi.attn_q_b.bytes.len(),
                wi.proj.tensor_type,
                wi.proj.bytes.len(),
            );
        }
    }
}
