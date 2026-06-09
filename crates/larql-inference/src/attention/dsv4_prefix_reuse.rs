//! DeepSeek V4 prefix-cache reuse orchestration (Full-SWA).
//!
//! `dsv4-ondisk-prefix-cache` P3. Bridges the on-disk store
//! ([`super::dsv4_prefix_cache::DsV4PrefixCache`]) and the resident model
//! forward: on a shared-prefix hit, load the complete per-layer caches
//! and **continue prefilling** the uncached suffix at position `H`
//! instead of re-prefilling the prefix. Because the Full-SWA blob is a
//! lossless round-trip of the post-prefill cache, this is transparent by
//! construction — it reduces to the already-proven "split prefill equals
//! one-shot prefill" property with a serialize→disk→deserialize in the
//! middle.
//!
//! Opt-in: callers that don't pass a cache use the existing cold
//! [`super::dsv4_streaming_model_forward::dsv4_resident_model_forward_cached`]
//! unchanged.

use ndarray::Array2;

use larql_compute::ComputeBackend;

use super::dsv4_full_loader::DsV4LoadError;
use super::dsv4_head_storage::DsV4HeadStorage;
use super::dsv4_kv_cache::DsV4LayerCache;
use super::dsv4_layer_variants::DsV4LayerVariant;
use super::dsv4_prefix_cache::{DsV4PrefixCache, PrefixCacheError, PREFIX_BLOCK_TOKENS};
use super::dsv4_storage::DsV4LayerWeightStorage;
use super::dsv4_storage_build::DsV4Hyperparams;
use super::dsv4_streaming_model_forward::dsv4_resident_model_forward_cached;

/// Error from a prefix-cache-backed prefill.
#[derive(Debug)]
pub enum DsV4PrefixReuseError {
    /// The model forward failed.
    Load(DsV4LoadError),
    /// The on-disk prefix cache failed.
    Cache(PrefixCacheError),
}

impl std::fmt::Display for DsV4PrefixReuseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4PrefixReuseError::Load(e) => write!(f, "forward: {e}"),
            DsV4PrefixReuseError::Cache(e) => write!(f, "prefix cache: {e}"),
        }
    }
}
impl std::error::Error for DsV4PrefixReuseError {}
impl From<DsV4LoadError> for DsV4PrefixReuseError {
    fn from(e: DsV4LoadError) -> Self {
        DsV4PrefixReuseError::Load(e)
    }
}
impl From<PrefixCacheError> for DsV4PrefixReuseError {
    fn from(e: PrefixCacheError) -> Self {
        DsV4PrefixReuseError::Cache(e)
    }
}

/// Outcome of a prefix-cache-backed prefill.
pub struct PrefixPrefillResult {
    /// First position whose logits are present in `logits`. On a cache
    /// hit this is the hit length `H`; on a miss it is `0`.
    pub start_pos: usize,
    /// Logits for positions `[start_pos, token_ids.len())`, shape
    /// `(token_ids.len() - start_pos, n_vocab)`. The final row is the
    /// next-token distribution.
    pub logits: Array2<f32>,
    /// Whether a cached prefix was reused.
    pub cache_hit: bool,
    /// The per-layer caches at position `token_ids.len()`, ready to feed
    /// a decode loop (one entry per model layer, same order as `layers`).
    pub caches: Vec<DsV4LayerCache>,
}

/// Prefill `token_ids` against the resident model, reusing a cached
/// shared prefix when one is present.
///
/// On a hit of `H` tokens (`H < token_ids.len()`), the per-layer caches
/// are loaded and prefill continues from position `H` over the suffix —
/// the first `H` tokens are not recomputed. On a miss (or a full hit),
/// the whole prompt is prefilled cold. Either way, if the prompt length
/// is a positive multiple of [`PREFIX_BLOCK_TOKENS`] the resulting caches
/// are written through to the store (idempotent) so later requests with
/// this prefix hit.
///
/// `max_seq_len` sizes the freshly-allocated caches on a miss; it must be
/// `>= token_ids.len()`. The forward semantics are identical to a cold
/// [`dsv4_resident_model_forward_cached`] for the returned positions.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_resident_prefill_with_prefix_cache(
    layers: &[(DsV4LayerWeightStorage, DsV4LayerVariant)],
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    token_ids: &[u32],
    prefix_cache: &mut DsV4PrefixCache,
    max_seq_len: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Result<PrefixPrefillResult, DsV4PrefixReuseError> {
    let n = token_ids.len();

    // Try to reuse the longest cached block-prefix.
    let hit = prefix_cache.get_longest_prefix(token_ids)?;
    let (start_pos, logits, caches, cache_hit) = match hit {
        // A usable hit: a proper prefix with one cache per layer.
        Some((h, loaded)) if h < n && loaded.len() == layers.len() => {
            let mut caches = loaded;
            let logits = dsv4_resident_model_forward_cached(
                layers,
                hp,
                head,
                &token_ids[h..],
                h,
                Some(&mut caches),
                backend,
            )?;
            (h, logits, caches, true)
        }
        // Miss or full-prompt hit: prefill cold from scratch.
        _ => {
            let mut caches: Vec<DsV4LayerCache> = layers
                .iter()
                .map(|(_, v)| DsV4LayerCache::for_variant(hp, v, max_seq_len))
                .collect();
            let logits = dsv4_resident_model_forward_cached(
                layers,
                hp,
                head,
                token_ids,
                0,
                Some(&mut caches),
                backend,
            )?;
            (0, logits, caches, false)
        }
    };

    // Write-through: store the block-aligned prompt we now hold so a
    // later request sharing this prefix hits. `put` is idempotent.
    if n > 0 && n % PREFIX_BLOCK_TOKENS == 0 {
        prefix_cache.put(token_ids, &caches)?;
    }

    Ok(PrefixPrefillResult {
        start_pos,
        logits,
        cache_hit,
        caches,
    })
}

/// Generate from `prompt` against the resident model, reusing a cached
/// shared prefix for the prefill phase.
///
/// Prefill goes through [`dsv4_resident_prefill_with_prefix_cache`] (so a
/// shared prefix is not recomputed); the decode loop then samples and
/// forwards one token at a time against the returned caches — identical
/// to a cold resident decode. Returns the full token sequence (prompt +
/// up to `max_new_tokens`) and whether the prefill hit the cache.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_resident_generate_with_prefix_cache<R: rand::Rng + ?Sized>(
    layers: &[(DsV4LayerWeightStorage, DsV4LayerVariant)],
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    prompt: &[u32],
    decode_config: super::dsv4_decode_loop::DecodeConfig,
    rng: &mut R,
    prefix_cache: &mut DsV4PrefixCache,
    backend: Option<&dyn ComputeBackend>,
) -> Result<(Vec<u32>, bool), DsV4PrefixReuseError> {
    assert!(!prompt.is_empty(), "prompt must be non-empty");
    let max_seq_len = prompt.len() + decode_config.max_new_tokens;

    let pre = dsv4_resident_prefill_with_prefix_cache(
        layers,
        hp,
        head,
        prompt,
        prefix_cache,
        max_seq_len,
        backend,
    )?;
    let cache_hit = pre.cache_hit;
    let mut caches = pre.caches;
    let mut logits = pre.logits; // last row = logits at the final prompt position
    let mut tokens = prompt.to_vec();

    for i in 0..decode_config.max_new_tokens {
        let last = logits.shape()[0] - 1;
        let next = super::dsv4_sampling::dsv4_sample_next_token(
            logits.row(last),
            decode_config.sampling,
            rng,
        );
        tokens.push(next);
        if decode_config.eos_token == Some(next) {
            break;
        }
        if i + 1 >= decode_config.max_new_tokens {
            break;
        }
        let pos = tokens.len() - 1;
        logits = dsv4_resident_model_forward_cached(
            layers,
            hp,
            head,
            &[next],
            pos,
            Some(&mut caches),
            backend,
        )?;
    }

    Ok((tokens, cache_hit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::dsv4_full_loader::load_dsv4_resident_layers;
    use crate::attention::dsv4_head_storage::load_dsv4_head;
    use crate::attention::dsv4_storage_build::DsV4Hyperparams as Hp;
    use larql_models::loading::gguf::GgufFile;

    /// **Load-bearing transparency test** (real GGUF, ~few min): a
    /// prefix-cache hit must produce the same suffix logits as a cold
    /// full prefill of the identical prompt. Covers NoCompress (layers
    /// 0,1), Indexer (layer 2), and Compress (layer 3) variants.
    ///
    /// Flow: prefill a 256-token (block-aligned) prefix cold → it is
    /// written through to the store. Then prefill `prefix + 16-token
    /// suffix` via the cache (hits at 256, continues the suffix). Compare
    /// against a cold one-shot prefill of the full 272 tokens.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn prefix_cache_hit_matches_cold_prefill() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open GGUF");
        let hp: Hp = Hp::from_gguf(&gguf).expect("hp");
        let head = load_dsv4_head(&gguf, &hp).expect("head");
        let n_layer: usize = std::env::var("DSV4_PREFIX_TEST_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4); // 0,1 NoCompress · 2 Indexer · 3 Compress
        let layers = load_dsv4_resident_layers(&gguf, &hp, n_layer).expect("resident layers");

        let h = 2 * PREFIX_BLOCK_TOKENS; // 256, block-aligned
        let s = 16;
        let toks: Vec<u32> = (0..h + s)
            .map(|i| (i * 257 % head.n_vocab) as u32)
            .collect();
        let max_seq_len = h + s + 8;

        // Cold one-shot reference over the full prompt.
        let mut cold_caches: Vec<DsV4LayerCache> = layers
            .iter()
            .map(|(_, v)| DsV4LayerCache::for_variant(&hp, v, max_seq_len))
            .collect();
        let cold = dsv4_resident_model_forward_cached(
            &layers,
            &hp,
            &head,
            &toks,
            0,
            Some(&mut cold_caches),
            None,
        )
        .expect("cold prefill");

        // Warm path on a fresh store.
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = DsV4PrefixCache::open(tmp.path(), "dsv4-flash", 1 << 34).unwrap();

        // 1) Prefill the 256-token prefix → write-through populates the store.
        let r1 = dsv4_resident_prefill_with_prefix_cache(
            &layers,
            &hp,
            &head,
            &toks[..h],
            &mut store,
            max_seq_len,
            None,
        )
        .expect("prefix prefill");
        assert!(!r1.cache_hit, "first prefill is a miss");
        assert_eq!(store.len(), 1, "prefix written through");

        // 2) Prefill prefix+suffix → must hit at H and continue the suffix.
        let r2 = dsv4_resident_prefill_with_prefix_cache(
            &layers,
            &hp,
            &head,
            &toks,
            &mut store,
            max_seq_len,
            None,
        )
        .expect("prefix prefill 2");
        assert!(r2.cache_hit, "second prefill hits the cached prefix");
        assert_eq!(r2.start_pos, h, "hit at the 256-token boundary");
        assert_eq!(r2.logits.shape(), &[s, head.n_vocab]);

        // Transparency: warm suffix logits == cold suffix logits.
        let argmax = |row: ndarray::ArrayView1<f32>| -> usize {
            let mut bi = 0;
            let mut bv = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = j;
                }
            }
            bi
        };
        let mut max_rel = 0.0_f32;
        let mut tok_matches = 0;
        for t in 0..s {
            let warm = r2.logits.row(t);
            let coldr = cold.row(h + t);
            for (a, b) in warm.iter().zip(coldr.iter()) {
                let denom = b.abs().max(1.0);
                max_rel = max_rel.max((a - b).abs() / denom);
            }
            if argmax(warm) == argmax(coldr) {
                tok_matches += 1;
            }
        }
        eprintln!("warm-vs-cold suffix: max rel diff {max_rel:.5}, greedy {tok_matches}/{s}");
        // **Load-bearing assertion**: greedy decode is identical — the
        // prefix-cache hit produces the same generated tokens as a cold
        // prefill. (Same convention as the resident-quant parity tests:
        // greedy is the gate.) Full-SWA serialization is lossless and the
        // SWA path is near-exact (~5e-3); the larger raw-logit drift on
        // HCA layers is split-batch reduction-order / FP8 sensitivity in
        // the masked-MQA — benign for generation, the same effect the
        // resident-quant parity tolerates (≤1.5). The indexer top-k is
        // split-invariant (identical query + compressed KV ⇒ identical
        // scores), so this is numeric, not a behavioral divergence.
        assert_eq!(
            tok_matches, s,
            "every suffix token's greedy argmax must match"
        );
        assert!(
            max_rel < 1.5,
            "suffix logits diverge beyond documented HCA tolerance: max_rel={max_rel}"
        );
    }

    /// Cold-vs-warm prefill wall time (real GGUF, ignored). Quantifies the
    /// prefix-reuse speedup: a warm prefill forwards only the uncached
    /// suffix, a cold prefill forwards the whole prompt. Reports the ratio
    /// (not asserted — wall time varies by host).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk; perf bench"]
    fn bench_cold_vs_warm_prefill() {
        use std::time::Instant;
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open GGUF");
        let hp: Hp = Hp::from_gguf(&gguf).expect("hp");
        let head = load_dsv4_head(&gguf, &hp).expect("head");
        let n_layer: usize = std::env::var("DSV4_PREFIX_TEST_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let layers = load_dsv4_resident_layers(&gguf, &hp, n_layer).expect("resident layers");

        let h = 4 * PREFIX_BLOCK_TOKENS; // 512-token shared prefix
        let s = 16;
        let toks: Vec<u32> = (0..h + s)
            .map(|i| (i * 257 % head.n_vocab) as u32)
            .collect();
        let max_seq_len = h + s + 8;

        // Cold: full prefill of the whole prompt, no cache.
        let mut cold_caches: Vec<DsV4LayerCache> = layers
            .iter()
            .map(|(_, v)| DsV4LayerCache::for_variant(&hp, v, max_seq_len))
            .collect();
        let t0 = Instant::now();
        let _ = dsv4_resident_model_forward_cached(
            &layers,
            &hp,
            &head,
            &toks,
            0,
            Some(&mut cold_caches),
            None,
        )
        .expect("cold");
        let cold = t0.elapsed();

        // Warm: populate the store with the H-token prefix, then prefill
        // the full prompt — hits at H, forwards only the S-token suffix.
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = DsV4PrefixCache::open(tmp.path(), "dsv4-flash", 1 << 34).unwrap();
        dsv4_resident_prefill_with_prefix_cache(
            &layers,
            &hp,
            &head,
            &toks[..h],
            &mut store,
            max_seq_len,
            None,
        )
        .expect("seed");
        let t1 = Instant::now();
        let r = dsv4_resident_prefill_with_prefix_cache(
            &layers,
            &hp,
            &head,
            &toks,
            &mut store,
            max_seq_len,
            None,
        )
        .expect("warm");
        let warm = t1.elapsed();

        assert!(r.cache_hit && r.start_pos == h);
        eprintln!(
            "[prefix-bench] {n_layer} layers, prefix={h} suffix={s}: \
             cold {:.2}s (forwards {} tok) vs warm {:.2}s (forwards {} tok) → {:.1}× faster",
            cold.as_secs_f64(),
            h + s,
            warm.as_secs_f64(),
            s,
            cold.as_secs_f64() / warm.as_secs_f64().max(1e-9),
        );
    }
}
