//! Real Gemma 3 4B Q4_K vindex parity test for the persistent KV cache path.
//!
//! Set `LARQL_VINDEX_PATH=output/gemma-3-4b-it-vindex` and run with
//! `--ignored` to exercise. Compares three computations of the same
//! position's hidden state and isolates which stage of the cache pipeline
//! (prefill snapshot vs decode-step continuation) introduces divergence.

use larql_inference::attention::KvCache;
use larql_inference::open_inference_vindex;
use larql_inference::vindex::{predict_q4k_hidden, predict_q4k_hidden_with_cache};

const ENV_VINDEX_PATH: &str = "LARQL_VINDEX_PATH";

fn max_abs_diff(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> f32 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "requires LARQL_VINDEX_PATH to a Gemma 3 4B Q4_K vindex"]
fn cache_prefill_matches_uncached_prefill() {
    // INVARIANT: prefilling via `predict_q4k_hidden_with_cache(.., Some(empty))`
    // MUST return the same hidden state as `predict_q4k_hidden(..)`. The
    // cache-prefill branch only adds a `cache.set_layer(..)` side effect.
    let Ok(vindex_path) = std::env::var(ENV_VINDEX_PATH) else {
        eprintln!("{ENV_VINDEX_PATH} not set — skipping");
        return;
    };
    let path = std::path::Path::new(&vindex_path);
    let index = open_inference_vindex(path).expect("open vindex");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_q4k(path, &mut cb).expect("load weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).expect("tokenizer");

    let prompt = "Hello";
    let token_ids =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, prompt).expect("tokenize");
    eprintln!("prompt {} tokens: {:?}", token_ids.len(), token_ids);

    let h_uncached = predict_q4k_hidden(&mut weights, &token_ids, &index, None);

    let mut cache = KvCache::with_layers(weights.num_layers);
    let h_cached_prefill =
        predict_q4k_hidden_with_cache(&mut weights, &token_ids, &index, None, Some(&mut cache));

    let diff = max_abs_diff(&h_uncached, &h_cached_prefill);
    eprintln!("prefill max-abs diff (uncached vs cache=Some(empty)): {diff:.6}");
    assert!(
        diff < 1e-3,
        "cache-prefill snapshot path diverged from uncached prefill (max-abs={diff}). \
         Bug is in `run_attention_block_with_kv_out_with_cache`'s prefill branch \
         or in `predict_q4k_hidden_with_cache`'s wrapper layer."
    );
}

#[test]
#[ignore = "requires LARQL_VINDEX_PATH to a Gemma 3 4B Q4_K vindex"]
fn cache_prefill_predict_matches_uncached_predict() {
    // INVARIANT: top-1 prediction from `predict_q4k_with_cache` on an empty
    // cache MUST match `predict_q4k` on the same prompt. If hidden states
    // match (proven by `cache_prefill_matches_uncached_prefill`) but
    // predictions diverge, the bug is in the lm_head path or the
    // predict-helper wrapper, not in the attention forward.
    let Ok(vindex_path) = std::env::var(ENV_VINDEX_PATH) else {
        eprintln!("{ENV_VINDEX_PATH} not set — skipping");
        return;
    };
    let path = std::path::Path::new(&vindex_path);
    let index = open_inference_vindex(path).expect("open vindex");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_q4k(path, &mut cb).expect("load weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).expect("tokenizer");

    let prompt = std::env::var("LARQL_SMOKE_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let token_ids =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, &prompt).expect("tokenize");
    eprintln!("prompt {} tokens", token_ids.len());

    let uncached =
        larql_inference::vindex::predict_q4k(&mut weights, &tokenizer, &token_ids, 5, &index);

    let mut cache = KvCache::with_layers(weights.num_layers);
    let cached = larql_inference::vindex::predict_q4k_with_cache(
        &mut weights,
        &tokenizer,
        &token_ids,
        5,
        &index,
        &mut cache,
    );

    eprintln!("uncached top-1: {:?}", uncached.token_ids.first());
    eprintln!("uncached predictions: {:?}", uncached.predictions.first());
    eprintln!("cached   top-1: {:?}", cached.token_ids.first());
    eprintln!("cached   predictions: {:?}", cached.predictions.first());
    assert_eq!(
        uncached.token_ids.first(),
        cached.token_ids.first(),
        "top-1 token diverges between uncached and cache=Some(empty)"
    );
}

#[test]
#[ignore = "requires LARQL_VINDEX_PATH to a Gemma 3 4B Q4_K vindex"]
fn cache_decode_step_matches_uncached_full_replay() {
    // INVARIANT: prefill N tokens via cache, then decode token N+1 via cache,
    // MUST produce the same last-row hidden as running uncached on N+1 tokens.
    let Ok(vindex_path) = std::env::var(ENV_VINDEX_PATH) else {
        eprintln!("{ENV_VINDEX_PATH} not set — skipping");
        return;
    };
    let path = std::path::Path::new(&vindex_path);
    let index = open_inference_vindex(path).expect("open vindex");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_q4k(path, &mut cb).expect("load weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).expect("tokenizer");

    let prompt = std::env::var("LARQL_SMOKE_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let token_ids =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, &prompt).expect("tokenize");
    assert!(token_ids.len() >= 2, "need at least 2 prompt tokens");
    let n = token_ids.len();
    let prompt_first = &token_ids[..n - 1];
    let prompt_last = &token_ids[n - 1..n];
    eprintln!("prompt {} tokens, prefill {}, decode 1", n, n - 1);

    let h_uncached_full = predict_q4k_hidden(&mut weights, &token_ids, &index, None);

    let mut cache = KvCache::with_layers(weights.num_layers);
    let _h_prefill =
        predict_q4k_hidden_with_cache(&mut weights, prompt_first, &index, None, Some(&mut cache));
    let h_decode =
        predict_q4k_hidden_with_cache(&mut weights, prompt_last, &index, None, Some(&mut cache));

    assert_eq!(h_decode.shape(), &[1, weights.hidden_size]);
    let expected_last = h_uncached_full.row(n - 1);
    let actual = h_decode.row(0);
    let max_diff = actual
        .iter()
        .zip(expected_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("decode-step max-abs diff: {max_diff:.6}");
    // Decode-step uses Q4kDirectFfn (Q4_K × Q8_K matvec on quantised bytes);
    // uncached full-replay's seq>=2 row uses WeightFfn (BLAS GEMM over the
    // f32 dequant cache). Output therefore diverges by the Q8_K activation
    // quantisation noise envelope (≤ 1.5 % relative). Tolerance is calibrated
    // against the largest activation magnitude in the residual stream
    // (Gemma 3 hidden states reach |a| ≈ 30 with the +1.0 norm offset on
    // post-norm layers); 1.5 % of that is ~0.45, so 0.5 absolute is the
    // Q8_K-noise headroom. A real FFN regression would diverge order-of-
    // magnitude higher.
    assert!(
        max_diff < 0.5,
        "decode-step path diverged from uncached full-replay (max-abs={max_diff}) \
         beyond Q8_K activation-noise envelope. Bug is either in \
         `run_attention_block_decode_step` for this arch (sliding-window, \
         per-layer rope_base, post-norms, …) or in the Q4kDirectFfn dispatch."
    );
}
