//! Per-feature top whole-word token — the "what activates this
//! feature" label used downstream by clustering.

use larql_models::{FfnType, ModelWeights};
use ndarray::Array2;

use crate::extract::constants::GATE_TOP_TOKEN_BATCH;

/// Compute gate top tokens for features at a layer using whole-word
/// embeddings. Returns one decoded whole-word token per feature.
pub(crate) fn compute_gate_top_tokens(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    layer: usize,
    num_features: usize,
    ww_ids: &[usize],
    ww_embed: &Array2<f32>,
) -> Vec<String> {
    // Gated FFN routes through `ffn_gate`; non-gated FFN (GPT-2,
    // StarCoder2) reuses `ffn_up` for the same per-feature input
    // direction.
    let gate_key = match weights.arch.ffn_type() {
        FfnType::Gated => weights.arch.ffn_gate_key(layer),
        FfnType::Standard => weights.arch.ffn_up_key(layer),
    };
    let w_gate = match weights.tensors.get(&gate_key) {
        Some(w) => w,
        None => return vec![String::new(); num_features],
    };

    let mut tokens = vec![String::new(); num_features];
    let gbatch = GATE_TOP_TOKEN_BATCH;
    for gstart in (0..num_features).step_by(gbatch) {
        let gend = (gstart + gbatch).min(num_features);
        let chunk = w_gate.slice(ndarray::s![gstart..gend, ..]);
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let proj = cpu.matmul_transb(ww_embed.view(), chunk.view());
        for f in 0..(gend - gstart) {
            let col = proj.column(f);
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in col.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            let tok_id = ww_ids[best_idx];
            tokens[gstart + f] = tokenizer
                .decode(&[tok_id as u32], true)
                .unwrap_or_default()
                .trim()
                .to_string();
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{insert_tensor, vocab_tokenizer, weights_with_embed};
    use super::super::vocab::build_whole_word_vocab;
    use super::*;

    #[test]
    fn compute_gate_top_tokens_returns_empty_strings_when_no_gate_tensor() {
        let toks = vocab_tokenizer(&["x"]);
        let embed = ndarray::Array2::<f32>::zeros((2, 4));
        let weights = weights_with_embed(embed.clone(), 2);
        let ww_ids = vec![0usize];
        let result = compute_gate_top_tokens(&weights, &toks, 0, 5, &ww_ids, &embed);
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|s| s.is_empty()));
    }

    #[test]
    fn compute_gate_top_tokens_picks_argmax_word_per_feature() {
        // hidden=4, vocab has 4 whole-word tokens at ids 1..=4 plus
        // [UNK] at 0; embed[i] = e_{i-1} so word `i-1` aligns with
        // axis i-1; gate has 4 features with weight 1 at position f.
        // ww_embed @ gate^T → score[i, f] = 1 iff i==f → feature f's
        // argmax word is the f-th whole-word id.
        let words = ["alpha", "beta", "gamma", "delta"];
        let toks = vocab_tokenizer(&words);
        let hidden = 4;
        let vocab_size = 5;

        let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
        for i in 0..hidden {
            embed[[i + 1, i]] = 1.0;
        }
        let mut weights = weights_with_embed(embed.clone(), vocab_size);

        let (ww_ids, ww_embed) = build_whole_word_vocab(&toks, &embed, vocab_size, hidden);
        assert_eq!(ww_ids.len(), 4, "expected 4 whole-words for the test");

        let mut gate = ndarray::Array2::<f32>::zeros((4, hidden));
        for f in 0..4 {
            gate[[f, f]] = 1.0;
        }
        let arch = &weights.arch;
        let gate_key = arch.ffn_gate_key(0);
        insert_tensor(&mut weights, &gate_key, gate);

        let result = compute_gate_top_tokens(&weights, &toks, 0, 4, &ww_ids, &ww_embed);
        assert_eq!(result.len(), 4);

        let decoded: Vec<String> = ww_ids
            .iter()
            .map(|&id| {
                toks.decode(&[id as u32], true)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            })
            .collect();
        for (f, top) in result.iter().enumerate() {
            assert_eq!(top, &decoded[f], "feature {f} should pick word {f}");
        }
    }

    #[test]
    fn compute_gate_top_tokens_iterates_in_batches() {
        // num_features > GATE_TOP_TOKEN_BATCH forces multi-chunk
        // execution; pin that all features still get labels.
        let num_features = GATE_TOP_TOKEN_BATCH + 5;

        let words = ["alpha"];
        let toks = vocab_tokenizer(&words);
        let hidden = 4;
        let vocab_size = 2;
        let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
        embed[[1, 0]] = 1.0;
        let mut weights = weights_with_embed(embed.clone(), vocab_size);

        let (ww_ids, ww_embed) = build_whole_word_vocab(&toks, &embed, vocab_size, hidden);
        assert_eq!(ww_ids.len(), 1);

        let mut gate = ndarray::Array2::<f32>::zeros((num_features, hidden));
        for f in 0..num_features {
            gate[[f, 0]] = 1.0;
        }
        let arch = &weights.arch;
        let gate_key = arch.ffn_gate_key(0);
        insert_tensor(&mut weights, &gate_key, gate);

        let result = compute_gate_top_tokens(&weights, &toks, 0, num_features, &ww_ids, &ww_embed);
        assert_eq!(result.len(), num_features);
        assert!(
            result.iter().all(|s| s == "alpha"),
            "all features should pick the only whole-word"
        );
    }
}
