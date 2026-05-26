//! Offset direction — normalised `embed[output] - embed[input]`,
//! the relation vector for clustering.

use larql_models::ModelWeights;

use crate::extract::constants::FIRST_CONTENT_TOKEN_ID;

/// Compute the offset direction for a gate→down feature pair.
/// Returns normalized(output_embed − input_embed) or None if invalid.
pub(crate) fn compute_offset_direction(
    gate_token: &str,
    output_token_id: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    hidden_size: usize,
    vocab_size: usize,
) -> Option<Vec<f32>> {
    if gate_token.is_empty()
        || output_token_id < FIRST_CONTENT_TOKEN_ID
        || output_token_id >= vocab_size
    {
        return None;
    }

    let enc = tokenizer.encode(gate_token, false).ok()?;
    let ids = enc.get_ids();
    let valid: Vec<usize> = ids
        .iter()
        .filter(|&&id| id as usize >= FIRST_CONTENT_TOKEN_ID)
        .map(|&id| id as usize)
        .filter(|&id| id < vocab_size)
        .collect();
    if valid.is_empty() {
        return None;
    }

    let mut input_avg = vec![0.0f32; hidden_size];
    for &id in &valid {
        for (j, &v) in weights.embed.row(id).iter().enumerate() {
            input_avg[j] += v;
        }
    }
    let n = valid.len() as f32;
    for v in &mut input_avg {
        *v /= n;
    }

    let output_embed = weights.embed.row(output_token_id);
    let offset: Vec<f32> = output_embed
        .iter()
        .zip(input_avg.iter())
        .map(|(o, i)| o - i)
        .collect();
    let norm: f32 = offset.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        Some(offset.iter().map(|v| v / norm).collect())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{vocab_tokenizer, weights_with_embed};
    use super::*;

    #[test]
    fn compute_offset_direction_returns_normalised_vector() {
        // Vocab: "Paris"=3, "France"=4 (≥ FIRST_CONTENT_TOKEN_ID=3).
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "Paris": 3, "France": 4},
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": {"type": "Whitespace"},
            "added_tokens": []
        }"#;
        let toks = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();

        let mut embed = ndarray::Array2::<f32>::zeros((5, 4));
        embed
            .row_mut(4)
            .assign(&ndarray::array![1.0, 0.0, 0.0, 0.0]);
        embed
            .row_mut(3)
            .assign(&ndarray::array![0.0, 1.0, 0.0, 0.0]);
        let weights = weights_with_embed(embed, 5);

        let dir =
            compute_offset_direction("France", 3, &weights, &toks, 4, 5).expect("offset computed");
        let expected_neg = -1.0_f32 / 2.0_f32.sqrt();
        let expected_pos = 1.0_f32 / 2.0_f32.sqrt();
        assert!((dir[0] - expected_neg).abs() < 1e-6);
        assert!((dir[1] - expected_pos).abs() < 1e-6);
        assert!(dir[2].abs() < 1e-6);
        assert!(dir[3].abs() < 1e-6);
    }

    #[test]
    fn compute_offset_direction_returns_none_for_empty_gate_token() {
        let toks = vocab_tokenizer(&["x"]);
        let embed = ndarray::Array2::<f32>::zeros((2, 4));
        let weights = weights_with_embed(embed, 2);
        assert!(compute_offset_direction("", 3, &weights, &toks, 4, 5).is_none());
    }

    #[test]
    fn compute_offset_direction_returns_none_for_special_token_output() {
        let toks = vocab_tokenizer(&["hello"]);
        let embed = ndarray::Array2::<f32>::zeros((5, 4));
        let weights = weights_with_embed(embed, 5);
        for special_id in 0..3 {
            assert!(
                compute_offset_direction("hello", special_id, &weights, &toks, 4, 5).is_none(),
                "id {special_id} must be rejected"
            );
        }
    }

    #[test]
    fn compute_offset_direction_returns_none_for_oob_output_id() {
        let toks = vocab_tokenizer(&["hello"]);
        let embed = ndarray::Array2::<f32>::zeros((5, 4));
        let weights = weights_with_embed(embed, 5);
        assert!(compute_offset_direction("hello", 99, &weights, &toks, 4, 5).is_none());
    }

    #[test]
    fn compute_offset_direction_returns_none_when_gate_decodes_to_unk() {
        let toks = vocab_tokenizer(&["hello"]);
        let embed = ndarray::Array2::<f32>::zeros((5, 4));
        let weights = weights_with_embed(embed, 5);
        assert!(compute_offset_direction("unknown_word", 3, &weights, &toks, 4, 5).is_none());
    }

    #[test]
    fn compute_offset_direction_returns_none_for_zero_offset() {
        let toks = vocab_tokenizer(&["[UNK]", "[PAD]", "[BOS]", "hello"]);
        let mut embed = ndarray::Array2::<f32>::zeros((5, 4));
        embed
            .row_mut(3)
            .assign(&ndarray::array![1.0, 0.0, 0.0, 0.0]);
        embed
            .row_mut(4)
            .assign(&ndarray::array![1.0, 0.0, 0.0, 0.0]);
        let weights = weights_with_embed(embed, 5);
        assert!(compute_offset_direction("hello", 3, &weights, &toks, 4, 5).is_none());
    }
}
