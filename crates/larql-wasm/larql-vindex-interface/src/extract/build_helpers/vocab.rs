//! Whole-word vocabulary reduction.

use ndarray::Array2;

/// Build the whole-word vocabulary: tokens that decode as 3+ char
/// alphabetic words. Returns (token_ids, reduced_embedding_matrix).
pub(crate) fn build_whole_word_vocab(
    tokenizer: &tokenizers::Tokenizer,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    vocab_size: usize,
    hidden_size: usize,
) -> (Vec<usize>, Array2<f32>) {
    let mut ww_ids: Vec<usize> = Vec::new();
    for id in 0..vocab_size {
        if let Ok(tok) = tokenizer.decode(&[id as u32], true) {
            let tok = tok.trim();
            if tok.len() >= 3
                && tok
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '\'')
            {
                ww_ids.push(id);
            }
        }
    }

    let ww_count = ww_ids.len();
    let mut ww_embed = Array2::<f32>::zeros((ww_count, hidden_size));
    for (i, &id) in ww_ids.iter().enumerate() {
        ww_embed.row_mut(i).assign(&embed.row(id));
    }

    eprintln!(
        "    Whole-word vocab: {} tokens (of {})",
        ww_count, vocab_size
    );
    (ww_ids, ww_embed)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::vocab_tokenizer;
    use super::*;

    #[test]
    fn build_whole_word_vocab_keeps_alphabetic_tokens_3plus_chars() {
        let toks = vocab_tokenizer(&["hello", "world", "hi", "no!", "foo123"]);
        let vocab_size = 6;
        let hidden = 4;
        let embed = ndarray::Array2::<f32>::from_shape_fn((vocab_size, hidden), |(i, j)| {
            (i * 100 + j) as f32
        });

        let (ids, ww_embed) = build_whole_word_vocab(&toks, &embed, vocab_size, hidden);
        assert!(ids.contains(&1), "hello kept");
        assert!(ids.contains(&2), "world kept");
        assert!(ids.contains(&5), "foo123 kept (alphanumeric)");
        assert!(!ids.contains(&3), "hi filtered (len<3)");
        assert!(!ids.contains(&4), "'no!' filtered (special char)");

        assert_eq!(ww_embed.shape(), &[ids.len(), hidden]);
        for (j, &v) in ww_embed.row(0).iter().enumerate() {
            assert_eq!(v, (ids[0] * 100 + j) as f32);
        }
    }

    #[test]
    fn build_whole_word_vocab_empty_vocab_returns_empty() {
        let toks = vocab_tokenizer(&[]);
        let embed = ndarray::Array2::<f32>::zeros((1, 4));
        let (ids, ww_embed) = build_whole_word_vocab(&toks, &embed, 1, 4);
        assert!(ids.is_empty());
        assert_eq!(ww_embed.shape(), &[0, 4]);
    }
}
