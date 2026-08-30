//! Token embedding — lookup + architecture-specific scaling.

use super::embedding_plan::{EmbeddingChunk, EmbeddingPlan};
use larql_models::{ModelWeights, PrecomputedScaling};
use ndarray::Array2;

/// Embed token IDs with architecture-specific scaling.
///
/// Looks up one row per token in `weights.embed`, multiplies by
/// `arch.embed_scale()`. The scale factor handles models that store
/// pre-scaled embeddings (e.g. Gemma) vs. those that don't.
pub fn embed_tokens_pub(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale_multiplier();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}

/// Multi-modal embedding — assemble one hidden-state matrix from a plan
/// of token chunks and pre-computed modal embeddings.
///
/// Phase 0 contract: when the plan is text-only (`plan.is_text_only()`),
/// this MUST produce **bit-identical** output to `embed_tokens_pub` on
/// the concatenation of all token chunks. That guarantee is what makes
/// the Phase 0 single-PR shape safe — re-routing a call site from
/// `embed_tokens_pub(toks)` to `embed_plan(&EmbeddingPlan::from_tokens(toks))`
/// is a no-op semantically. The bit-identity is pinned by tests at the
/// bottom of this file.
///
/// For mixed plans (Phase 1+), `Precomputed` chunks are spliced in as
/// they appear. The host is responsible for applying any
/// modality-specific scaling per `MultiModalProtocol::precomputed_scaling()`
/// *before* placing the rows on the chunk — see the doc-comment on
/// `EmbeddingChunk::Precomputed`. When `SameAsTokens` is requested,
/// this function applies the extra `arch.embed_scale()` factor; when
/// `None` (the default), precomputed rows go in as-is.
///
/// `PositionScheme::Mrope` is accepted but does not affect embedding
/// values — positions are consumed downstream by RoPE. Phase 0 simply
/// passes the plan through; Phase 4 wires M-RoPE.
pub fn embed_plan(weights: &ModelWeights, plan: &EmbeddingPlan) -> Array2<f32> {
    // Text-only fast path: bit-identity with embed_tokens_pub.
    // Concatenate every Tokens chunk and call straight through.
    if plan.is_text_only() {
        if plan.chunks.len() == 1 {
            if let EmbeddingChunk::Tokens(toks) = &plan.chunks[0] {
                return embed_tokens_pub(weights, toks);
            }
        }
        let mut combined: Vec<u32> = Vec::with_capacity(plan.total_rows());
        for chunk in &plan.chunks {
            if let EmbeddingChunk::Tokens(toks) = chunk {
                combined.extend_from_slice(toks);
            }
        }
        return embed_tokens_pub(weights, &combined);
    }

    // Mixed-modality (or M-RoPE-positioned) path. Build chunks
    // independently then row-stack.
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale_multiplier();
    let precomputed_scaling = weights
        .arch
        .multimodal()
        .map(|m| m.precomputed_scaling())
        .unwrap_or(PrecomputedScaling::None);

    let mut h = Array2::<f32>::zeros((plan.total_rows(), hidden));
    let mut cursor = 0usize;

    for chunk in &plan.chunks {
        match chunk {
            EmbeddingChunk::Tokens(toks) => {
                for &tok_id in toks {
                    let row = weights.embed.row(tok_id as usize);
                    for j in 0..hidden {
                        h[[cursor, j]] = row[j] * scale;
                    }
                    cursor += 1;
                }
            }
            EmbeddingChunk::Precomputed { rows, .. } => {
                let row_scale = match precomputed_scaling {
                    PrecomputedScaling::None => 1.0_f32,
                    PrecomputedScaling::SameAsTokens => scale,
                };
                for i in 0..rows.nrows() {
                    let row = rows.row(i);
                    for j in 0..hidden {
                        h[[cursor, j]] = row[j] * row_scale;
                    }
                    cursor += 1;
                }
            }
        }
    }

    h
}

/// Multi-table summed embedding — one id column per table, rows summed.
///
/// The input algebra of multi-codebook speech models (MOSS-TTS-Realtime,
/// `docs/tts-funnel.md` §1.3): each sequence position carries one id per
/// table (text + one per RVQ codebook), and the position's embedding is
/// the **sum** of the looked-up rows. Contrast [`embed_plan`], which
/// row-*stacks* chunks: stacking extends the sequence, summing fuses
/// channels into one position.
///
/// No scaling is applied — the reference sums raw rows. Summation runs in
/// ascending table order so results are bit-stable against the reference,
/// which accumulates tables 0..C in order.
///
/// Panics (loudly, with context) on: no tables, id-column/table-count
/// mismatch, hidden-size mismatch between tables, or an id outside its
/// table — each of which is a caller bug, not a data condition.
pub fn embed_tables_sum(tables: &[ndarray::ArrayView2<f32>], ids: &Array2<u32>) -> Array2<f32> {
    assert!(!tables.is_empty(), "embed_tables_sum: no embedding tables");
    assert_eq!(
        tables.len(),
        ids.ncols(),
        "embed_tables_sum: {} tables but ids have {} columns",
        tables.len(),
        ids.ncols()
    );
    let hidden = tables[0].ncols();
    for (index, table) in tables.iter().enumerate() {
        assert_eq!(
            table.ncols(),
            hidden,
            "embed_tables_sum: table {index} hidden size {} != table 0 hidden size {hidden}",
            table.ncols()
        );
    }

    let mut h = Array2::<f32>::zeros((ids.nrows(), hidden));
    for position in 0..ids.nrows() {
        for (index, table) in tables.iter().enumerate() {
            let id = ids[[position, index]] as usize;
            assert!(
                id < table.nrows(),
                "embed_tables_sum: id {id} out of range for table {index} \
                 with {} rows (position {position})",
                table.nrows()
            );
            let row = table.row(id);
            for j in 0..hidden {
                h[[position, j]] += row[j];
            }
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_weights;

    #[test]
    fn embed_tokens_shape() {
        let weights = make_test_weights();
        let ids = [0u32, 1, 5];
        let out = embed_tokens_pub(&weights, &ids);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
    }

    #[test]
    fn embed_tokens_single() {
        let weights = make_test_weights();
        let out = embed_tokens_pub(&weights, &[0u32]);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embed_different_tokens_differ() {
        let weights = make_test_weights();
        let e0 = embed_tokens_pub(&weights, &[0u32]);
        let e1 = embed_tokens_pub(&weights, &[1u32]);
        let differ = e0.iter().zip(e1.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "different token ids should produce different embeddings"
        );
    }

    #[test]
    fn embed_same_token_is_deterministic() {
        let weights = make_test_weights();
        let a = embed_tokens_pub(&weights, &[3u32]);
        let b = embed_tokens_pub(&weights, &[3u32]);
        assert_eq!(a, b, "embedding should be deterministic");
    }

    #[test]
    fn embed_empty_token_list_returns_zero_rows() {
        let weights = make_test_weights();
        let out = embed_tokens_pub(&weights, &[]);
        assert_eq!(out.shape(), &[0, weights.hidden_size]);
    }

    #[test]
    fn embed_scaled_by_arch_embed_scale() {
        // TinyModel reports embed_scale = 1.0, so embedded row equals
        // the raw embed table row. Pin that contract for future
        // architectures that override the scale.
        let weights = make_test_weights();
        let out = embed_tokens_pub(&weights, &[2u32]);
        let scale = weights.arch.embed_scale_multiplier();
        let raw = weights.embed.row(2);
        for (j, v) in out.row(0).iter().enumerate() {
            assert!(
                (v - raw[j] * scale).abs() < 1e-6,
                "embed_tokens_pub did not apply arch.embed_scale() at col {j}"
            );
        }
    }

    // ─── Phase 0 bit-identity tests: embed_plan ⇄ embed_tokens_pub ──────────
    //
    // These pin the load-bearing contract of multi-modal Phase 0: routing a
    // text-only payload through `embed_plan(EmbeddingPlan::from_tokens(t))`
    // produces output exactly equal to `embed_tokens_pub(t)`. No epsilon,
    // no rtol — the bytes match. If any of these fail after a future
    // refactor, the entire forward path has silently changed behaviour.
    // See `docs/multi-modal.md` for why this is the Phase 0 acceptance
    // criterion.

    #[test]
    fn embed_plan_bit_identical_short_sequence() {
        let weights = make_test_weights();
        let ids = [0u32, 1, 2, 3, 4];
        let via_plan = embed_plan(&weights, &EmbeddingPlan::from_tokens(&ids));
        let via_pub = embed_tokens_pub(&weights, &ids);
        assert_eq!(via_plan, via_pub);
    }

    #[test]
    fn embed_plan_bit_identical_empty_sequence() {
        let weights = make_test_weights();
        let via_plan = embed_plan(&weights, &EmbeddingPlan::from_tokens(&[]));
        let via_pub = embed_tokens_pub(&weights, &[]);
        assert_eq!(via_plan, via_pub);
        assert_eq!(via_plan.shape(), &[0, weights.hidden_size]);
    }

    #[test]
    fn embed_plan_bit_identical_single_token() {
        let weights = make_test_weights();
        // make_test_weights produces a 32-row embed table, so token IDs
        // 0..32 are valid. Walk a spread of them.
        for tok in [0u32, 1, 7, 13, 31] {
            let via_plan = embed_plan(&weights, &EmbeddingPlan::from_tokens(&[tok]));
            let via_pub = embed_tokens_pub(&weights, &[tok]);
            assert_eq!(via_plan, via_pub, "drift at token {tok}");
        }
    }

    #[test]
    fn embed_plan_bit_identical_repeated_tokens() {
        // Repeated token IDs exercise the same embed-table row multiple
        // times; if the plan path accidentally caches or memoises, this
        // catches it.
        let weights = make_test_weights();
        let ids = [3u32, 3, 3, 1, 3, 1];
        let via_plan = embed_plan(&weights, &EmbeddingPlan::from_tokens(&ids));
        let via_pub = embed_tokens_pub(&weights, &ids);
        assert_eq!(via_plan, via_pub);
    }

    #[test]
    fn embed_plan_bit_identical_multi_chunk_text() {
        // Multiple Tokens chunks should concatenate to a result
        // bit-identical to embed_tokens_pub on the concatenated tokens.
        // This is the contract that lets multi-chunk plans build up the
        // text portion of a multi-modal sequence chunk-by-chunk.
        use super::super::embedding_plan::{EmbeddingChunk, PositionScheme};
        let weights = make_test_weights();
        let plan = EmbeddingPlan {
            chunks: vec![
                EmbeddingChunk::Tokens(vec![0, 1, 2]),
                EmbeddingChunk::Tokens(vec![3, 4]),
                EmbeddingChunk::Tokens(vec![5]),
            ],
            positions: PositionScheme::Sequential,
        };
        assert!(plan.is_text_only(), "fixture should hit the fast path");
        let via_plan = embed_plan(&weights, &plan);
        let via_pub = embed_tokens_pub(&weights, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(via_plan, via_pub);
    }

    // ─── Phase 1d-prep: multi-chunk plans actually exercise the mixed code path
    //
    // Until Phase 1d, every `embed_plan` call has been a single Tokens chunk
    // (or all-Tokens chunks, which still hits the fast path). These tests
    // build mixed Tokens/Precomputed plans, which is the FIRST exercise of
    // the cursor/row-stacking logic in the non-fast-path branch.
    //
    // If any of these fail, off-by-one bugs in the row-ordering machinery
    // would later get attributed to the encoder or connector. Better to fail
    // here, where there is no encoder to blame.

    #[test]
    fn embed_plan_mixed_chunks_preserve_row_order() {
        // Build [Tokens(N=2), Precomputed(M=3), Tokens(K=2)] with the
        // precomputed rows holding distinguishable sentinel values.
        // Assert:
        //   - total rows = N + M + K
        //   - rows 0..N == embed of first tokens
        //   - rows N..N+M == precomputed rows verbatim (PrecomputedScaling::None)
        //   - rows N+M..N+M+K == embed of last tokens
        use super::super::embedding_plan::{EmbeddingChunk, PositionScheme};
        use larql_models::Modality;

        let weights = make_test_weights();
        let hidden = weights.hidden_size;

        let first_tokens = vec![1u32, 2];
        let last_tokens = vec![4u32, 5];

        // Distinguishable per-row sentinel values that won't collide
        // with anything from the embed table.
        let m: usize = 3;
        let mut precomputed = Array2::<f32>::zeros((m, hidden));
        for r in 0..m {
            for c in 0..hidden {
                // base 100.0 + per-row tag * 10 + per-col fraction —
                // distinct from embed table values and per (r, c).
                precomputed[[r, c]] = 100.0 + (r as f32) * 10.0 + (c as f32) * 0.1;
            }
        }
        let sentinel = precomputed.clone();

        let plan = EmbeddingPlan {
            chunks: vec![
                EmbeddingChunk::Tokens(first_tokens.clone()),
                EmbeddingChunk::Precomputed {
                    rows: precomputed,
                    modality: Modality::Image,
                },
                EmbeddingChunk::Tokens(last_tokens.clone()),
            ],
            positions: PositionScheme::Sequential,
        };
        assert!(
            !plan.is_text_only(),
            "fixture must force the mixed code path, not the fast path"
        );

        let out = embed_plan(&weights, &plan);
        let n = first_tokens.len();
        let k = last_tokens.len();
        assert_eq!(
            out.shape(),
            &[n + m + k, hidden],
            "total rows should be sum of chunk row counts"
        );

        // Rows 0..N: first tokens, embed-table lookup with embed_scale.
        let first_embed = embed_tokens_pub(&weights, &first_tokens);
        for r in 0..n {
            for c in 0..hidden {
                assert_eq!(
                    out[[r, c]],
                    first_embed[[r, c]],
                    "first-Tokens-chunk drift at row {r} col {c}"
                );
            }
        }

        // Rows N..N+M: precomputed sentinel rows verbatim.
        // make_test_weights's arch returns multimodal()=None (TinyModel),
        // so PrecomputedScaling defaults to None → rows go in as-is.
        for r in 0..m {
            for c in 0..hidden {
                assert_eq!(
                    out[[n + r, c]],
                    sentinel[[r, c]],
                    "precomputed row drift at chunk row {r} col {c}"
                );
            }
        }

        // Rows N+M..N+M+K: last tokens, embed-table lookup.
        let last_embed = embed_tokens_pub(&weights, &last_tokens);
        for r in 0..k {
            for c in 0..hidden {
                assert_eq!(
                    out[[n + m + r, c]],
                    last_embed[[r, c]],
                    "last-Tokens-chunk drift at row {r} col {c}"
                );
            }
        }
    }

    #[test]
    fn embed_plan_precomputed_only_passes_rows_through() {
        // Pure precomputed plan (no Tokens chunks) — exercises the mixed
        // path with the cursor never touching the embed table.
        use super::super::embedding_plan::{EmbeddingChunk, PositionScheme};
        use larql_models::Modality;

        let weights = make_test_weights();
        let hidden = weights.hidden_size;
        let rows = Array2::<f32>::from_shape_fn((4, hidden), |(r, c)| {
            42.0 + (r as f32) + (c as f32) * 0.01
        });
        let sentinel = rows.clone();
        let plan = EmbeddingPlan {
            chunks: vec![EmbeddingChunk::Precomputed {
                rows,
                modality: Modality::Image,
            }],
            positions: PositionScheme::Sequential,
        };
        let out = embed_plan(&weights, &plan);
        assert_eq!(out.shape(), &[4, hidden]);
        assert_eq!(
            out, sentinel,
            "precomputed-only plan must pass rows through verbatim"
        );
    }

    // ─── embed_tables_sum: the multi-codebook input algebra ──────────────

    fn two_tables() -> (Array2<f32>, Array2<f32>) {
        // Distinguishable values: table 0 rows carry 10*row, table 1 rows
        // carry 0.1*row, so any sum uniquely identifies its inputs.
        let a = Array2::from_shape_fn((4, 3), |(r, _)| 10.0 * r as f32);
        let b = Array2::from_shape_fn((5, 3), |(r, _)| 0.1 * r as f32);
        (a, b)
    }

    #[test]
    fn tables_sum_sums_one_row_per_column() {
        let (a, b) = two_tables();
        let ids = ndarray::arr2(&[[1u32, 2], [3, 0]]);
        let out = embed_tables_sum(&[a.view(), b.view()], &ids);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out[[0, 0]], 10.0 + 0.2);
        assert_eq!(out[[1, 0]], 30.0 + 0.0);
    }

    #[test]
    fn tables_sum_single_table_is_plain_lookup() {
        let (a, _) = two_tables();
        let ids = ndarray::arr2(&[[2u32], [0]]);
        let out = embed_tables_sum(&[a.view()], &ids);
        assert_eq!(out[[0, 0]], 20.0);
        assert_eq!(out[[1, 0]], 0.0);
    }

    #[test]
    fn tables_sum_zero_rows() {
        let (a, b) = two_tables();
        let ids = Array2::<u32>::zeros((0, 2));
        let out = embed_tables_sum(&[a.view(), b.view()], &ids);
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn tables_sum_matches_manual_accumulation_order() {
        // Bit-stability contract: identical to a manual ascending-order sum.
        let (a, b) = two_tables();
        let ids = ndarray::arr2(&[[3u32, 4]]);
        let out = embed_tables_sum(&[a.view(), b.view()], &ids);
        for j in 0..3 {
            let manual = a[[3, j]] + b[[4, j]];
            assert_eq!(out[[0, j]], manual);
        }
    }

    #[test]
    #[should_panic(expected = "ids have")]
    fn tables_sum_rejects_column_mismatch() {
        let (a, b) = two_tables();
        let ids = ndarray::arr2(&[[1u32, 2, 3]]);
        embed_tables_sum(&[a.view(), b.view()], &ids);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn tables_sum_rejects_out_of_range_id() {
        let (a, b) = two_tables();
        let ids = ndarray::arr2(&[[9u32, 0]]);
        embed_tables_sum(&[a.view(), b.view()], &ids);
    }

    #[test]
    #[should_panic(expected = "hidden size")]
    fn tables_sum_rejects_hidden_mismatch() {
        let (a, _) = two_tables();
        let narrow = Array2::<f32>::zeros((4, 2));
        let ids = ndarray::arr2(&[[0u32, 0]]);
        embed_tables_sum(&[a.view(), narrow.view()], &ids);
    }

    #[test]
    fn embed_plan_precomputed_chunk_at_position_zero() {
        // Image-first prefix-only layout that Phase 1d will actually produce:
        // [Precomputed(image rows), Tokens(text)]. Confirms row ordering
        // when the precomputed chunk leads.
        use super::super::embedding_plan::{EmbeddingChunk, PositionScheme};
        use larql_models::Modality;

        let weights = make_test_weights();
        let hidden = weights.hidden_size;
        let m: usize = 2;
        let img = Array2::<f32>::from_shape_fn((m, hidden), |(r, c)| {
            7.0 + (r as f32) * 3.0 + (c as f32) * 0.5
        });
        let img_sentinel = img.clone();
        let text_tokens = vec![0u32, 1, 2];

        let plan = EmbeddingPlan {
            chunks: vec![
                EmbeddingChunk::Precomputed {
                    rows: img,
                    modality: Modality::Image,
                },
                EmbeddingChunk::Tokens(text_tokens.clone()),
            ],
            positions: PositionScheme::Sequential,
        };
        let out = embed_plan(&weights, &plan);
        assert_eq!(out.shape(), &[m + text_tokens.len(), hidden]);

        // Rows 0..m are the image.
        for r in 0..m {
            for c in 0..hidden {
                assert_eq!(out[[r, c]], img_sentinel[[r, c]]);
            }
        }
        // Rows m..m+text_len are the tokens.
        let text_embed = embed_tokens_pub(&weights, &text_tokens);
        for r in 0..text_tokens.len() {
            for c in 0..hidden {
                assert_eq!(out[[m + r, c]], text_embed[[r, c]]);
            }
        }
    }
}
