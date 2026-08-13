//! MAP-2b: exact-boundary tile swap (follow-up to MAP-2).
//!
//! Registered as chuk-experiments programme `map`, experiment `map-2`
//! (recorded as this experiment's `next_action` after the precision
//! correction to MAP-2's original conclusion).
//!
//! MAP-2 (`map2_tile_swap_matrix.rs`) spliced a foreign layer's FFN delta
//! directly into the residual stream, bypassing the model's own
//! per-layer norm/PLE/scalar machinery at the splice point — even its
//! `m == l` "native" reconstruction diverged hugely from an unmodified
//! forward (KL=13.17, true top token demoted to rank 7665). That
//! confound made "swap ≈ random" ambiguous: both arms could have simply
//! been landing in the same broad off-manifold catastrophic regime
//! rather than genuinely tying.
//!
//! MAP-2b removes that confound structurally rather than patching around
//! it: instead of overwriting the post-layer residual, it substitutes
//! WHICH LAYER'S WEIGHTS get used *inside* the model's own `FfnBackend`
//! plug point (`TileSwapFfn`/`RandomDeltaFfn` below). The library's real
//! per-layer dispatch (`larql_compute::forward::layer::run_ffn`) still
//! applies the target layer's own pre-FFN norm, post-FFN norm, PLE, and
//! layer scalar exactly as it would for an unmodified forward — only the
//! gate/up/down matmul weights differ. This has two consequences that
//! directly answer MAP-2's methodological gap:
//!
//! - `foreign_layer == target_layer` is now **bit-identical** to an
//!   unmodified forward (same code path, same weights — not "close",
//!   literally identical), not a reconstruction with its own error floor.
//! - It also happens to fix a SEPARATE bug MAP-1 and MAP-2 both had:
//!   both fed the raw, un-normalized post-attention residual directly
//!   into their hand-rolled FFN math, when the real per-layer dispatch
//!   normalizes it first (`run_ffn` in `layer.rs`: `apply_norm` before
//!   calling `ffn.forward`). That single omission was likely the
//!   dominant contributor to MAP-2's 13.17-KL abstraction floor — Gemma's
//!   gate/up weights are trained on normalized input, and feeding them
//!   raw residual scale produces a very different intermediate
//!   activation regardless of which layer's tile is used. It didn't
//!   affect MAP-1's conclusion (permutation-equivariance holds for ANY
//!   consistently-defined function, norm-included or not), but it does
//!   matter here where the goal is matching the model's real behavior.
//!
//! Design (per MAP-2's `next_action`, itself following the reviewer's
//! spec):
//! 1. Adjacent-layer pairs only (`l <-> l+1`), not distant offsets —
//!    MAP-2 found no distance-dependent pattern, so this concentrates
//!    replication where locality would be easiest to detect if it exists.
//! 2. 8 diverse prompts (the lower end of the reviewer's 8-16 range —
//!    scaling to 16 is a one-line const change once this pilot runs).
//!    Prompt diversity, not pair count, was MAP-2's real replication gap.
//! 3. Two swap arms per cell: raw foreign-tile delta, and the same delta
//!    rescaled to the native delta's per-row L2 norm.
//! 4. Three random directions per cell (not one), norm-matched to native,
//!    averaged — reduces the random control's own sampling noise.
//! 5. Metrics: `rank_of_native_top1` (primary, non-saturating — see
//!    MAP-2's own gotcha), KL (context only), centered-logit cosine
//!    (mean-centered raw logits via `hidden_to_raw_logits`, a shape
//!    metric insensitive to a constant offset), and a downstream
//!    recoverability check (does the near-swap rank recover by the
//!    final layer?). Statistics are computed PER PROMPT first (mean rank
//!    across that prompt's cells), then compared paired ACROSS prompts —
//!    prompt is the replication unit, layer-pair is a repeated measure,
//!    not "14 x 8 = 112 independent trials."
//!
//! Usage: `LARQL_MAP2B_VINDEX=/path/to/model.vindex cargo run --release -p larql-inference --example map2b_exact_boundary_tile_swap`

use ndarray::{Array1, Array2, Axis};

use larql_inference::ffn::WeightFfn;
use larql_inference::forward::{logit_lens_topk, trace_forward_full_hooked, RecordHook};
use larql_inference::{hidden_to_raw_logits, FfnBackend};
use larql_models::{ModelWeights, WeightsView};

fn dense_ffn(weights: &ModelWeights, layer: usize, x: &Array2<f32>) -> Array2<f32> {
    larql_compute::ffn::dense_ffn_forward(WeightsView::dense(weights), layer, x).0
}

fn row_norms(a: &Array2<f32>) -> Array1<f32> {
    a.map_axis(Axis(1), |row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
}

fn rescale_rows_to_match(source: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    let target_norms = row_norms(target);
    let source_norms = row_norms(source);
    let mut out = source.clone();
    for (mut row, (&sn, &tn)) in out
        .axis_iter_mut(Axis(0))
        .zip(source_norms.iter().zip(target_norms.iter()))
    {
        if sn > 1e-12 {
            row.mapv_inplace(|v| v * (tn / sn));
        }
    }
    out
}

fn random_matrix_matching_row_norms(target: &Array2<f32>, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let mut next_f32 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let raw = Array2::from_shape_fn(target.dim(), |_| next_f32());
    rescale_rows_to_match(&raw, target)
}

/// Substitutes `foreign_layer`'s gate/up/down weights for `target_layer`'s
/// own, leaving every surrounding mechanism (pre/post-FFN norm, PLE,
/// residual add, layer scalar — all applied by the real per-layer
/// dispatch, `larql_compute::forward::layer::run_ffn`, around whatever
/// this returns) completely untouched. `x` here is ALREADY the correctly
/// normed pre-FFN tensor — `run_ffn` applies the norm before calling this.
struct TileSwapFfn<'a> {
    weights: &'a ModelWeights,
    target_layer: usize,
    foreign_layer: usize,
    rescale_to_native_norm: bool,
}

impl FfnBackend for TileSwapFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        if layer != self.target_layer {
            return dense_ffn(self.weights, layer, x);
        }
        let foreign = dense_ffn(self.weights, self.foreign_layer, x);
        if self.rescale_to_native_norm {
            let native = dense_ffn(self.weights, self.target_layer, x);
            rescale_rows_to_match(&foreign, &native)
        } else {
            foreign
        }
    }
    fn name(&self) -> &str {
        "tile-swap"
    }
}

/// Same substitution point as `TileSwapFfn`, but the target layer's
/// contribution is a norm-matched random direction instead of any real
/// tile's output — the noise-floor control.
struct RandomDeltaFfn<'a> {
    weights: &'a ModelWeights,
    target_layer: usize,
    seed: u64,
}

impl FfnBackend for RandomDeltaFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        if layer != self.target_layer {
            return dense_ffn(self.weights, layer, x);
        }
        let native = dense_ffn(self.weights, self.target_layer, x);
        random_matrix_matching_row_norms(&native, self.seed)
    }
    fn name(&self) -> &str {
        "random-delta"
    }
}

fn dense_from_sorted(sorted: &[(u32, f32)], vocab: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; vocab];
    for &(id, p) in sorted {
        dense[id as usize] = p;
    }
    dense
}

fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    const EPS: f32 = 1e-30;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else {
                pi * (pi.max(EPS) / qi.max(EPS)).ln()
            }
        })
        .sum()
}

fn rank_of(sorted: &[(u32, f32)], token_id: u32) -> usize {
    sorted
        .iter()
        .position(|&(id, _)| id == token_id)
        .unwrap_or(sorted.len())
}

fn centered_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mean_a = a.iter().sum::<f32>() / a.len() as f32;
    let mean_b = b.iter().sum::<f32>() / b.len() as f32;
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (cx, cy) = (x - mean_a, y - mean_b);
        dot += cx * cy;
        na += cx * cx;
        nb += cy * cy;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Runs a forward pass with the given FFN backend, returning:
/// - the full sorted (descending) final-layer distribution
/// - the sorted distribution at `near_layer` (for the recoverability check)
/// - raw final-layer logits (for centered-logit cosine)
struct PassResult {
    final_sorted: Vec<(u32, f32)>,
    near_sorted: Vec<(u32, f32)>,
    final_logits: Vec<f32>,
}

fn run_pass(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    prompt: &[u32],
    near_layer: usize,
    last_layer: usize,
) -> PassResult {
    let layers = if near_layer == last_layer {
        vec![last_layer]
    } else {
        vec![near_layer, last_layer]
    };
    let mut record = RecordHook::for_layers(layers.iter().copied());
    let _ = trace_forward_full_hooked(weights, prompt, &layers, false, 0, false, ffn, &mut record);
    let near = record
        .post_layer
        .get(&near_layer)
        .unwrap_or_else(|| record.post_layer.get(&last_layer).unwrap());
    let last = record.post_layer.get(&last_layer).unwrap();
    let near_row = near.row(near.nrows() - 1).to_vec();
    let last_row = last.row(last.nrows() - 1).to_vec();
    PassResult {
        final_sorted: logit_lens_topk(weights, &last_row, weights.vocab_size),
        near_sorted: logit_lens_topk(weights, &near_row, weights.vocab_size),
        final_logits: hidden_to_raw_logits(
            weights,
            &last.row(last.nrows() - 1).to_owned().insert_axis(Axis(0)),
        ),
    }
}

struct CellMetrics {
    from_layer: usize,
    tile_layer: usize,
    kl_raw: f32,
    kl_rescaled: f32,
    kl_random: f32,
    rank_raw: usize,
    rank_rescaled: usize,
    rank_random: usize,
    centered_cosine_raw: f32,
    centered_cosine_rescaled: f32,
    centered_cosine_random: f32,
    /// near-layer rank minus final-layer rank, for the RAW swap arm —
    /// positive means later layers partially recovered the native answer.
    recoverability_raw: i64,
    recoverability_random: i64,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_cell(
    weights: &ModelWeights,
    prompt: &[u32],
    from_layer: usize,
    tile_layer: usize,
    near_layer: usize,
    last_layer: usize,
    p_native_final: &[(u32, f32)],
    p_native_near: &[(u32, f32)],
    native_final_logits: &[f32],
    seeds: [u64; 3],
) -> CellMetrics {
    let vocab = weights.vocab_size;
    let native_top1 = p_native_final[0].0;
    let dense_native = dense_from_sorted(p_native_final, vocab);

    let raw = run_pass(
        weights,
        &TileSwapFfn {
            weights,
            target_layer: from_layer,
            foreign_layer: tile_layer,
            rescale_to_native_norm: false,
        },
        prompt,
        near_layer,
        last_layer,
    );
    let rescaled = run_pass(
        weights,
        &TileSwapFfn {
            weights,
            target_layer: from_layer,
            foreign_layer: tile_layer,
            rescale_to_native_norm: true,
        },
        prompt,
        near_layer,
        last_layer,
    );

    let random_passes: Vec<PassResult> = seeds
        .iter()
        .map(|&s| {
            run_pass(
                weights,
                &RandomDeltaFfn {
                    weights,
                    target_layer: from_layer,
                    seed: s,
                },
                prompt,
                near_layer,
                last_layer,
            )
        })
        .collect();
    let mean_kl_random: f32 = random_passes
        .iter()
        .map(|p| kl_divergence(&dense_native, &dense_from_sorted(&p.final_sorted, vocab)))
        .sum::<f32>()
        / random_passes.len() as f32;
    let mean_rank_random: f64 = random_passes
        .iter()
        .map(|p| rank_of(&p.final_sorted, native_top1) as f64)
        .sum::<f64>()
        / random_passes.len() as f64;
    let mean_cos_random: f32 = random_passes
        .iter()
        .map(|p| centered_cosine(native_final_logits, &p.final_logits))
        .sum::<f32>()
        / random_passes.len() as f32;
    let mean_near_rank_random: f64 = random_passes
        .iter()
        .map(|p| rank_of(&p.near_sorted, native_top1) as f64)
        .sum::<f64>()
        / random_passes.len() as f64;
    let mean_final_rank_random: f64 = random_passes
        .iter()
        .map(|p| rank_of(&p.final_sorted, native_top1) as f64)
        .sum::<f64>()
        / random_passes.len() as f64;
    let _ = p_native_near; // near-layer native distribution kept for parity/debuggability, not directly scored

    CellMetrics {
        from_layer,
        tile_layer,
        kl_raw: kl_divergence(&dense_native, &dense_from_sorted(&raw.final_sorted, vocab)),
        kl_rescaled: kl_divergence(
            &dense_native,
            &dense_from_sorted(&rescaled.final_sorted, vocab),
        ),
        kl_random: mean_kl_random,
        rank_raw: rank_of(&raw.final_sorted, native_top1),
        rank_rescaled: rank_of(&rescaled.final_sorted, native_top1),
        rank_random: mean_rank_random.round() as usize,
        centered_cosine_raw: centered_cosine(native_final_logits, &raw.final_logits),
        centered_cosine_rescaled: centered_cosine(native_final_logits, &rescaled.final_logits),
        centered_cosine_random: mean_cos_random,
        recoverability_raw: rank_of(&raw.near_sorted, native_top1) as i64
            - rank_of(&raw.final_sorted, native_top1) as i64,
        recoverability_random: (mean_near_rank_random - mean_final_rank_random).round() as i64,
    }
}

fn print_cell(c: &CellMetrics) {
    println!(
        "  L{:>2}->tile L{:>2}:  rank raw={:>7} rescaled={:>7} random(x3)={:>7}  KL raw={:>8.4} rescaled={:>8.4} random={:>8.4}  cos raw={:>7.4} rescaled={:>7.4} random={:>7.4}  recover raw={:>6} random={:>6}",
        c.from_layer, c.tile_layer, c.rank_raw, c.rank_rescaled, c.rank_random,
        c.kl_raw, c.kl_rescaled, c.kl_random,
        c.centered_cosine_raw, c.centered_cosine_rescaled, c.centered_cosine_random,
        c.recoverability_raw, c.recoverability_random
    );
}

const ADJACENT_HOME_LAYERS: [usize; 7] = [4, 8, 12, 16, 20, 24, 28];

const PROMPTS: [&str; 8] = [
    "The capital of France is",
    "The capital of Japan is",
    "Two plus two equals",
    "The chemical symbol for water is",
    "Once upon a time there was a",
    "The largest planet in the solar system is",
    "The president of the United States is",
    "She walked into the room and saw",
];

fn build_adjacent_pairs(num_layers: usize) -> Vec<(usize, usize)> {
    ADJACENT_HOME_LAYERS
        .iter()
        .filter(|&&l| l + 1 < num_layers)
        .map(|&l| (l, l + 1))
        .collect()
}

struct PromptSummary {
    mean_rank_raw: f64,
    mean_rank_rescaled: f64,
    mean_rank_random: f64,
}

fn run_prompt(
    weights: &ModelWeights,
    ffn: &WeightFfn,
    tokenizer: &tokenizers::Tokenizer,
    prompt_text: &str,
    pairs: &[(usize, usize)],
    last_layer: usize,
    seed_base: u64,
) -> PromptSummary {
    let prompt = larql_inference::encode_prompt(tokenizer, &*weights.arch, prompt_text)
        .expect("tokenize prompt");
    println!("\n--- prompt: {prompt_text:?} ---");

    // One TRUE, bit-identical native pass per touched layer (this IS the
    // unmodified forward — no reconstruction, no abstraction floor).
    let mut touched: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    touched.sort_unstable();
    touched.dedup();
    let mut native_final: std::collections::HashMap<usize, Vec<(u32, f32)>> =
        std::collections::HashMap::new();
    let mut native_near: std::collections::HashMap<usize, Vec<(u32, f32)>> =
        std::collections::HashMap::new();
    let mut native_final_logits: std::collections::HashMap<usize, Vec<f32>> =
        std::collections::HashMap::new();
    for &l in &touched {
        let near_layer = (l + 2).min(last_layer);
        let pass = run_pass(weights, ffn, &prompt, near_layer, last_layer);
        native_final.insert(l, pass.final_sorted);
        native_near.insert(l, pass.near_sorted);
        native_final_logits.insert(l, pass.final_logits);
    }

    let mut cells = Vec::new();
    let mut seed = seed_base;
    for &(a, b) in pairs {
        for &(from, to) in &[(a, b), (b, a)] {
            let near_layer = (from + 2).min(last_layer);
            let cell = evaluate_cell(
                weights,
                &prompt,
                from,
                to,
                near_layer,
                last_layer,
                &native_final[&from],
                &native_near[&from],
                &native_final_logits[&from],
                [seed, seed + 1, seed + 2],
            );
            print_cell(&cell);
            cells.push(cell);
            seed += 3;
        }
    }

    let n = cells.len() as f64;
    PromptSummary {
        mean_rank_raw: cells.iter().map(|c| c.rank_raw as f64).sum::<f64>() / n,
        mean_rank_rescaled: cells.iter().map(|c| c.rank_rescaled as f64).sum::<f64>() / n,
        mean_rank_random: cells.iter().map(|c| c.rank_random as f64).sum::<f64>() / n,
    }
}

fn main() {
    println!("=== MAP-2b: exact-boundary tile swap ===");

    let vindex_path = match std::env::var("LARQL_MAP2B_VINDEX") {
        Ok(p) => p,
        Err(_) => {
            println!("skipped — set LARQL_MAP2B_VINDEX=/path/to/model.vindex to run (needs real depth, no synthetic-fixture phase)");
            return;
        }
    };
    println!("loading real vindex: {vindex_path}");
    let dir = std::path::Path::new(&vindex_path);
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_with_opts(
        dir,
        &mut cb,
        larql_vindex::LoadWeightsOptions::default(),
    )
    .expect("load real vindex weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(dir).expect("load vindex tokenizer");
    let ffn = WeightFfn { weights: &weights };
    let last_layer = weights.num_layers - 1;
    let pairs = build_adjacent_pairs(weights.num_layers);
    println!(
        "{} adjacent pairs ({} directed cells/prompt) x {} prompts\n",
        pairs.len(),
        pairs.len() * 2,
        PROMPTS.len()
    );

    // Bit-identical sanity: m==l via TileSwapFfn must match the plain
    // unmodified forward exactly (not "close" — this is the whole point).
    {
        let probe_prompt = larql_inference::encode_prompt(&tokenizer, &*weights.arch, PROMPTS[0])
            .expect("tokenize");
        let l = pairs[0].0;
        let plain = run_pass(&weights, &ffn, &probe_prompt, l, last_layer);
        let identity_swap = TileSwapFfn {
            weights: &weights,
            target_layer: l,
            foreign_layer: l,
            rescale_to_native_norm: false,
        };
        let via_wrapper = run_pass(&weights, &identity_swap, &probe_prompt, l, last_layer);
        let max_diff: f32 = plain
            .final_logits
            .iter()
            .zip(via_wrapper.final_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("sanity: TileSwapFfn(m==l) vs unmodified forward, max logit |Δ| = {max_diff:.3e} (must be exactly 0.0 — same code path, same weights)\n");
        assert_eq!(
            max_diff, 0.0,
            "m==l must be bit-identical to an unmodified forward — this is MAP-2b's entire point"
        );
    }

    let mut summaries = Vec::new();
    for (i, prompt_text) in PROMPTS.iter().enumerate() {
        let summary = run_prompt(
            &weights,
            &ffn,
            &tokenizer,
            prompt_text,
            &pairs,
            last_layer,
            10_000 + i as u64 * 100,
        );
        summaries.push(summary);
    }

    println!("\n=== per-prompt summary (mean rank of native top-1 across all cells) ===");
    let mut raw_beats_random = 0;
    let mut random_beats_raw = 0;
    let mut rescaled_beats_random = 0;
    let mut random_beats_rescaled = 0;
    for (text, s) in PROMPTS.iter().zip(&summaries) {
        println!(
            "  {text:?}: raw={:.1} rescaled={:.1} random={:.1}",
            s.mean_rank_raw, s.mean_rank_rescaled, s.mean_rank_random
        );
        if s.mean_rank_raw < s.mean_rank_random {
            raw_beats_random += 1
        } else if s.mean_rank_random < s.mean_rank_raw {
            random_beats_raw += 1
        }
        if s.mean_rank_rescaled < s.mean_rank_random {
            rescaled_beats_random += 1
        } else if s.mean_rank_random < s.mean_rank_rescaled {
            random_beats_rescaled += 1
        }
    }

    println!(
        "\n=== decisive comparison: PROMPT is the replication unit (n={}) ===",
        summaries.len()
    );
    println!(
        "raw swap beats random:      {raw_beats_random}/{}",
        summaries.len()
    );
    println!(
        "random beats raw swap:      {random_beats_raw}/{}",
        summaries.len()
    );
    println!(
        "rescaled swap beats random: {rescaled_beats_random}/{}",
        summaries.len()
    );
    println!(
        "random beats rescaled swap: {random_beats_rescaled}/{}",
        summaries.len()
    );
    println!(
        "\nThis is a genuine n={}-independent-trial comparison (one prompt = one trial), not {} dependent\n\
         per-cell counts. A lopsided split here (e.g. 7/8 or 8/8 either direction) is real evidence;\n\
         a near-even split confirms MAP-2's finding under an abstraction that no longer has an\n\
         off-manifold confound to hide behind.",
        summaries.len(), summaries.len() * pairs.len() * 2
    );
}
