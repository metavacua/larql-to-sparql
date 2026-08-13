//! MAP-5c: is the planets pair's continuation anomaly a layer-depth issue?
//!
//! MAP-5b found that for the planets pair (only, of 5 tested), neither
//! one-shot NOR repeated residual teleportation reaches the donor's exact
//! token at the one real discriminating branch point (step 3) — both land
//! one rank off. This is unlike every other pair, where REPEATED patching
//! reliably reaches rank 0 exactly, since it re-injects the donor's own
//! true captured residual at that exact step. That planets was the one
//! pair patched at the shallower L16 (the earliest layer MAP-5 found
//! sufficient for planets' SINGLE-token recovery) rather than L24 (used
//! for the other four pairs) suggests "layer sufficient for the immediate
//! token" and "layer sufficient for durable branch-token construction"
//! may be different thresholds. This sweeps L16/L20/L24/L28 for the
//! planets pair only, in both one-shot and repeated mode, to test that
//! directly.
//!
//! Reporting applies the discrimination-aware analysis from MAP-5b's own
//! correction: a step only counts as evidence if the recipient's
//! unpatched baseline would have differed from the donor there (baseline
//! KL vs donor > 1.0 bit) — matching a non-discriminating boilerplate
//! step proves nothing.
//!
//! Usage: `LARQL_MAP5_VINDEX=/path/to/model.vindex cargo run --release -p larql-inference --example map5c_planets_layer_sweep`

use std::collections::HashMap;

use larql_inference::ffn::WeightFfn;
use larql_inference::forward::patching::{patch_and_trace_with_ffn, DonorState};
use larql_inference::forward::{logit_lens_topk, trace_forward_full};
use larql_inference::FfnBackend;
use larql_models::ModelWeights;

const N_STEPS: usize = 5;
const LAYER_SWEEP: [usize; 4] = [16, 20, 24, 28];
const DISCRIMINATING_KL_THRESHOLD: f32 = 1.0;

const DONOR_TEXT: &str = "The largest planet in the solar system is";
const RECIPIENT_TEXT: &str = "The smallest planet in the solar system is";

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

fn dense_from_sorted(sorted: &[(u32, f32)], vocab: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; vocab];
    for &(id, p) in sorted {
        dense[id as usize] = p;
    }
    dense
}

fn rank_of(sorted: &[(u32, f32)], token_id: u32) -> usize {
    sorted
        .iter()
        .position(|&(id, _)| id == token_id)
        .unwrap_or(sorted.len())
}

fn plain_step(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    tokens: &[u32],
    capture_layer: usize,
    last_layer: usize,
) -> (u32, Vec<(u32, f32)>, Vec<f32>) {
    let layers = if capture_layer == last_layer {
        vec![last_layer]
    } else {
        vec![capture_layer, last_layer]
    };
    let trace = trace_forward_full(weights, tokens, &layers, false, 0, false, ffn);
    let capture_row = trace
        .residuals
        .iter()
        .find(|(l, _)| *l == capture_layer)
        .unwrap()
        .1
        .clone();
    let final_row = trace
        .residuals
        .iter()
        .find(|(l, _)| *l == last_layer)
        .unwrap()
        .1
        .clone();
    let dist = logit_lens_topk(weights, &final_row, weights.vocab_size);
    (dist[0].0, dist, capture_row)
}

fn patched_step(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    tokens: &[u32],
    patch_layer: usize,
    patch_row: &[f32],
    last_layer: usize,
) -> (u32, Vec<(u32, f32)>) {
    let pos = tokens.len() - 1;
    let mut records = HashMap::new();
    records.insert((patch_layer, pos), patch_row.to_vec());
    let donor = DonorState { records };
    let trace = patch_and_trace_with_ffn(weights, tokens, &donor, &[last_layer], ffn);
    let final_row = &trace
        .residuals
        .iter()
        .find(|(l, _)| *l == last_layer)
        .unwrap()
        .1;
    let dist = logit_lens_topk(weights, final_row, weights.vocab_size);
    (dist[0].0, dist)
}

/// What one reference run yields: the generated sequence (prompt + new
/// tokens), the residual captured at `capture_layer` per generated position,
/// and the full sorted distribution at each step.
type ReferenceRun = (Vec<u32>, Vec<Vec<f32>>, Vec<Vec<(u32, f32)>>);

fn run_reference(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    prompt_tokens: &[u32],
    capture_layer: usize,
    last_layer: usize,
    n_steps: usize,
) -> ReferenceRun {
    let mut seq = prompt_tokens.to_vec();
    let mut rows = Vec::with_capacity(n_steps);
    let mut dists = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let (next, dist, row) = plain_step(weights, ffn, &seq, capture_layer, last_layer);
        rows.push(row);
        dists.push(dist);
        seq.push(next);
    }
    (seq, rows, dists)
}

struct StepScore {
    token: u32,
    kl_vs_donor: f32,
    rank_of_donor_token: usize,
}

fn score_step(
    dist: &[(u32, f32)],
    donor_dist: &[(u32, f32)],
    donor_token: u32,
    vocab: usize,
) -> StepScore {
    StepScore {
        token: dist[0].0,
        kl_vs_donor: kl_divergence(
            &dense_from_sorted(donor_dist, vocab),
            &dense_from_sorted(dist, vocab),
        ),
        rank_of_donor_token: rank_of(dist, donor_token),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_condition(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    recipient_prompt: &[u32],
    patch_layer: usize,
    last_layer: usize,
    n_steps: usize,
    donor_dists: &[Vec<(u32, f32)>],
    donor_tokens: &[u32],
    repeated_rows: Option<&[Vec<f32>]>,
    one_shot_row: Option<&[f32]>,
) -> Vec<StepScore> {
    let mut seq = recipient_prompt.to_vec();
    let mut scores = Vec::with_capacity(n_steps);
    let vocab = weights.vocab_size;
    for step in 0..n_steps {
        let dist = if let Some(rows) = repeated_rows {
            patched_step(weights, ffn, &seq, patch_layer, &rows[step], last_layer).1
        } else if step == 0 {
            if let Some(row) = one_shot_row {
                patched_step(weights, ffn, &seq, patch_layer, row, last_layer).1
            } else {
                plain_step(weights, ffn, &seq, patch_layer, last_layer).1
            }
        } else {
            plain_step(weights, ffn, &seq, patch_layer, last_layer).1
        };
        let score = score_step(&dist, &donor_dists[step], donor_tokens[step], vocab);
        seq.push(score.token);
        scores.push(score);
    }
    scores
}

fn print_scores(
    label: &str,
    scores: &[StepScore],
    discriminating: &[bool],
    tokenizer: &tokenizers::Tokenizer,
) {
    print!("    {label:>20}: ");
    let mut all_discriminating_won = true;
    for (s, &disc) in scores.iter().zip(discriminating) {
        let text = tokenizer.decode(&[s.token], false).unwrap_or_default();
        let won = s.rank_of_donor_token == 0;
        if disc && !won {
            all_discriminating_won = false;
        }
        let marker = if disc {
            if won {
                "OK"
            } else {
                "MISS"
            }
        } else {
            "  "
        };
        print!(
            "[{text:?} kl={:.2} rank_d={:>3} {marker:>4}] ",
            s.kl_vs_donor, s.rank_of_donor_token
        );
    }
    let any_disc = discriminating.iter().any(|&d| d);
    println!(
        " -> {}",
        if !any_disc {
            "no discriminating steps".to_string()
        } else if all_discriminating_won {
            "WINS every discriminating step".to_string()
        } else {
            "misses at least one discriminating step".to_string()
        }
    );
}

fn main() {
    println!("=== MAP-5c: planets pair, layer sweep (L16/L20/L24/L28), one-shot + repeated ===");
    let vindex_path = match std::env::var("LARQL_MAP5_VINDEX") {
        Ok(p) => p,
        Err(_) => {
            println!("skipped — set LARQL_MAP5_VINDEX=/path/to/model.vindex to run");
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

    let donor_tokens = larql_inference::encode_prompt(&tokenizer, &*weights.arch, DONOR_TEXT)
        .expect("tokenize donor");
    let recipient_tokens =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, RECIPIENT_TEXT)
            .expect("tokenize recipient");

    // donor's generated tokens/distributions are layer-independent (always argmax of the
    // final-layer distribution) — compute once.
    let (donor_seq, _donor_rows_unused, donor_dists) = run_reference(
        &weights,
        &ffn,
        &donor_tokens,
        last_layer,
        last_layer,
        N_STEPS,
    );
    let donor_generated_tokens: Vec<u32> = donor_seq[donor_tokens.len()..].to_vec();
    println!(
        "donor reference continuation: {:?}",
        donor_generated_tokens
            .iter()
            .map(|&t| tokenizer.decode(&[t], false).unwrap_or_default())
            .collect::<Vec<_>>()
    );

    // baseline is also layer-independent for its tokens/dists; used once to find discriminating steps.
    let (_, _, recipient_dists_baseline) = run_reference(
        &weights,
        &ffn,
        &recipient_tokens,
        last_layer,
        last_layer,
        N_STEPS,
    );
    let discriminating: Vec<bool> = recipient_dists_baseline
        .iter()
        .enumerate()
        .map(|(step, dist)| {
            kl_divergence(
                &dense_from_sorted(&donor_dists[step], weights.vocab_size),
                &dense_from_sorted(dist, weights.vocab_size),
            ) > DISCRIMINATING_KL_THRESHOLD
        })
        .collect();
    println!("discriminating steps (baseline vs donor KL > {DISCRIMINATING_KL_THRESHOLD}): {discriminating:?}\n");

    for &layer in &LAYER_SWEEP {
        println!("--- patch layer L{layer} ---");
        let (_, donor_rows, _) =
            run_reference(&weights, &ffn, &donor_tokens, layer, last_layer, N_STEPS);
        let (_, _recipient_rows, _) = run_reference(
            &weights,
            &ffn,
            &recipient_tokens,
            layer,
            last_layer,
            N_STEPS,
        );

        let one_shot = run_condition(
            &weights,
            &ffn,
            &recipient_tokens,
            layer,
            last_layer,
            N_STEPS,
            &donor_dists,
            &donor_generated_tokens,
            None,
            Some(&donor_rows[0]),
        );
        print_scores("one-shot", &one_shot, &discriminating, &tokenizer);

        let repeated = run_condition(
            &weights,
            &ffn,
            &recipient_tokens,
            layer,
            last_layer,
            N_STEPS,
            &donor_dists,
            &donor_generated_tokens,
            Some(&donor_rows),
            None,
        );
        print_scores("repeated", &repeated, &discriminating, &tokenizer);
        println!();
    }

    println!(
        "== reading the result ==\n\
         MAP-5b found L16 fails to reach rank 0 at the real branch point (step 3) under EITHER one-shot\n\
         or repeated patching. If a deeper layer (L20/L24/L28) reaches 'WINS every discriminating step'\n\
         under repeated patching where L16 didn't, that confirms 'sufficient for the immediate token' and\n\
         'sufficient for durable branch-token construction' are different depth thresholds — the single-\n\
         layer sweep in MAP-5 understated how deep a patch needs to be for reliable multi-token use."
    );
}
