//! MAP-5b: does the teleported waypoint persist, or was MAP-5 only a
//! one-token effect?
//!
//! MAP-5 (`map5_waypoint_teleportation.rs`) showed a single-position,
//! single-layer residual splice fully redirects the recipient's IMMEDIATE
//! next-token distribution to the donor's answer, in 5/5 pairs, at a
//! per-pair depth threshold (L16 for the planets pair, L24 for the other
//! four — read directly from that run's own output, not re-discovered
//! here). That leaves open whether the recipient enters a self-sustaining
//! "donor trajectory," or whether the redirection is purely local and the
//! system snaps back once the patch stops. Two conditions distinguish
//! these:
//!
//! - **One-shot**: patch ONLY at generation step 0, then free-run for the
//!   remaining steps with no further intervention. Tests whether the
//!   transplant moves the system onto a self-sustaining donor-consistent
//!   trajectory.
//! - **Repeated**: patch at EVERY generation step, injecting the donor's
//!   OWN reference-generation residual (captured from the donor's real,
//!   independently-generated continuation) at the matching step. Tests
//!   whether the donor trajectory can be externally sustained via residual
//!   waypoints alone, without ever touching K/V.
//!
//! Plus four reference/control conditions: donor's own reference
//! generation, recipient's own unpatched baseline, a magnitude-matched
//! random-direction control (one-shot only), and a wrong-donor control
//! (one-shot only, using a REAL donor residual from a DIFFERENT pair —
//! tests whether ANY confident donor-like residual redirects generically,
//! or specifically the semantically-matched one does).
//!
//! Per-step metrics: generated token, KL(donor's own reference
//! distribution at that step, this condition's distribution), rank of the
//! donor's actual generated token in this condition's distribution, and
//! the first step (if any) where the generated token diverges from the
//! donor's own reference sequence.
//!
//! Usage: `LARQL_MAP5_VINDEX=/path/to/model.vindex cargo run --release -p larql-inference --example map5b_teleportation_continuation`

use std::collections::HashMap;

use larql_inference::ffn::WeightFfn;
use larql_inference::forward::patching::{patch_and_trace_with_ffn, DonorState};
use larql_inference::forward::{logit_lens_topk, trace_forward_full};
use larql_inference::FfnBackend;
use larql_models::ModelWeights;

const N_STEPS: usize = 5;

/// (donor prompt, recipient prompt, patch layer) — the patch layer is the
/// FIRST layer MAP-5 found to fully recover the donor's answer for that
/// pair (read from that run's own printed output, not re-derived here).
const PAIRS: [(&str, &str, usize); 5] = [
    ("The capital of France is", "The capital of Japan is", 24),
    (
        "The largest planet in the solar system is",
        "The smallest planet in the solar system is",
        16,
    ),
    ("The opposite of hot is", "The opposite of up is", 24),
    (
        "The chemical symbol for water is",
        "The chemical symbol for gold is",
        24,
    ),
    ("Two plus two equals", "Three plus three equals", 24),
];

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

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn random_unit_vector(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next_f32 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let raw: Vec<f32> = (0..len).map(|_| next_f32()).collect();
    let n = norm(&raw).max(1e-12);
    raw.iter().map(|v| v / n).collect()
}

/// One plain (unpatched) greedy step: returns (next_token, full_sorted_dist, residual_row_at_capture_layer).
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

/// One patched greedy step: injects `patch_row` at (patch_layer, last position), returns (next_token, full_sorted_dist).
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

/// Runs `n_steps` of free-running greedy generation, capturing the
/// residual at `capture_layer` for each newly generated position (used
/// later as the "repeated" condition's per-step donor injection) and the
/// full distribution at each step (used as the KL/rank scoring target).
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
fn run_condition_generation(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    recipient_prompt: &[u32],
    patch_layer: usize,
    last_layer: usize,
    n_steps: usize,
    donor_dists: &[Vec<(u32, f32)>],
    donor_tokens: &[u32],
    // Some(rows) => patch at every step with rows[step]; None => never patch (used by one-shot, which patches only step 0 via `one_shot_row`)
    repeated_rows: Option<&[Vec<f32>]>,
    one_shot_row: Option<&[f32]>,
) -> Vec<StepScore> {
    let mut seq = recipient_prompt.to_vec();
    let mut scores = Vec::with_capacity(n_steps);
    let vocab = weights.vocab_size;
    for step in 0..n_steps {
        let (dist,) = if let Some(rows) = repeated_rows {
            let (_, dist) = patched_step(weights, ffn, &seq, patch_layer, &rows[step], last_layer);
            (dist,)
        } else if step == 0 {
            if let Some(row) = one_shot_row {
                let (_, dist) = patched_step(weights, ffn, &seq, patch_layer, row, last_layer);
                (dist,)
            } else {
                let (_, dist, _) = plain_step(weights, ffn, &seq, patch_layer, last_layer);
                (dist,)
            }
        } else {
            let (_, dist, _) = plain_step(weights, ffn, &seq, patch_layer, last_layer);
            (dist,)
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
    donor_tokens: &[u32],
    tokenizer: &tokenizers::Tokenizer,
) {
    let split_at = scores
        .iter()
        .zip(donor_tokens)
        .position(|(s, &d)| s.token != d);
    print!("  {label:>28}: ");
    for s in scores {
        let text = tokenizer.decode(&[s.token], false).unwrap_or_default();
        print!(
            "[{text:?} kl={:.3} rank_d={:>4}] ",
            s.kl_vs_donor, s.rank_of_donor_token
        );
    }
    println!(
        " split_at_step={}",
        split_at
            .map(|s| s.to_string())
            .unwrap_or_else(|| "never (matches donor throughout)".into())
    );
}

fn main() {
    println!("=== MAP-5b: does teleportation persist across generation steps, or only redirect one token? ===");
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

    // Precompute every pair's donor reference generation once (needed both
    // as its own condition and as the "wrong donor" source for other pairs).
    let mut donor_seqs = Vec::with_capacity(PAIRS.len());
    let mut donor_rows_all = Vec::with_capacity(PAIRS.len());
    let mut donor_dists_all = Vec::with_capacity(PAIRS.len());
    for &(donor_text, _, patch_layer) in &PAIRS {
        let donor_tokens = larql_inference::encode_prompt(&tokenizer, &*weights.arch, donor_text)
            .expect("tokenize donor");
        let (seq, rows, dists) = run_reference(
            &weights,
            &ffn,
            &donor_tokens,
            patch_layer,
            last_layer,
            N_STEPS,
        );
        donor_seqs.push(seq);
        donor_rows_all.push(rows);
        donor_dists_all.push(dists);
    }

    for (i, &(donor_text, recipient_text, patch_layer)) in PAIRS.iter().enumerate() {
        let recipient_tokens =
            larql_inference::encode_prompt(&tokenizer, &*weights.arch, recipient_text)
                .expect("tokenize recipient");
        let donor_prompt_len = donor_seqs[i].len() - N_STEPS;
        let donor_generated_tokens: Vec<u32> = donor_seqs[i][donor_prompt_len..].to_vec();
        println!("\n--- pair {i}: donor={donor_text:?} recipient={recipient_text:?} patch_layer=L{patch_layer} ---");
        println!(
            "  donor reference continuation: {:?}",
            donor_generated_tokens
                .iter()
                .map(|&t| tokenizer.decode(&[t], false).unwrap_or_default())
                .collect::<Vec<_>>()
        );

        // Recipient baseline (unpatched) — also gives us the natural row per
        // step, needed to size the magnitude-matched random control.
        let (_, recipient_rows, recipient_dists) = run_reference(
            &weights,
            &ffn,
            &recipient_tokens,
            patch_layer,
            last_layer,
            N_STEPS,
        );
        let baseline_scores: Vec<StepScore> = recipient_dists
            .iter()
            .enumerate()
            .map(|(step, dist)| {
                score_step(
                    dist,
                    &donor_dists_all[i][step],
                    donor_generated_tokens[step],
                    weights.vocab_size,
                )
            })
            .collect();
        print_scores(
            "recipient baseline",
            &baseline_scores,
            &donor_generated_tokens,
            &tokenizer,
        );

        // One-shot: patch step 0 with the donor's REAL step-0 row, free-run after.
        let one_shot = run_condition_generation(
            &weights,
            &ffn,
            &recipient_tokens,
            patch_layer,
            last_layer,
            N_STEPS,
            &donor_dists_all[i],
            &donor_generated_tokens,
            None,
            Some(&donor_rows_all[i][0]),
        );
        print_scores(
            "one-shot teleport",
            &one_shot,
            &donor_generated_tokens,
            &tokenizer,
        );

        // Repeated: patch every step with the donor's own reference row for that step.
        let repeated = run_condition_generation(
            &weights,
            &ffn,
            &recipient_tokens,
            patch_layer,
            last_layer,
            N_STEPS,
            &donor_dists_all[i],
            &donor_generated_tokens,
            Some(&donor_rows_all[i]),
            None,
        );
        print_scores(
            "repeated teleport",
            &repeated,
            &donor_generated_tokens,
            &tokenizer,
        );

        // Magnitude-matched random control (one-shot only): recipient's own
        // natural row + a random direction scaled to the REAL donor-vs-
        // recipient displacement at step 0.
        let delta_mag = norm(
            &donor_rows_all[i][0]
                .iter()
                .zip(recipient_rows[0].iter())
                .map(|(d, r)| d - r)
                .collect::<Vec<f32>>(),
        );
        let rand_dir = random_unit_vector(recipient_rows[0].len(), 30_000 + i as u64);
        let control_row: Vec<f32> = recipient_rows[0]
            .iter()
            .zip(rand_dir.iter())
            .map(|(r, d)| r + d * delta_mag)
            .collect();
        let control = run_condition_generation(
            &weights,
            &ffn,
            &recipient_tokens,
            patch_layer,
            last_layer,
            N_STEPS,
            &donor_dists_all[i],
            &donor_generated_tokens,
            None,
            Some(&control_row),
        );
        print_scores(
            "magnitude-matched random",
            &control,
            &donor_generated_tokens,
            &tokenizer,
        );

        // Wrong-donor control (one-shot only): a REAL donor row, but from a DIFFERENT pair.
        let wrong_i = (i + 1) % PAIRS.len();
        let wrong_row = &donor_rows_all[wrong_i][0];
        let wrong = run_condition_generation(
            &weights,
            &ffn,
            &recipient_tokens,
            patch_layer,
            last_layer,
            N_STEPS,
            &donor_dists_all[i],
            &donor_generated_tokens,
            None,
            Some(wrong_row),
        );
        print_scores(
            &format!("wrong-donor (pair {wrong_i})"),
            &wrong,
            &donor_generated_tokens,
            &tokenizer,
        );
    }

    println!(
        "\n== reading the result ==\n\
         'split_at_step=0' on one-shot means the effect was purely local — the system reverted to its own\n\
         answer the very next step once the patch stopped. 'never' means the teleport put the system on a\n\
         self-sustaining donor-consistent trajectory using nothing but K/V the recipient already had.\n\
         If repeated teleport tracks the donor much longer than one-shot, that's evidence the residual is\n\
         sufficient locally each step but does not by itself carry enough to bridge to the NEXT step —\n\
         i.e. some state needs refreshing every step, consistent with K/V mattering for persistence even\n\
         though it didn't matter for the single-step MAP-5 result."
    );
}
