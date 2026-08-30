//! MAP-5: waypoint teleportation — is a waypoint self-contained in the
//! residual state, or does the executable waypoint require attention K/V
//! history too?
//!
//! Scope (per review): the full ladder proposed for this experiment has
//! six conditions, three of which (donor KV at causal-address positions,
//! full donor KV history, recipient-residual+donor-causal-KV) require
//! capturing and overriding attention Key/Value tensors mid-forward-pass.
//! That capability does not exist anywhere in the hook system today —
//! `LayerHook`'s five methods only ever see residual rows and post-softmax
//! attention weights; K/V is computed internally by
//! `run_layer_with_capture_hooked` and discarded. Building it is real,
//! non-trivial engine work (a new hook point threading K/V through the
//! attention call), not attempted here. This ships the two conditions
//! that ARE buildable today with existing, unit-tested infrastructure
//! (`forward::patching`), and reports honestly where the ladder stops:
//!
//! - **Condition A** (last-position residual teleport): donor's post-layer
//!   residual at ONE layer, ONE position (the final/answer token), spliced
//!   into the recipient at the same position, rest of the forward pass
//!   runs normally (`PatchHook::on_post_layer`, unmodified).
//! - **Condition A-wide** (not in the original ladder, free with the same
//!   primitive): donor's residual at EVERY overlapping position, not just
//!   the last one — tests the spatial-extent axis independently of the
//!   K/V question.
//! - **Condition F** (magnitude-matched control, per
//!   `feedback_concentration_of_measure_perturbation_design`): same splice
//!   point, but the donor's real row is replaced by the recipient's OWN
//!   baseline row plus a random-direction vector scaled to the REAL
//!   donor-vs-baseline displacement magnitude — tests whether ANY
//!   perturbation of that size is equally disruptive, or whether the
//!   donor's specific content is what moves the output.
//!
//! Swept across a layer ladder (early/mid/late) on 5 donor/recipient
//! prompt pairs sharing a template but differing in factual content, so
//! "did teleportation work" has an unambiguous, single-token answer to
//! check rank/KL against.
//!
//! NOT attempted here (needs new K/V hook infrastructure, flagged as
//! follow-up): donor KV at the causal-address positions, full donor KV
//! transplant, recipient-residual+donor-KV, and any KV-mismatch control.
//!
//! Usage: `LARQL_MAP5_VINDEX=/path/to/model.vindex cargo run --release -p larql-inference --example map5_waypoint_teleportation`

use std::collections::HashMap;

use larql_inference::ffn::WeightFfn;
use larql_inference::forward::patching::DonorState;
use larql_inference::forward::{logit_lens_topk, trace_forward_full};
use larql_inference::FfnBackend;
use larql_models::ModelWeights;

const LAYER_SWEEP: [usize; 7] = [4, 8, 12, 16, 20, 24, 28];

/// (donor prompt, recipient prompt) — same template, different content, so
/// teleportation success/failure has an unambiguous single-token target.
const PAIRS: [(&str, &str); 5] = [
    ("The capital of France is", "The capital of Japan is"),
    (
        "The largest planet in the solar system is",
        "The smallest planet in the solar system is",
    ),
    ("The opposite of hot is", "The opposite of up is"),
    (
        "The chemical symbol for water is",
        "The chemical symbol for gold is",
    ),
    ("Two plus two equals", "Three plus three equals"),
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

/// Deterministic, seedable pseudo-random unit vector (xorshift-style —
/// same generator used elsewhere in the MAP examples, no external crate).
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

/// Final-layer residual row (last token) → sorted logit-lens distribution.
fn final_sorted(weights: &ModelWeights, row: &[f32]) -> Vec<(u32, f32)> {
    logit_lens_topk(weights, row, weights.vocab_size)
}

/// Post-layer residual row at `layer`, last token, from a plain (unhooked)
/// forward pass on `tokens`.
fn capture_row(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    tokens: &[u32],
    layer: usize,
) -> Vec<f32> {
    let trace = trace_forward_full(weights, tokens, &[layer], false, 0, false, ffn);
    trace
        .residuals
        .into_iter()
        .find(|(l, _)| *l == layer)
        .expect("layer must be captured")
        .1
}

/// Per-pair targets that do NOT depend on the sweep layer — computed once
/// per pair, not once per (pair, layer) cell (the original version
/// redundantly re-ran a full-depth forward pass for these on every layer
/// in the sweep, which is what made the first run time out).
struct PairTargets {
    donor_final_dense: Vec<f32>,
    recipient_final_dense: Vec<f32>,
    donor_top1: u32,
    recipient_top1: u32,
}

fn compute_pair_targets(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    donor_tokens: &[u32],
    recipient_tokens: &[u32],
    last_layer: usize,
) -> PairTargets {
    let donor_final = capture_row(weights, ffn, donor_tokens, last_layer);
    let recipient_final = capture_row(weights, ffn, recipient_tokens, last_layer);
    let donor_final_sorted = final_sorted(weights, &donor_final);
    let recipient_final_sorted = final_sorted(weights, &recipient_final);
    PairTargets {
        donor_top1: donor_final_sorted[0].0,
        recipient_top1: recipient_final_sorted[0].0,
        donor_final_dense: dense_from_sorted(&donor_final_sorted, weights.vocab_size),
        recipient_final_dense: dense_from_sorted(&recipient_final_sorted, weights.vocab_size),
    }
}

struct Baselines {
    donor_row: Vec<f32>,
    recipient_row: Vec<f32>,
}

fn compute_baselines(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    donor_tokens: &[u32],
    recipient_tokens: &[u32],
    layer: usize,
) -> Baselines {
    Baselines {
        donor_row: capture_row(weights, ffn, donor_tokens, layer),
        recipient_row: capture_row(weights, ffn, recipient_tokens, layer),
    }
}

/// Runs `patch_and_trace` with the given donor-state records, returns the
/// recipient's post-patch final-layer sorted distribution.
fn run_patched(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    recipient_tokens: &[u32],
    records: HashMap<(usize, usize), Vec<f32>>,
    last_layer: usize,
) -> Vec<(u32, f32)> {
    let donor = DonorState { records };
    let trace = larql_inference::forward::patching::patch_and_trace_with_ffn(
        weights,
        recipient_tokens,
        &donor,
        &[last_layer],
        ffn,
    );
    let row = trace
        .residuals
        .into_iter()
        .find(|(l, _)| *l == last_layer)
        .expect("last layer must be captured")
        .1;
    final_sorted(weights, &row)
}

struct CellResult {
    layer: usize,
    kl_a_vs_donor: f32,
    kl_a_vs_recipient: f32,
    rank_a_of_donor_top1: usize,
    rank_a_of_recipient_top1: usize,
    kl_wide_vs_donor: f32,
    rank_wide_of_donor_top1: usize,
    kl_control_vs_donor: f32,
    rank_control_of_donor_top1: usize,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_layer(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    donor_tokens: &[u32],
    recipient_tokens: &[u32],
    layer: usize,
    last_layer: usize,
    targets: &PairTargets,
    seed: u64,
) -> CellResult {
    let b = compute_baselines(weights, ffn, donor_tokens, recipient_tokens, layer);
    let recipient_last_pos = recipient_tokens.len() - 1;

    // Condition A: single position (last token), donor's real row.
    let mut records_a = HashMap::new();
    records_a.insert((layer, recipient_last_pos), b.donor_row.clone());
    let a = run_patched(weights, ffn, recipient_tokens, records_a, last_layer);
    let a_dense = dense_from_sorted(&a, weights.vocab_size);

    // Condition A-wide: every overlapping position gets the donor's row at
    // that SAME position index (not just the last one). Needs the FULL
    // per-position matrix, not the last-token-only row `trace_forward_full`
    // gives — captured directly via RecordHook instead.
    let overlap = recipient_tokens.len().min(donor_tokens.len());
    let mut records_wide = HashMap::new();
    {
        use larql_inference::forward::{trace_forward_full_hooked, RecordHook};
        let mut record = RecordHook::for_layers([layer]);
        let _ = trace_forward_full_hooked(
            weights,
            donor_tokens,
            &[layer],
            false,
            0,
            false,
            ffn,
            &mut record,
        );
        if let Some(matrix) = record.post_layer.get(&layer) {
            for pos in 0..overlap {
                records_wide.insert((layer, pos), matrix.row(pos).to_vec());
            }
        }
    }
    let wide = run_patched(weights, ffn, recipient_tokens, records_wide, last_layer);

    // Condition F: magnitude-matched random-direction control at the same
    // single position — real displacement magnitude, random direction.
    let delta_mag = norm(
        &b.donor_row
            .iter()
            .zip(b.recipient_row.iter())
            .map(|(d, r)| d - r)
            .collect::<Vec<f32>>(),
    );
    let rand_dir = random_unit_vector(b.recipient_row.len(), seed);
    let control_row: Vec<f32> = b
        .recipient_row
        .iter()
        .zip(rand_dir.iter())
        .map(|(r, d)| r + d * delta_mag)
        .collect();
    let mut records_control = HashMap::new();
    records_control.insert((layer, recipient_last_pos), control_row);
    let control = run_patched(weights, ffn, recipient_tokens, records_control, last_layer);

    CellResult {
        layer,
        kl_a_vs_donor: kl_divergence(&targets.donor_final_dense, &a_dense),
        kl_a_vs_recipient: kl_divergence(&targets.recipient_final_dense, &a_dense),
        rank_a_of_donor_top1: rank_of(&a, targets.donor_top1),
        rank_a_of_recipient_top1: rank_of(&a, targets.recipient_top1),
        kl_wide_vs_donor: kl_divergence(
            &targets.donor_final_dense,
            &dense_from_sorted(&wide, weights.vocab_size),
        ),
        rank_wide_of_donor_top1: rank_of(&wide, targets.donor_top1),
        kl_control_vs_donor: kl_divergence(
            &targets.donor_final_dense,
            &dense_from_sorted(&control, weights.vocab_size),
        ),
        rank_control_of_donor_top1: rank_of(&control, targets.donor_top1),
    }
}

fn print_cell(c: &CellResult) {
    println!(
        "  L{:>2}: A(last-pos) rank(donor_top1)={:>6} rank(recipient_top1)={:>6} KL_vs_donor={:>8.4} KL_vs_recipient={:>8.4}  |  A-wide rank(donor_top1)={:>6} KL_vs_donor={:>8.4}  |  control(magnitude-matched) rank(donor_top1)={:>6} KL_vs_donor={:>8.4}",
        c.layer, c.rank_a_of_donor_top1, c.rank_a_of_recipient_top1, c.kl_a_vs_donor, c.kl_a_vs_recipient,
        c.rank_wide_of_donor_top1, c.kl_wide_vs_donor,
        c.rank_control_of_donor_top1, c.kl_control_vs_donor,
    );
}

fn main() {
    println!("=== MAP-5: waypoint teleportation (residual-only conditions A, A-wide, F — see doc comment for KV-condition scope) ===");
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
    let sweep: Vec<usize> = LAYER_SWEEP
        .iter()
        .copied()
        .filter(|&l| l < last_layer)
        .collect();
    println!(
        "{} layers x {} donor/recipient pairs\n",
        sweep.len(),
        PAIRS.len()
    );

    let mut per_pair_first_success: Vec<Option<usize>> = Vec::new();

    for (i, &(donor_text, recipient_text)) in PAIRS.iter().enumerate() {
        let donor_tokens = larql_inference::encode_prompt(&tokenizer, &*weights.arch, donor_text)
            .expect("tokenize donor");
        let recipient_tokens =
            larql_inference::encode_prompt(&tokenizer, &*weights.arch, recipient_text)
                .expect("tokenize recipient");
        println!("--- pair {i}: donor={donor_text:?} recipient={recipient_text:?} ---");
        let targets =
            compute_pair_targets(&weights, &ffn, &donor_tokens, &recipient_tokens, last_layer);

        let mut first_success = None;
        for (j, &layer) in sweep.iter().enumerate() {
            let cell = evaluate_layer(
                &weights,
                &ffn,
                &donor_tokens,
                &recipient_tokens,
                layer,
                last_layer,
                &targets,
                20_000 + i as u64 * 1000 + j as u64 * 10,
            );
            print_cell(&cell);
            if first_success.is_none() && cell.rank_a_of_donor_top1 == 0 {
                first_success = Some(layer);
            }
        }
        println!(
            "  first layer where condition A recovers donor's true top-1 as rank 0: {}\n",
            first_success
                .map(|l| l.to_string())
                .unwrap_or_else(|| "none in sweep".to_string())
        );
        per_pair_first_success.push(first_success);
    }

    println!("=== summary: does last-position residual-only teleportation ever recover the donor's answer? ===");
    for (i, &(donor_text, recipient_text)) in PAIRS.iter().enumerate() {
        println!(
            "  pair {i} ({donor_text:?} -> {recipient_text:?}): {}",
            per_pair_first_success[i]
                .map(|l| format!("YES, first at L{l}"))
                .unwrap_or_else(|| "NO, never in this sweep".to_string())
        );
    }
    let n_success = per_pair_first_success
        .iter()
        .filter(|s| s.is_some())
        .count();
    println!(
        "\n{n_success}/{} pairs: residual-only teleportation at a single position recovers the donor's exact answer at some layer.\n\
         If this is LOW, that's real evidence the waypoint needs more than the residual alone (motivating the K/V follow-up).\n\
         If this is HIGH but the magnitude-matched control ALSO frequently recovers rank 0, that would instead suggest the \
         recipient's own late-layer computation is fragile to any large perturbation there, not that the donor's specific \
         content is what's doing the work — compare the A vs control rows above at whichever layer A succeeds.",
        PAIRS.len()
    );
}
