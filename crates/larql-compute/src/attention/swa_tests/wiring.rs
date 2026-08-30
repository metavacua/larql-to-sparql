//! The per-layer seam must actually pass the window on.

use super::ramp;
use ndarray::Array2;

/// End-to-end proof that `run_attention_block` *uses* the window rather
/// than merely being able to.
///
/// Two hidden-state sequences that are identical inside the window and
/// differ only outside it must produce the same output at the final
/// position on a **sliding** layer, and different output on a
/// **full-attention** layer of the same model. Testing the primitives in
/// isolation cannot catch a seam that computes the window and then
/// forgets to pass it; this can.
#[test]
fn run_attention_block_honours_the_window_on_sliding_layers_only() {
    let weights = larql_models::test_fixtures::make_test_q4k_weights_rope_scaled();
    let arch = &*weights.arch;
    let window = arch
        .sliding_window_size()
        .expect("fixture declares a window");
    let hidden = weights.hidden_size;

    // Sequence long enough that the window drops the opening positions.
    let seq = window + 8;
    let base = ramp(seq, hidden, 0.11);
    // Perturb only positions the window excludes for the FINAL query.
    // Final query index is seq-1, so it admits [seq-window, seq-1].
    let mut perturbed = base.clone();
    let excluded_end = seq - window;
    for r in 0..excluded_end {
        for c in 0..hidden {
            perturbed[[r, c]] += 0.5;
        }
    }
    assert!(excluded_end > 0, "test needs positions outside the window");

    let sliding_layer = (0..weights.num_layers)
        .find(|&l| arch.is_sliding_window_layer(l))
        .expect("fixture has a sliding layer");
    let full_layer = (0..weights.num_layers)
        .find(|&l| !arch.is_sliding_window_layer(l))
        .expect("fixture has a full-attention layer");

    let last = seq - 1;
    let run = |h: &Array2<f32>, layer: usize| -> Vec<f32> {
        let view = larql_models::WeightsView::dense(&weights);
        let (_, attn, _) =
            crate::attention::block::run_attention_block(view, h, layer, false).expect("block ran");
        attn.row(last).to_vec()
    };

    let sliding_a = run(&base, sliding_layer);
    let sliding_b = run(&perturbed, sliding_layer);
    for (i, (a, b)) in sliding_a.iter().zip(sliding_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "sliding layer {sliding_layer}, dim {i}: changing tokens OUTSIDE the \
             {window}-token window moved the output ({a} vs {b}) — the window is \
             not reaching the attention call"
        );
    }

    let full_a = run(&base, full_layer);
    let full_b = run(&perturbed, full_layer);
    let moved = full_a
        .iter()
        .zip(full_b.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        moved,
        "full-attention layer {full_layer} ignored a change to early tokens — it \
         should attend them, so either the control is degenerate or the window \
         is being applied to every layer"
    );
}
