//! Which layers are windowed, and how wide.

use crate::forward_overrides::effective_attention_window_for_layer;

/// Gemma 3's 5:1 pattern: layers 5, 11, 17, … are full attention, the
/// rest slide. The window must follow the architecture, not the layer
/// index modulo anything this test invents.
///
/// Uses the rope-scaled fixture specifically because it **declares a
/// real window** (512) across 6 layers, so the 5:1 boundary lands inside
/// the model. The plain Gemma-3 fixtures declare sliding layers with no
/// width, which resolves to `None` everywhere and would make this test
/// pass without ever exercising the windowed branch.
#[test]
fn window_follows_the_architecture_per_layer() {
    let weights = larql_models::test_fixtures::make_test_q4k_weights_rope_scaled();
    let arch = &*weights.arch;
    let declared = arch.sliding_window_size();
    assert_eq!(
        declared,
        Some(512),
        "fixture must declare a real window or this test proves nothing"
    );

    let mut saw_windowed = 0usize;
    let mut saw_full = 0usize;
    for layer in 0..weights.num_layers {
        let got = effective_attention_window_for_layer(arch, layer);
        if arch.is_sliding_window_layer(layer) {
            assert_eq!(
                got, declared,
                "layer {layer} slides, so it must report the declared window"
            );
            saw_windowed += 1;
        } else {
            assert_eq!(
                got, None,
                "layer {layer} is a full-attention layer and must not be windowed"
            );
            saw_full += 1;
        }
    }
    // Both branches must actually have been taken.
    assert!(
        saw_windowed > 0 && saw_full > 0,
        "fixture must contain both sliding and full layers (saw {saw_windowed} / {saw_full})"
    );
}

/// A layer that declares itself sliding but supplies no width is
/// answered `None` (full attention) rather than `Some(0)` (attend
/// nothing). Several fixtures are in exactly this state, and the old
/// inline `unwrap_or(0)` in the Metal spec meant the same literal `0`
/// stood for both "no window" and "empty window".
#[test]
fn a_sliding_layer_without_a_declared_width_is_not_windowed() {
    let weights = larql_inference_test_q4k_weights();
    let arch = &*weights.arch;
    assert!(
        arch.is_sliding_window_layer(0),
        "fixture is expected to declare sliding layers"
    );
    assert_eq!(
        arch.sliding_window_size(),
        None,
        "fixture is expected to omit the window width"
    );
    assert_eq!(
        effective_attention_window_for_layer(arch, 0),
        None,
        "no width means no window — never an empty one"
    );
}

fn larql_inference_test_q4k_weights() -> larql_models::ModelWeights {
    larql_models::test_fixtures::make_test_q4k_weights()
}

/// Architectures with no sliding layers at all are never windowed.
#[test]
fn non_sliding_architectures_are_never_windowed() {
    let weights = larql_models::test_fixtures::make_test_q4k_weights_silu();
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        assert!(!arch.is_sliding_window_layer(layer));
        assert_eq!(effective_attention_window_for_layer(arch, layer), None);
    }
}
