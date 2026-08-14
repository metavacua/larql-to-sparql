//! Teacher-forced parity: VINDEX2 experts vs the same experts from a VINDEX3
//! container.
//!
//! # Why a greedy continuation is not the gate
//!
//! Running both ways and seeing "Paris." twice proves almost nothing. A short
//! greedy match survives large divergence: the argmax is stable long after the
//! distribution has moved, so agreement on a handful of tokens licenses a claim
//! about *those tokens* and not about the forward pass. This compares the
//! objects that actually differ.
//!
//! # The control is the same loop, not the same answer
//!
//! Both arms call `predict_kquant_hidden_checked` with a bound `MoeRoute`, over
//! identical token ids, on one loaded model. The only difference between them
//! is where the routed expert bytes are read from:
//!
//! ```text
//! control   InProcessMoeBackend      experts from the VINDEX2 mapped index
//! test      ContainerRoutedBackend   experts from the VINDEX3 container
//! ```
//!
//! Comparing against a plain `larql run` instead would change the decode loop
//! as well as the byte source, and a difference could then be either.
//!
//! # What the expected result is, and why
//!
//! c9 established that every routed region in the container is byte-identical
//! to its VINDEX2 source (7680/7680 on `gemma4-26b-a4b`). The same bytes
//! through the same kernels must therefore give **bit-identical** hidden
//! states. Not "close" — identical. A max absolute difference of 1e-7 would be
//! a finding, not a pass: it would mean the two paths are not doing the same
//! arithmetic, and the container would be introducing something.
//!
//! ```text
//! cargo run --release --example vindex3_composed_parity -- <vindex2> <vindex3> [prompt]
//! ```

use larql_inference::ffn::{ContainerRoutedBackend, InProcessMoeBackend, MoeRoute};
use larql_inference::vindex::predict_kquant_hidden_checked;

const DEFAULT_PROMPT: &str = "The capital of France is";

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let v2 = args
        .next()
        .ok_or("usage: vindex3_composed_parity <vindex2-dir> <vindex3-dir> [prompt]")?;
    let v3 = args
        .next()
        .ok_or("usage: vindex3_composed_parity <vindex2-dir> <vindex3-dir> [prompt]")?;
    let prompt = args.next().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let v2_path = std::path::Path::new(&v2);
    let v3_path = std::path::Path::new(&v3);

    println!("teacher-forced parity — VINDEX2 experts vs VINDEX3 container experts");
    println!("  spine     {v2}");
    println!("  container {v3}");
    println!("  prompt    {prompt:?}\n");

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(v2_path, &mut cb)
        .map_err(|e| format!("load weights: {e}"))?;
    let tokenizer =
        larql_vindex::load_vindex_tokenizer(v2_path).map_err(|e| format!("load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(v2_path, &mut cb)
        .map_err(|e| format!("load vindex: {e}"))?;
    index
        .load_attn_kquant(v2_path)
        .map_err(|e| format!("load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(v2_path)
        .map_err(|e| format!("load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(v2_path);

    // Wrap exactly as `larql run` does. Scoring the raw string instead would
    // teacher-force a token sequence the deployed path never sees, and a parity
    // result on inputs nobody serves is a weaker claim than it appears.
    let wrapped =
        larql_inference::chat::render_user_prompt(v2_path, weights.arch.family(), &prompt)
            .map_err(|e| format!("render prompt: {e}"))?;
    let ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped)
        .map_err(|e| format!("tokenise: {e}"))?;
    println!("  {} token(s) teacher-forced\n", ids.len());

    let routed = ContainerRoutedBackend::open(v3_path, &weights, true)
        .map_err(|e| format!("compose: {e}"))?;

    // Control first: if the in-process route itself cannot run, a difference
    // afterwards would have two candidate causes rather than one.
    let control = predict_kquant_hidden_checked(
        &weights,
        &ids,
        &index,
        Some(MoeRoute::fatal(&InProcessMoeBackend)),
    )
    .map_err(|e| format!("control run: {e}"))?;

    let test =
        predict_kquant_hidden_checked(&weights, &ids, &index, Some(MoeRoute::fatal(&routed)))
            .map_err(|e| format!("container run: {e}"))?;

    if control.shape() != test.shape() {
        return Err(format!(
            "shape differs: control {:?}, container {:?}",
            control.shape(),
            test.shape()
        ));
    }

    let mut identical = 0usize;
    let mut max_abs = 0.0f32;
    for (a, b) in control.iter().zip(test.iter()) {
        if a.to_bits() == b.to_bits() {
            identical += 1;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let total = control.len();

    println!("hidden state");
    println!("  elements          {total}");
    println!("  bit-identical     {identical} / {total}");
    println!("  max |difference|  {max_abs:e}");

    // The decision the model would actually make. Only the **last** position
    // predicts the next token, so the argmax has to be taken over that row —
    // over the whole flattened `[seq_len, vocab]` it is the largest logit
    // anywhere in the prompt, which is not a decision the model ever makes.
    let last = |h: &ndarray::Array2<f32>| -> ndarray::Array2<f32> {
        let r = h.nrows().saturating_sub(1);
        h.slice(ndarray::s![r..r + 1, ..]).to_owned()
    };
    let logits_control = larql_inference::forward::hidden_to_raw_logits(&weights, &last(&control));
    let logits_test = larql_inference::forward::hidden_to_raw_logits(&weights, &last(&test));
    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let top_control = argmax(&logits_control);
    let top_test = argmax(&logits_test);
    let logit_max_abs = logits_control
        .iter()
        .zip(logits_test.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\nfinal logits");
    println!("  max |difference|  {logit_max_abs:e}");
    println!(
        "  top-1             control {top_control} ({:?}), container {top_test} ({:?})",
        tokenizer
            .decode(&[top_control as u32], true)
            .unwrap_or_default(),
        tokenizer
            .decode(&[top_test as u32], true)
            .unwrap_or_default()
    );

    if identical != total || max_abs != 0.0 || logit_max_abs != 0.0 {
        return Err(format!(
            "NOT bit-identical — {} of {total} hidden elements differ, max |Δ| {max_abs:e}. \
             The container's regions are byte-identical to source (c9), so the same bytes \
             through the same kernels must give the same numbers; a difference here is the \
             binding, not the storage.",
            total - identical
        ));
    }

    println!("\nPASS — bit-identical. The routed experts were read from the VINDEX3");
    println!("container and the forward pass is indistinguishable from the VINDEX2 one.");
    println!("Composed run only: the container still has no spine and no tokenizer.");
    Ok(())
}
