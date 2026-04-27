//! FFN Graph Walk Coverage — Stage 1 decoupled-mode runner.
//!
//! Spec: `experiments/25_ffn_coverage/SPEC.md`
//!
//! For each prompt, runs one forward pass. At each FFN layer, the instrumented
//! backend computes BOTH the dense WeightFfn output and the WalkFfn output for
//! the same input, records cos(walk, dense) + top gate score + lookup count,
//! then returns the dense output so walk error does not propagate downstream
//! (spec §6.3 decoupled mode).
//!
//! Usage:
//!   cargo run --example ffn_coverage --features real-model --release -- \
//!     google/gemma-3-4b-it output/gemma3-4b-q4k-v2.vindex \
//!     --k full \
//!     --output experiments/25_ffn_coverage/results/factual_narrow.json

#[cfg(feature = "real-model")]
fn main() {
    ffn_coverage::run();
}

#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("This example requires the 'real-model' feature:");
    eprintln!("  cargo run --example ffn_coverage --features real-model --release -- ...");
    std::process::exit(1);
}

#[cfg(feature = "real-model")]
mod ffn_coverage {
    use std::cell::RefCell;
    use std::path::Path;

    use ndarray::{Array1, Array2, ArrayView1};
    use serde::Serialize;

    use kv_cache_benchmark::accuracy_suite::prompts::{diverse_100, TestPrompt};
    use larql_inference::ffn::{FfnBackend, WeightFfn};
    use larql_inference::forward::predict_with_ffn;
    use larql_inference::model::ModelWeights;
    use larql_inference::{default_backend, InferenceModel, WalkFfn, WalkFfnConfig};
    use larql_vindex::{SilentLoadCallbacks, VectorIndex};

    /// CLI args. Minimal; no clap dependency.
    struct Args {
        model: String,
        vindex: String,
        output: String,
        k: Option<usize>, // None = dense walk (all features)
        limit: Option<usize>,
    }

    fn parse_args() -> Args {
        let mut raw: Vec<String> = std::env::args().skip(1).collect();
        let mut k: Option<usize> = None;
        let mut output = String::from("experiments/25_ffn_coverage/results/factual_narrow.json");
        let mut limit: Option<usize> = None;

        let mut i = 0;
        while i < raw.len() {
            match raw[i].as_str() {
                "--k" => {
                    let v = raw.get(i + 1).cloned().unwrap_or_else(|| "full".into());
                    k = if v == "full" { None } else { Some(v.parse().expect("--k must be int or 'full'")) };
                    raw.drain(i..i + 2);
                }
                "--output" | "-o" => {
                    output = raw.get(i + 1).cloned().expect("--output needs a path");
                    raw.drain(i..i + 2);
                }
                "--limit" => {
                    limit = Some(raw.get(i + 1).and_then(|s| s.parse().ok()).expect("--limit needs int"));
                    raw.drain(i..i + 2);
                }
                _ => i += 1,
            }
        }

        if raw.len() < 2 {
            eprintln!("Usage: ffn_coverage <model> <vindex> [--k N|full] [--output PATH] [--limit N]");
            std::process::exit(2);
        }
        Args { model: raw[0].clone(), vindex: raw[1].clone(), output, k, limit }
    }

    // ── Measurement records ──

    #[derive(Serialize, Clone)]
    struct LayerMeasurement {
        layer: usize,
        /// cos(walked FFN output row, dense FFN output row) at the last token position.
        cos_walk_vs_dense: f32,
        /// Top-1 gate score at the last token (routing confidence proxy).
        gate_top_score: f32,
        /// Top-1 feature id.
        gate_top_feature: usize,
        /// Margin between top-1 and top-2 gate scores (|s1| - |s2|).
        gate_top_margin: f32,
        /// Lookup count for this layer: 1 gate_knn call + K feature reads + K down reads.
        /// For the spec's "3 per hop" comparison, report gate_knn + best-feature retrieval only.
        lookup_count: usize,
        /// Top-1 token that the winning feature projects to (from vindex metadata).
        gate_top_token: String,
    }

    #[derive(Serialize, Clone)]
    struct QueryResult {
        prompt: String,
        category: String,
        expected_contains: String,
        dense_top1_token: String,
        dense_top1_prob: f64,
        /// Raw per-layer measurements.
        layers: Vec<LayerMeasurement>,
    }

    // ── Instrumented FFN backend ──
    //
    // The walk used here is LIVE (computed inside `forward`). That's required
    // because the input `x` to the FFN at layer L depends on the dense
    // residual at L-1 (which is what decoupled mode preserves). We can't
    // replay the walk offline against captured inputs without re-running
    // every layer's attention and the pre-FFN norm — which is exactly what
    // `predict_with_ffn` does for us.

    struct InstrumentedFfn<'a> {
        weights: &'a ModelWeights,
        walk: WalkFfn<'a>,
        index: &'a VectorIndex,
        gate_k_for_measurement: usize,
        measurements: RefCell<Vec<LayerMeasurement>>,
    }

    impl<'a> FfnBackend for InstrumentedFfn<'a> {
        fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
            let dense = WeightFfn { weights: self.weights };
            let dense_out = dense.forward(layer, x);
            let walk_out = self.walk.forward(layer, x);

            let last = x.shape()[0] - 1;
            let cos = cosine(dense_out.row(last), walk_out.row(last));

            // Record gate KNN top-K on the same input — this is the routing
            // confidence signal §2.4 needs. The walk itself already ran a
            // gate_knn internally; we re-run with a small K purely to grab
            // top-K scores for measurement. Redundant but cheap.
            let x_last = Array1::from_iter(x.row(last).iter().copied());
            let top_hits = self.index.gate_knn(layer, &x_last, self.gate_k_for_measurement);
            let (feat0, score0) = top_hits.first().copied().unwrap_or((0, 0.0));
            let score1 = top_hits.get(1).map(|(_, s)| s.abs()).unwrap_or(0.0);
            let margin = score0.abs() - score1;
            let token = self.index.feature_meta(layer, feat0).map(|m| m.top_token).unwrap_or_default();

            // Lookup count: gate_knn (1) + K feature reads (K) + K down reads (K).
            // When K_walk = features, this is ~2*F + 1. Report the effective K
            // the walk used (None = features count).
            let num_features = self.index.num_features(layer);
            let walk_k = self.walk.config.k_for(layer).unwrap_or(num_features);
            let lookup_count = 1 + 2 * walk_k;

            self.measurements.borrow_mut().push(LayerMeasurement {
                layer,
                cos_walk_vs_dense: cos,
                gate_top_score: score0,
                gate_top_feature: feat0,
                gate_top_margin: margin,
                lookup_count,
                gate_top_token: token,
            });

            dense_out
        }

        fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
            let (out, act) = WeightFfn { weights: self.weights }.forward_with_activation(layer, x);
            // Re-run walk for measurement; discard its activation (we return dense).
            let _ = self.forward(layer, x);
            (out, act)
        }

        fn name(&self) -> &str {
            "instrumented"
        }
    }

    fn cosine(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    // ── Main ──

    pub fn run() {
        let args = parse_args();

        println!("Loading model: {}", args.model);
        let model = InferenceModel::load(&args.model).expect("Failed to load model");

        println!("Loading vindex from: {}", args.vindex);
        let index = VectorIndex::load_vindex(Path::new(&args.vindex), &mut SilentLoadCallbacks)
            .expect("Failed to load vindex");

        let backend = default_backend();
        let num_layers = model.weights().num_layers;
        let walk_config = match args.k {
            None => WalkFfnConfig::dense(num_layers),
            Some(k) => WalkFfnConfig::sparse(num_layers, k),
        };
        println!(
            "WalkFfn: {} layers, K = {}",
            num_layers,
            args.k.map(|k| k.to_string()).unwrap_or_else(|| "full".into())
        );

        let all_prompts = diverse_100();
        let prompts: Vec<&TestPrompt> = match args.limit {
            Some(n) => all_prompts.iter().take(n).collect(),
            None => all_prompts.iter().collect(),
        };
        println!("\nRunning {} prompts\n", prompts.len());

        let mut results: Vec<QueryResult> = Vec::with_capacity(prompts.len());

        for (i, tp) in prompts.iter().enumerate() {
            let encoding = model
                .tokenizer()
                .encode(tp.text, true)
                .expect("tokenize failed");
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let walk = WalkFfn::from_config(model.weights(), &index, walk_config.clone())
                .with_backend(backend.as_ref());
            let instrumented = InstrumentedFfn {
                weights: model.weights(),
                walk,
                index: &index,
                gate_k_for_measurement: 8,
                measurements: RefCell::new(Vec::with_capacity(num_layers)),
            };

            let t0 = std::time::Instant::now();
            let pred = predict_with_ffn(
                model.weights(),
                model.tokenizer(),
                &token_ids,
                5,
                &instrumented,
            );
            let elapsed = t0.elapsed();

            let (top_tok, top_prob) = pred
                .predictions
                .first()
                .cloned()
                .unwrap_or_else(|| (String::new(), 0.0));

            let mut layers = instrumented.measurements.into_inner();
            layers.sort_by_key(|m| m.layer);

            let worst_cos = layers.iter().map(|m| m.cos_walk_vs_dense).fold(f32::INFINITY, f32::min);
            let mean_cos = layers.iter().map(|m| m.cos_walk_vs_dense).sum::<f32>() / layers.len() as f32;
            println!(
                "[{:>3}/{}] {:<60}  top1={:<15} mean_cos={:.4} worst_cos={:.4}  {:>6.1}s",
                i + 1,
                prompts.len(),
                truncate(tp.text, 60),
                truncate(&top_tok, 15),
                mean_cos,
                worst_cos,
                elapsed.as_secs_f32(),
            );

            results.push(QueryResult {
                prompt: tp.text.to_string(),
                category: tp.category.to_string(),
                expected_contains: tp.expected_contains.to_string(),
                dense_top1_token: top_tok,
                dense_top1_prob: top_prob,
                layers,
            });
        }

        // ── Emit per-query JSON + summary ──

        let out_path = Path::new(&args.output);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let json = serde_json::to_string_pretty(&results).expect("serialize");
        std::fs::write(out_path, json).expect("write output");
        println!("\nWrote {} query results to {}", results.len(), out_path.display());

        print_coverage_summary(&results);
    }

    fn truncate(s: &str, n: usize) -> String {
        if s.chars().count() <= n {
            s.to_string()
        } else {
            let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
            out.push('…');
            out
        }
    }

    fn print_coverage_summary(results: &[QueryResult]) {
        let thresholds: [f32; 5] = [0.95, 0.99, 0.999, 0.9999, 1.0];

        println!("\n── Coverage summary ──");
        println!("queries={}, layers/query={}", results.len(), results.first().map(|r| r.layers.len()).unwrap_or(0));

        println!("\nFully-walked rate (all layers cos ≥ τ):");
        for &tau in &thresholds {
            let fully_walked = results
                .iter()
                .filter(|r| r.layers.iter().all(|m| m.cos_walk_vs_dense >= tau))
                .count();
            println!("  τ={:<8} fully-walked: {}/{} ({:>5.1}%)",
                     format_tau(tau), fully_walked, results.len(),
                     100.0 * fully_walked as f32 / results.len() as f32);
        }

        println!("\nPer-layer walk rate at τ=0.99:");
        let num_layers = results.first().map(|r| r.layers.len()).unwrap_or(0);
        for l in 0..num_layers {
            let hits = results.iter().filter(|r| r.layers[l].cos_walk_vs_dense >= 0.99).count();
            let bar = "█".repeat(((hits as f32 / results.len() as f32) * 20.0) as usize);
            println!("  L{:<2} {:<20} {}/{}", l, bar, hits, results.len());
        }

        // Silent wrongness: high confidence (top gate score >= 0.99 of max observed)
        // AND walk fidelity < 0.99. Confidence normalisation needs a proper
        // definition (spec §10); for Stage 1 we report raw absolute gate scores.
        let mut silent_wrong = 0;
        for r in results {
            for m in &r.layers {
                if m.gate_top_score.abs() >= 10.0 && m.cos_walk_vs_dense < 0.99 {
                    silent_wrong += 1;
                }
            }
        }
        println!(
            "\nCandidate silent-wrongness cells (|gate_top_score| ≥ 10.0 ∧ cos < 0.99): {}",
            silent_wrong
        );
    }

    fn format_tau(t: f32) -> String {
        if t == 1.0 {
            "1.0".to_string()
        } else {
            format!("{t}")
        }
    }
}
