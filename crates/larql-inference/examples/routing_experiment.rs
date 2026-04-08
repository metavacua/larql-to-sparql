//! Routing experiment: measure template-dependence of attention and FFN routing.
//!
//! For each template × entity:
//!   - Capture residual at every layer (last position)
//!   - Capture attention weights at every layer
//!   - Capture top-K FFN activations at every layer
//!
//! Then measure:
//!   1. Residual cosine stability within template (should be ~0.99)
//!   2. Attention pattern cosine stability within template
//!   3. FFN feature Jaccard overlap within template
//!   4. Cross-template separation (different templates → different routing?)
//!
//! Usage:
//!   cargo run --release -p larql-inference --example routing_experiment

use std::collections::HashSet;
use larql_inference::{InferenceModel, WeightFfn};
use larql_inference::forward::trace_forward_full;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { return 0.0; }
    dot / (na * nb)
}

fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f32 {
    if a.is_empty() && b.is_empty() { return 1.0; }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { return 0.0; }
    inter as f32 / union as f32
}

/// Flatten attention weights into a single vector for cosine comparison.
fn flatten_attn(weights: &larql_inference::attention::AttentionWeights) -> Vec<f32> {
    let mut flat = Vec::new();
    for head in &weights.heads {
        flat.extend_from_slice(head);
    }
    flat
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    let dense_ffn = WeightFfn { weights };

    let templates: Vec<(&str, &str, Vec<&str>)> = vec![
        ("capital", "The capital of {} is", vec![
            "France", "Germany", "Japan", "Brazil", "Egypt",
            "Australia", "Mexico", "India", "Canada", "Italy",
            "Spain", "China", "Russia", "Turkey", "Thailand",
            "Argentina", "Nigeria", "Kenya", "Poland", "Sweden",
        ]),
        ("language", "The language spoken in {} is", vec![
            "France", "Germany", "Japan", "Brazil", "Egypt",
            "China", "Russia", "Thailand", "Mexico", "Italy",
            "Spain", "India", "Turkey", "Poland", "Sweden",
            "Greece", "Portugal", "Vietnam", "Indonesia", "Korea",
        ]),
        ("currency", "The currency of {} is the", vec![
            "Japan", "Brazil", "India", "Mexico", "China",
            "Russia", "Thailand", "Turkey", "Poland", "Sweden",
            "Australia", "Canada", "Egypt", "Nigeria", "Kenya",
            "Argentina", "Switzerland", "Norway", "Denmark", "Hungary",
        ]),
        ("born", "{} was born in", vec![
            "Einstein", "Mozart", "Shakespeare", "Picasso", "Darwin",
            "Beethoven", "Galileo", "Newton", "Tesla", "Curie",
            "Aristotle", "Plato", "Napoleon", "Cleopatra", "Gandhi",
            "Confucius", "Columbus", "Copernicus", "Gutenberg", "Euler",
        ]),
    ];

    let all_layers: Vec<usize> = (0..num_layers).collect();
    let activation_top_k = 200;

    println!("=== Routing Stability Experiment ===\n");
    println!("{} templates, {} entities each, {} layers\n",
        templates.len(), templates[0].2.len(), num_layers);

    // Store all results for cross-template analysis
    let mut all_residuals: Vec<(String, Vec<Vec<Vec<f32>>>)> = Vec::new(); // (template, [entity][layer][hidden])
    let mut all_attn: Vec<(String, Vec<Vec<Vec<f32>>>)> = Vec::new();     // (template, [entity][layer][flat_attn])
    let mut all_features: Vec<(String, Vec<Vec<HashSet<usize>>>)> = Vec::new(); // (template, [entity][layer]{features})

    for (tname, template, entities) in &templates {
        println!("--- Template: {tname} (\"{template}\") ---");

        let mut t_residuals: Vec<Vec<Vec<f32>>> = Vec::new();  // [entity][layer][hidden]
        let mut t_attn: Vec<Vec<Vec<f32>>> = Vec::new();       // [entity][layer][flat_attn]
        let mut t_features: Vec<Vec<HashSet<usize>>> = Vec::new(); // [entity][layer]{features}

        for entity in entities {
            let prompt = template.replace("{}", entity);
            let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| format!("{e}"))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let trace = trace_forward_full(
                weights, &token_ids, &all_layers,
                true, activation_top_k, true, &dense_ffn,
            );

            // Extract per-layer data
            let mut e_residuals = Vec::new();
            let mut e_attn = Vec::new();
            let mut e_features = Vec::new();

            for layer in 0..num_layers {
                // Residual
                if let Some((_, res)) = trace.residuals.iter().find(|(l, _)| *l == layer) {
                    e_residuals.push(res.clone());
                } else {
                    e_residuals.push(vec![0.0; weights.hidden_size]);
                }

                // Attention (flatten for cosine comparison)
                if let Some(cap) = trace.attention.iter().find(|c| c.layer == layer) {
                    e_attn.push(flatten_attn(&cap.weights));
                } else {
                    e_attn.push(vec![]);
                }

                // FFN features (top activations with |act| > 1.0)
                let feats: HashSet<usize> = trace.activations.iter()
                    .find(|(l, _)| *l == layer)
                    .map(|(_, acts)| acts.iter()
                        .filter(|(_, a)| a.abs() > 1.0)
                        .map(|(f, _)| *f)
                        .collect())
                    .unwrap_or_default();
                e_features.push(feats);
            }

            t_residuals.push(e_residuals);
            t_attn.push(e_attn);
            t_features.push(e_features);
        }

        let n = entities.len();

        // Per-layer stability metrics
        println!("  {:>5} {:>8} {:>9} {:>9} {:>9}", "Layer", "Res cos", "Attn cos", "FFN Jacc", "FFN union");

        for layer in 0..num_layers {
            // Pairwise residual cosine
            let mut res_cos_sum = 0.0f64;
            let mut attn_cos_sum = 0.0f64;
            let mut jacc_sum = 0.0f64;
            let mut pairs = 0usize;

            let mut feature_union: HashSet<usize> = HashSet::new();
            for feat in t_features.iter().take(n) {
                feature_union.extend(feat[layer].iter());
            }

            for i in 0..n {
                for j in (i+1)..n {
                    res_cos_sum += cosine(&t_residuals[i][layer], &t_residuals[j][layer]) as f64;
                    if !t_attn[i][layer].is_empty() && !t_attn[j][layer].is_empty() {
                        attn_cos_sum += cosine(&t_attn[i][layer], &t_attn[j][layer]) as f64;
                    }
                    jacc_sum += jaccard(&t_features[i][layer], &t_features[j][layer]) as f64;
                    pairs += 1;
                }
            }

            if pairs > 0 && (layer % 4 == 0 || layer == num_layers - 1) {
                let res_cos = res_cos_sum / pairs as f64;
                let attn_cos = attn_cos_sum / pairs as f64;
                let jacc = jacc_sum / pairs as f64;
                println!("  L{layer:2}:   {res_cos:>7.4}  {attn_cos:>8.4}  {jacc:>8.4}  {feature_union:>8}",
                    feature_union = feature_union.len());
            }
        }

        all_residuals.push((tname.to_string(), t_residuals));
        all_attn.push((tname.to_string(), t_attn));
        all_features.push((tname.to_string(), t_features));
        println!();
    }

    // Cross-template separation: residual cosine between templates
    println!("--- Cross-template residual cosine (L16, entity 0 vs entity 0) ---");
    for i in 0..all_residuals.len() {
        for j in (i+1)..all_residuals.len() {
            let cos = cosine(&all_residuals[i].1[0][16], &all_residuals[j].1[0][16]);
            println!("  {} vs {}: {cos:.4}", all_residuals[i].0, all_residuals[j].0);
        }
    }

    println!("\n--- Cross-template FFN Jaccard (L16, entity 0 vs entity 0) ---");
    for i in 0..all_features.len() {
        for j in (i+1)..all_features.len() {
            let jacc = jaccard(&all_features[i].1[0][16], &all_features[j].1[0][16]);
            println!("  {} vs {}: {jacc:.4}", all_features[i].0, all_features[j].0);
        }
    }

    // Feature union size across all entities per template (how many distinct features per layer?)
    println!("\n--- Feature universe per template per layer ---");
    println!("  {:>10} {:>5} {:>5} {:>5} {:>5} {:>5}", "", "L0", "L8", "L16", "L24", "L33");
    for (tname, _, t_features) in all_features.iter()
        .map(|(name, feats)| (name, &templates, feats))
    {
        let mut line = format!("  {tname:>10}");
        for &layer in &[0, 8, 16, 24, 33] {
            let mut union: HashSet<usize> = HashSet::new();
            for entity_feats in t_features {
                union.extend(entity_feats[layer].iter());
            }
            line.push_str(&format!(" {union:>5}", union = union.len()));
        }
        println!("{line}");
    }

    println!("\n=== Done ===");
    Ok(())
}
