use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{predict, trace_forward, InferenceModel};
use serde::Serialize;

/// Default templates — the same factual patterns as chuk-larql.
const DEFAULT_TEMPLATES: &[(&str, &str)] = &[
    ("capital-of", "The capital of {subject} is"),
    ("language-of", "The official language of {subject} is"),
    ("currency", "The currency of {subject} is the"),
    ("continent", "{subject} is located in the continent of"),
    ("birthplace", "{subject} was born in"),
    ("nationality", "The nationality of {subject} is"),
    ("known-for", "{subject} is best known as a"),
    ("located-in", "{subject} is located in"),
    ("author-of", "The author of {subject} is"),
    ("spoken-in", "{subject} is spoken in"),
    ("birth-year", "{subject} was born in the year"),
    ("death-year", "{subject} died in the year"),
];

/// Default entities — diverse enough to find general patterns.
const DEFAULT_ENTITIES: &[&str] = &[
    "France",
    "Germany",
    "Japan",
    "Brazil",
    "Egypt",
    "Australia",
    "India",
    "Canada",
    "Italy",
    "China",
    "Mozart",
    "Einstein",
    "Shakespeare",
    "Cleopatra",
    "Darwin",
    "London",
    "Tokyo",
    "Paris",
    "Cairo",
    "Sydney",
];

#[derive(Args)]
pub struct ExtractRoutesArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Output JSON file for the routing table.
    #[arg(short, long)]
    output: PathBuf,

    /// Top features to capture per layer per forward pass.
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Minimum absolute activation to record.
    #[arg(long, default_value = "1.0")]
    min_activation: f32,

    /// Comma-separated entities (overrides defaults).
    #[arg(short, long)]
    entities: Option<String>,

    /// Comma-separated layers to capture (default: all).
    #[arg(long)]
    layers: Option<String>,
}

#[derive(Serialize)]
struct FeatureHit {
    layer: usize,
    feature: usize,
    activation: f32,
}

#[derive(Serialize)]
struct RouteEntry {
    relation: String,
    template: String,
    entity: String,
    prompt: String,
    prediction: String,
    confidence: f64,
    features: Vec<FeatureHit>,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct RouteTable {
    model_name: String,
    num_passes: usize,
    total_elapsed_ms: f64,
    routes: Vec<RouteEntry>,
}

pub fn run(args: ExtractRoutesArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let load_elapsed = start.elapsed();
    eprintln!(
        "  {} layers, hidden_size={}, vocab_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        model.weights().vocab_size,
        load_elapsed.as_secs_f64()
    );

    let entities: Vec<String> = if let Some(ref e) = args.entities {
        e.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        DEFAULT_ENTITIES.iter().map(|s| s.to_string()).collect()
    };

    let num_layers = model.num_layers();
    let capture_layers: Vec<usize> = if let Some(ref spec) = args.layers {
        parse_layer_spec(spec, num_layers)?
    } else {
        (0..num_layers).collect()
    };

    let total_passes = DEFAULT_TEMPLATES.len() * entities.len();
    eprintln!(
        "  {} templates x {} entities = {} passes",
        DEFAULT_TEMPLATES.len(),
        entities.len(),
        total_passes
    );
    eprintln!(
        "  Capturing {} layers, top-{} features, min_activation={}",
        capture_layers.len(),
        args.top_k,
        args.min_activation
    );
    eprintln!();

    let total_start = Instant::now();
    let mut routes = Vec::new();
    let mut completed = 0;

    for &(relation, template) in DEFAULT_TEMPLATES {
        for entity in &entities {
            let prompt = template.replace("{subject}", entity);
            let pass_start = Instant::now();

            // Tokenize
            let encoding = model
                .tokenizer()
                .encode(prompt.as_str(), true)
                .map_err(|e| format!("tokenize error: {e}"))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            // Run forward pass with activation capture
            let trace = trace_forward(
                model.weights(),
                &token_ids,
                &capture_layers,
                true,
                args.top_k,
            );

            // Get prediction from a full forward pass
            let pred_result = predict(model.weights(), model.tokenizer(), &token_ids, 1);
            let (prediction, confidence) = pred_result
                .predictions
                .first()
                .map(|(t, p)| (t.clone(), *p))
                .unwrap_or_default();

            // Collect feature hits
            let mut features = Vec::new();
            for (layer, layer_acts) in &trace.activations {
                for &(feat_idx, act) in layer_acts {
                    if act.abs() >= args.min_activation {
                        features.push(FeatureHit {
                            layer: *layer,
                            feature: feat_idx,
                            activation: (act * 10000.0).round() / 10000.0,
                        });
                    }
                }
            }

            let elapsed_ms = pass_start.elapsed().as_secs_f64() * 1000.0;

            completed += 1;
            let total_elapsed = total_start.elapsed().as_secs_f64();
            let rate = completed as f64 / total_elapsed;
            let eta = (total_passes - completed) as f64 / rate;

            let status = if prediction.is_empty() { "??" } else { "OK" };
            eprintln!(
                "  [{:3}/{total_passes}] {relation:15} {entity:15} \
                 → {prediction:15} ({confidence:.2}) \
                 {:4} features  [{status}] ETA {eta:.0}s",
                completed,
                features.len(),
            );

            routes.push(RouteEntry {
                relation: relation.to_string(),
                template: template.to_string(),
                entity: entity.clone(),
                prompt,
                prediction,
                confidence,
                features,
                elapsed_ms,
            });
        }
    }

    let total_elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    let table = RouteTable {
        model_name: args.model.clone(),
        num_passes: routes.len(),
        total_elapsed_ms,
        routes,
    };

    // Count unique features
    let mut unique_features = std::collections::HashSet::new();
    for route in &table.routes {
        for f in &route.features {
            unique_features.insert((f.layer, f.feature));
        }
    }

    // Save
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&table)?;
    std::fs::write(&args.output, json)?;

    eprintln!();
    eprintln!("Route table saved: {}", args.output.display());
    eprintln!("  Passes:          {}", table.num_passes);
    eprintln!("  Unique features: {}", unique_features.len());
    eprintln!("  Total time:      {:.1}s", total_elapsed_ms / 1000.0);
    eprintln!(
        "  Avg per pass:    {:.0}ms",
        total_elapsed_ms / table.num_passes.max(1) as f64
    );

    // Summary by relation
    eprintln!();
    eprintln!("Routes by relation:");
    for &(relation, _) in DEFAULT_TEMPLATES {
        let rel_routes: Vec<&RouteEntry> = table
            .routes
            .iter()
            .filter(|r| r.relation == relation)
            .collect();
        let predicted = rel_routes.iter().filter(|r| !r.prediction.is_empty()).count();
        let total_feats: usize = rel_routes.iter().map(|r| r.features.len()).sum();
        eprintln!(
            "  {relation:20}  {predicted}/{} predicted  {total_feats} features",
            rel_routes.len()
        );
    }

    Ok(())
}

fn parse_layer_spec(spec: &str, num_layers: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start.parse()?;
            let e: usize = end.parse()?;
            layers.extend(s..=e.min(num_layers - 1));
        } else {
            let l: usize = part.parse()?;
            if l < num_layers {
                layers.push(l);
            }
        }
    }
    Ok(layers)
}
