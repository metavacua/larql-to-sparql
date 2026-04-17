use std::time::Instant;

use larql_inference::forward::predict_with_temperature;
use clap::Args;
use larql_inference::{
    calibrate_scalar_gains, predict, predict_with_ffn, predict_with_router, predict_with_strategy,
    FfnBackend, GateIndex, GraphFfn, InferenceModel, LayerFfnRouter, LayerMode, RouteFfn,
    RouteGuidedFfn, RouteTable, SparseFfn, WeightFfn,
};

#[derive(Args)]
pub struct PredictArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prompt text to predict the next token for.
    #[arg(short, long)]
    prompt: String,

    /// Number of top predictions to show.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,
    /// Sampling temperature (default 1.0). < 1.0 = more focused, > 1.0 = more random.
    #[arg(short = 't', long, default_value = "1.0")]
    temperature: f32,

    /// FFN backend: "weights" (dense, default), "sparse:K" (top-K features),
    /// "graph" (uses --gate-index), or layer ranges like "weights:0-25,sparse100:26-33".
    #[arg(long, default_value = "weights")]
    ffn: String,

    /// Pre-built gate index file (from `larql index-gates`). Required for --ffn graph.
    #[arg(long)]
    gate_index: Option<std::path::PathBuf>,

    /// Top tokens for graph FFN residual matching. [default: 10]
    #[arg(long, default_value = "10")]
    graph_top_tokens: usize,

    /// Max features for graph FFN per position. [default: 200]
    #[arg(long, default_value = "200")]
    graph_top_k: usize,

    /// Route table file (from `larql extract-routes`). Required for --ffn routes.
    #[arg(long)]
    routes: Option<std::path::PathBuf>,

    /// Relation pattern for route-based FFN (e.g., "capital-of").
    #[arg(long)]
    relation: Option<String>,

    /// Entity for route-based FFN (e.g., "France").
    #[arg(long)]
    entity: Option<String>,

    /// Max features per layer for route-based FFN. [default: 100]
    #[arg(long, default_value = "100")]
    route_top_k: usize,

    /// Compare all backends side by side.
    #[arg(long)]
    compare: bool,

    /// Layer strategy with scalar bypass: "dense:0-8,scalar:9-14,dense:15-33".
    /// Scalar gains are auto-calibrated from a forward pass on the same prompt.
    /// Supports: dense, sparse<K>, scalar, walk.
    #[arg(long)]
    mode: Option<String>,
}

pub fn run(args: PredictArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let load_elapsed = start.elapsed();
    eprintln!(
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        load_elapsed.as_secs_f64()
    );

    eprintln!("Prompt: {:?}", args.prompt);

    let token_ids = larql_inference::encode_prompt(
        model.tokenizer(),
        &*model.weights().arch,
        args.prompt.as_str(),
    )
    .map_err(|e| format!("tokenize error: {e}"))?;
    eprintln!("  {} tokens: {:?}", token_ids.len(), token_ids);

    if args.compare {
        run_comparison(&model, &token_ids, args.top_k, &args)?;
    } else {
        run_single(&model, &token_ids, args.top_k, &args)?;
    }

    Ok(())
}

fn run_single(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    args: &PredictArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();

    // --mode takes precedence: supports scalar bypass
    if let Some(ref mode_spec) = args.mode {
        return run_with_mode(model, token_ids, top_k, mode_spec);
    }

    let ffn_spec = args.ffn.as_str();

    // Parse FFN spec
    if ffn_spec == "weights" {
        eprintln!("FFN: weights (dense)");
        let start = Instant::now();
        let result = predict_with_temperature(weights, model.tokenizer(), token_ids, top_k, args.temperature);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions("weights", &result.predictions);
    } else if let Some(k_str) = ffn_spec.strip_prefix("sparse:") {
        let k: usize = k_str.parse().map_err(|_| format!("invalid K: {k_str}"))?;
        eprintln!("FFN: sparse (top-{k})");
        let ffn = SparseFfn { weights, top_k: k };
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &ffn);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions(&format!("sparse:{k}"), &result.predictions);
    } else if ffn_spec == "graph" {
        let index_path = args.gate_index.as_ref().ok_or(
            "--gate-index required for --ffn graph. Build with: larql index-gates <model> -o gates.index",
        )?;
        eprintln!("Loading gate index: {}", index_path.display());
        let load_start = Instant::now();
        let gate_index = GateIndex::load(index_path, args.graph_top_tokens)?;
        eprintln!(
            "  {} layers, {} entries ({:.1}s)",
            gate_index.num_layers(),
            gate_index.total_entries(),
            load_start.elapsed().as_secs_f64()
        );

        let ffn = GraphFfn {
            weights,
            gate_index: &gate_index,
            top_k: args.graph_top_k,
        };
        eprintln!(
            "FFN: graph (top_tokens={}, top_k={})",
            args.graph_top_tokens, args.graph_top_k
        );
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &ffn);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions("graph", &result.predictions);
    } else if ffn_spec == "routes" {
        let routes_path = args.routes.as_ref().ok_or(
            "--routes required for --ffn routes. Build with: larql extract-routes <model> -o routes.json",
        )?;
        let relation = args.relation.as_deref().ok_or(
            "--relation required for --ffn routes (e.g., --relation capital-of)",
        )?;
        let entity = args.entity.as_deref().ok_or(
            "--entity required for --ffn routes (e.g., --entity France)",
        )?;

        eprintln!("Loading route table: {}", routes_path.display());
        let load_start = Instant::now();
        let route_table = RouteTable::load(routes_path)?;
        eprintln!(
            "  {} routes, relations: {:?} ({:.1}s)",
            route_table.num_routes(),
            route_table.relations(),
            load_start.elapsed().as_secs_f64()
        );

        // Pure route FFN (all layers)
        let route_ffn = RouteFfn {
            weights,
            route_table: &route_table,
            relation: relation.to_string(),
            entity: entity.to_string(),
            top_k: args.route_top_k,
        };

        eprintln!(
            "FFN: routes (relation={}, entity={}, top_k={})",
            relation, entity, args.route_top_k
        );

        // Run pure routes
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &route_ffn);
        eprintln!("  Pure routes: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions(&format!("routes:{relation}:{entity}"), &result.predictions);

        // Route-guided: uses route table for feature SELECTION,
        // computes actual gate @ hidden for those features
        let guided_ffn = RouteGuidedFfn {
            weights,
            route_table: &route_table,
            relation: relation.to_string(),
            entity: entity.to_string(),
            top_k: args.route_top_k,
        };

        // Pure route-guided (all layers)
        eprintln!("FFN: route-guided (all layers, top_k={})", args.route_top_k);
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &guided_ffn);
        eprintln!("  Route-guided: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions("route-guided (all)", &result.predictions);

        // Hybrids: dense early layers, route-guided for factual layers
        let weight_ffn = WeightFfn { weights };
        let num_layers = weights.num_layers;

        for switch_layer in [
            num_layers - 2,
            num_layers - 4,
            num_layers - 8,
            num_layers * 3 / 4,
        ] {
            // Route-guided hybrid
            let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; num_layers];
            (switch_layer..num_layers).for_each(|layer| {
                backends[layer] = &guided_ffn;
            });
            let router = LayerFfnRouter::per_layer(backends);

            let label = format!(
                "weights:0-{},guided:{}-{}",
                switch_layer - 1, switch_layer, num_layers - 1
            );
            let start = Instant::now();
            let result = predict_with_router(weights, model.tokenizer(), token_ids, top_k, &router);
            let elapsed = start.elapsed();

            let top1 = result.predictions.first()
                .map(|(t, p)| format!("{t} ({:.1}%)", p * 100.0))
                .unwrap_or_default();
            eprintln!("  {label}: {top1} [{:.1}s]", elapsed.as_secs_f64());
            print_predictions(&label, &result.predictions);
        }
    } else if ffn_spec.contains(':') && ffn_spec.contains(',') {
        // Layer-range spec: "weights:0-25,sparse100:26-33"
        run_with_layer_spec(model, token_ids, top_k, ffn_spec)?;
    } else {
        return Err(format!(
            "unknown --ffn value: {ffn_spec}. Use 'weights', 'sparse:K', 'graph', 'routes', or layer ranges"
        )
        .into());
    }

    Ok(())
}

fn run_with_layer_spec(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    spec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let weight_ffn = WeightFfn { weights };

    // Parse spec like "weights:0-25,sparse100:26-33"
    // We need to hold SparseFfn instances alive, so collect them first
    let mut sparse_backends: Vec<SparseFfn> = Vec::new();

    // First pass: figure out which layers need which backend
    let mut layer_specs: Vec<(&str, usize, usize)> = Vec::new(); // (backend_name, start, end)
    for part in spec.split(',') {
        let (backend_name, range) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid layer spec: {part}"))?;
        let (start, end) = if range.contains('-') {
            let (a, b) = range
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {range}"))?;
            (a.parse::<usize>()?, b.parse::<usize>()?)
        } else {
            let l = range.parse::<usize>()?;
            (l, l)
        };
        layer_specs.push((backend_name, start, end));

        // Pre-create sparse backends
        if let Some(k_str) = backend_name.strip_prefix("sparse") {
            let k: usize = k_str.parse().unwrap_or(100);
            sparse_backends.push(SparseFfn { weights, top_k: k });
        }
    }

    // Build per-layer backend array
    let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; num_layers];
    let mut sparse_idx = 0;
    for (backend_name, start, end) in &layer_specs {
        let backend: &dyn FfnBackend = if *backend_name == "weights" {
            &weight_ffn
        } else if backend_name.starts_with("sparse") {
            let b = &sparse_backends[sparse_idx];
            sparse_idx += 1;
            b
        } else {
            return Err(format!("unknown backend: {backend_name}").into());
        };
        (*start..=(*end).min(num_layers - 1)).for_each(|l| {
            backends[l] = backend;
        });
    }

    let router = LayerFfnRouter::per_layer(backends);
    eprintln!("FFN: layer-routed ({spec})");

    let start = Instant::now();
    let result = predict_with_router(weights, model.tokenizer(), token_ids, top_k, &router);
    eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
    print_predictions(spec, &result.predictions);

    Ok(())
}

fn run_with_mode(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    spec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();
    let num_layers = weights.num_layers;

    // Parse mode spec: "dense:0-8,scalar:9-14,dense:15-33"
    #[derive(Debug, Clone)]
    enum BackendKind {
        Dense,
        Sparse(usize),
        Scalar,
    }

    let mut layer_kinds = vec![BackendKind::Dense; num_layers];
    for part in spec.split(',') {
        let (name, range) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid mode spec: {part}"))?;
        let (start, end) = if range.contains('-') {
            let (a, b) = range.split_once('-').unwrap();
            (a.parse::<usize>()?, b.parse::<usize>()?)
        } else {
            let l = range.parse::<usize>()?;
            (l, l)
        };

        let kind = if name == "dense" {
            BackendKind::Dense
        } else if name == "scalar" {
            BackendKind::Scalar
        } else if let Some(k_str) = name.strip_prefix("sparse") {
            let k: usize = if k_str.is_empty() { 100 } else { k_str.parse()? };
            BackendKind::Sparse(k)
        } else {
            return Err(format!("unknown mode: {name}. Use dense, scalar, sparse<K>").into());
        };

        for l in start..=end.min(num_layers - 1) {
            layer_kinds[l] = kind.clone();
        }
    }

    // Check if any scalar layers
    let has_scalar = layer_kinds.iter().any(|k| matches!(k, BackendKind::Scalar));

    if has_scalar {
        // Calibrate scalar gains from a full forward pass
        eprintln!("Calibrating scalar gains...");
        let cal_start = Instant::now();
        let gains = calibrate_scalar_gains(weights, token_ids);
        eprintln!(
            "  Calibrated {} layers in {:.1}s",
            gains.len(),
            cal_start.elapsed().as_secs_f64()
        );

        // Print the gain schedule
        let scalar_layers: Vec<usize> = layer_kinds
            .iter()
            .enumerate()
            .filter_map(|(l, k)| if matches!(k, BackendKind::Scalar) { Some(l) } else { None })
            .collect();
        eprintln!("  Scalar layers: {:?}", scalar_layers);
        for &l in &scalar_layers {
            eprintln!("    L{l}: gain={:.4}", gains[l]);
        }

        // Build FFN backends for non-scalar layers
        let weight_ffn = WeightFfn { weights };
        let sparse_backends: Vec<SparseFfn> = layer_kinds
            .iter()
            .filter_map(|k| {
                if let BackendKind::Sparse(top_k) = k {
                    Some(SparseFfn { weights, top_k: *top_k })
                } else {
                    None
                }
            })
            .collect();

        // Build strategy
        let mut strategy: Vec<LayerMode> = Vec::with_capacity(num_layers);
        let mut sparse_idx = 0;
        for (l, kind) in layer_kinds.iter().enumerate() {
            match kind {
                BackendKind::Dense => {
                    strategy.push(LayerMode::Compute(&weight_ffn));
                }
                BackendKind::Sparse(_) => {
                    strategy.push(LayerMode::Compute(&sparse_backends[sparse_idx]));
                    sparse_idx += 1;
                }
                BackendKind::Scalar => {
                    strategy.push(LayerMode::ScalarGain(gains[l]));
                }
            }
        }

        eprintln!("\nMode: {spec}");
        let start = Instant::now();
        let result = predict_with_strategy(weights, model.tokenizer(), token_ids, top_k, &strategy);
        let elapsed = start.elapsed();

        let compute_layers = layer_kinds
            .iter()
            .filter(|k| !matches!(k, BackendKind::Scalar))
            .count();
        eprintln!(
            "  Forward pass: {:.1}s ({} compute layers, {} scalar bypass)",
            elapsed.as_secs_f64(),
            compute_layers,
            num_layers - compute_layers,
        );
        print_predictions(spec, &result.predictions);

        // Also run dense baseline for comparison
        eprintln!("\nBaseline (dense all layers):");
        let start = Instant::now();
        let baseline = predict(weights, model.tokenizer(), token_ids, top_k);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions("dense (baseline)", &baseline.predictions);
    } else {
        // No scalar — fall back to router
        let weight_ffn = WeightFfn { weights };
        let sparse_backends: Vec<SparseFfn> = layer_kinds
            .iter()
            .filter_map(|k| {
                if let BackendKind::Sparse(top_k) = k {
                    Some(SparseFfn { weights, top_k: *top_k })
                } else {
                    None
                }
            })
            .collect();

        let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; num_layers];
        let mut sparse_idx = 0;
        for (l, kind) in layer_kinds.iter().enumerate() {
            match kind {
                BackendKind::Dense => {}
                BackendKind::Sparse(_) => {
                    backends[l] = &sparse_backends[sparse_idx];
                    sparse_idx += 1;
                }
                BackendKind::Scalar => unreachable!(),
            }
        }
        let router = LayerFfnRouter::per_layer(backends);
        eprintln!("Mode: {spec}");
        let start = Instant::now();
        let result = predict_with_router(weights, model.tokenizer(), token_ids, top_k, &router);
        eprintln!("  Forward pass: {:.1}s", start.elapsed().as_secs_f64());
        print_predictions(spec, &result.predictions);
    }

    Ok(())
}

fn run_comparison(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    args: &PredictArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();

    println!();
    println!(
        "{:<20} {:<15} {:>8} {:>10}  {:<20}",
        "Backend", "Top-1", "Prob", "Time", "Top-3"
    );
    println!("{}", "-".repeat(80));

    // Dense (ground truth)
    let start = Instant::now();
    let dense_result = predict(weights, model.tokenizer(), token_ids, top_k);
    let dense_time = start.elapsed();
    print_comparison_row("weights (dense)", &dense_result.predictions, dense_time);

    // Sparse at various K values
    for k in [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32] {
        let ffn = SparseFfn { weights, top_k: k };
        let start = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &ffn);
        let elapsed = start.elapsed();
        print_comparison_row(&format!("sparse:{k}"), &result.predictions, elapsed);
    }

    // Mixed: weights for early layers, sparse for knowledge layers
    let weight_ffn = WeightFfn { weights };
    let sparse_100 = SparseFfn {
        weights,
        top_k: 100,
    };
    let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; weights.num_layers];
    (26..weights.num_layers).for_each(|l| {
        backends[l] = &sparse_100;
    });
    let router = LayerFfnRouter::per_layer(backends);
    let start = Instant::now();
    let result = predict_with_router(weights, model.tokenizer(), token_ids, top_k, &router);
    let elapsed = start.elapsed();
    print_comparison_row("weights:0-25,sparse100:26-33", &result.predictions, elapsed);

    // Graph FFN — only if --gate-index provided
    if let Some(ref index_path) = args.gate_index {
        eprintln!("  Loading gate index: {}", index_path.display());
        let gate_index = GateIndex::load(index_path, args.graph_top_tokens)?;
        eprintln!(
            "  {} layers, {} entries",
            gate_index.num_layers(),
            gate_index.total_entries()
        );

        for total_k in [1000, 500, 200, 100] {
            let graph_ffn = GraphFfn {
                weights,
                gate_index: &gate_index,
                top_k: total_k,
            };
            let start = Instant::now();
            let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &graph_ffn);
            let elapsed = start.elapsed();
            print_comparison_row(&format!("graph:{total_k}"), &result.predictions, elapsed);
        }

        // Hybrid: weights for early layers, graph FFN for late layers
        let graph_200 = GraphFfn {
            weights,
            gate_index: &gate_index,
            top_k: 200,
        };
        let mut hybrid_backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; weights.num_layers];
        (26..weights.num_layers).for_each(|l| {
            hybrid_backends[l] = &graph_200;
        });
        let hybrid_router = LayerFfnRouter::per_layer(hybrid_backends);
        let start = Instant::now();
        let result =
            predict_with_router(weights, model.tokenizer(), token_ids, top_k, &hybrid_router);
        let elapsed = start.elapsed();
        print_comparison_row("weights:0-25,graph200:26-33", &result.predictions, elapsed);
    }

    Ok(())
}

fn print_predictions(label: &str, predictions: &[(String, f64)]) {
    println!();
    println!("Top predictions ({label}):");
    for (i, (token, prob)) in predictions.iter().enumerate() {
        println!(
            "  {:2}. {:20} {:.4} ({:.2}%)",
            i + 1,
            token,
            prob,
            prob * 100.0
        );
    }
}

fn print_comparison_row(label: &str, predictions: &[(String, f64)], elapsed: std::time::Duration) {
    let (top1, prob1) = predictions
        .first()
        .map(|(t, p)| (t.as_str(), *p))
        .unwrap_or(("?", 0.0));

    let top3: String = predictions
        .iter()
        .take(3)
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "{:<20} {:<15} {:>7.2}% {:>8.0}ms  {:<20}",
        label,
        top1,
        prob1 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
        top3,
    );
}
