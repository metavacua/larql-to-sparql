//! I/O-bound capture driver: loads the model exactly like the bench
//! remote-FFN runtime, runs single-stream decode per prompt against the live
//! expert server with the residual-capture sink attached, and writes the
//! pool. Coverage-excluded (needs live model weights + a running server);
//! all validation and formatting logic lives in `capture_format.rs`.

use super::args::CaptureArgs;
use super::capture_format::{CapturePool, RoutingCapture};

pub(super) fn run_capture(args: &CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::{
        generate_with_remote_ffn_batch_captured, LayerShardedBackend, ResidualCaptureSink,
    };
    use std::time::Duration;

    let vindex_path = crate::commands::primary::cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display(),
        )
        .into());
    }

    let prompts_raw = std::fs::read_to_string(&args.prompts_file)
        .map_err(|e| format!("read prompts file {}: {e}", args.prompts_file.display()))?;
    let prompts: Vec<String> = prompts_raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .take(args.num_prompts)
        .map(String::from)
        .collect();
    if prompts.len() < args.num_prompts {
        return Err(format!(
            "prompts file has {} usable prompts, --num-prompts asked for {}",
            prompts.len(),
            args.num_prompts
        )
        .into());
    }

    // An explicit `--metal` with no usable device is a loud error, not a CPU
    // fallback — a DEC capture pool must not silently come off the wrong
    // substrate (see `backend_select`).
    let backend: Box<dyn larql_compute::ComputeBackend> =
        crate::backend_select::backend_for_metal_flag(args.metal)?;

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    let timeout = Duration::from_secs(args.ffn_timeout_secs);
    eprintln!("Connecting to remote FFN at {}…", args.ffn);
    let remote = LayerShardedBackend::connect(&args.ffn, timeout)
        .map_err(|e| format!("failed to connect to remote FFN: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  FFN:        remote  ({})", args.ffn);

    // `--routing` needs a routable MoE model: `top_k` records come from the
    // arch and the capture computes `MoeRouterWeights::route` per layer.
    // Loud error, not a silent walk-ffn-only pool (a DEC routed arm fed a
    // routing-less pool would be a corrupted run with no signal).
    let routing_top_k = if args.routing {
        let arch = &*weights.arch;
        if !arch.is_hybrid_moe() && !arch.is_moe() {
            return Err(format!(
                "--routing requires an MoE model, but {} ({}) has no experts",
                args.model,
                arch.family()
            )
            .into());
        }
        let top_k = arch.num_experts_per_token();
        if top_k == 0 {
            return Err(format!(
                "--routing: model {} reports num_experts_per_token == 0",
                args.model
            )
            .into());
        }
        Some(top_k)
    } else {
        None
    };

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(&vindex_path);
    // The sink records one entry per decode-loop step; the first generated
    // token comes straight from prefill logits (no captured step), so ask
    // for steps+1 tokens to get `steps` captured entries (EOS may still cut
    // a prompt short — the pool truncates to the min across prompts).
    let max_tokens = args.steps + 1;

    let mut sinks: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(prompts.len());
    let mut routing_capture = routing_top_k.map(|top_k| RoutingCapture {
        top_k,
        raw: Vec::with_capacity(prompts.len()),
        normed: Vec::with_capacity(prompts.len()),
        routing: Vec::with_capacity(prompts.len()),
    });
    for (i, prompt) in prompts.iter().enumerate() {
        let wrapped =
            larql_inference::chat::render_user_prompt(&vindex_path, weights.arch.family(), prompt)
                .unwrap_or_else(|_| prompt.clone());
        let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped)
            .map_err(|e| format!("tokenise prompt {i}: {e}"))?;

        let mut sink = if args.routing {
            ResidualCaptureSink::with_routing()
        } else {
            ResidualCaptureSink::default()
        };
        generate_with_remote_ffn_batch_captured(
            &weights,
            &tokenizer,
            prompt_ids,
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            1,
            Some(&mut sink),
        )
        .map_err(|e| format!("capture decode failed on prompt {i}: {e}"))?;

        if args.verbose {
            eprintln!(
                "[capture] prompt {}/{}: {} steps",
                i + 1,
                prompts.len(),
                sink.steps.len()
            );
        }
        if sink.steps.is_empty() {
            return Err(format!(
                "prompt {i} ({prompt:?}) produced no decode steps (immediate EOS) — \
                 replace it in the prompts file"
            )
            .into());
        }
        if let Some(rc) = routing_capture.as_mut() {
            rc.raw.push(sink.raw_steps);
            rc.normed.push(sink.normed_steps);
            rc.routing.push(sink.routing_steps);
        }
        sinks.push(sink.steps);
    }

    let created_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let manifest = CapturePool::write_with_routing(
        &args.out,
        &args.model,
        hidden,
        num_layers,
        &prompts,
        &sinks,
        routing_capture.as_ref(),
        created_unix,
    )?;

    eprintln!(
        "Captured pool: {} prompts × {} steps × {} layers × hidden {} → {} ({} MB){}",
        manifest.prompts.len(),
        manifest.steps,
        manifest.num_layers,
        manifest.hidden_size,
        args.out.display(),
        manifest.expected_bytes() / (1024 * 1024),
        match &manifest.routing {
            Some(r) => format!(" + routed sidecars (top_k {})", r.top_k),
            None => String::new(),
        },
    );
    Ok(())
}
