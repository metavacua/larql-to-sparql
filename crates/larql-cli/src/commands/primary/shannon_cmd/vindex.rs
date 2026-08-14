//! Vindex-backed encode/decode — scoring straight from a quantised index.

use super::*;

pub(super) struct VindexShannonRuntime {
    pub(super) weights: larql_inference::ModelWeights,
    pub(super) tokenizer: tokenizers::Tokenizer,
    pub(super) index: larql_vindex::VectorIndex,
    pub(super) backend: Box<dyn larql_compute::ComputeBackend>,
}

/// Build the Metal compute backend for `--metal`, or a clear error when the
/// binary lacks the backend or the host lacks a device. Delegates to the
/// shared registry-backed factory in `backend_select`.
pub(super) fn metal_backend_box(
) -> Result<Box<dyn larql_compute::ComputeBackend>, Box<dyn std::error::Error>> {
    crate::backend_select::backend_for_metal_flag(true)
}

pub(super) fn load_vindex_runtime(
    vindex: &Path,
    metal: bool,
) -> Result<VindexShannonRuntime, Box<dyn std::error::Error>> {
    if !metal {
        return Err("--vindex Shannon encode/decode currently requires --metal".into());
    }

    eprintln!("loading vindex {}...", vindex.display());
    let start = Instant::now();
    let cfg = larql_vindex::load_vindex_config(vindex)?;
    if cfg.quant != larql_vindex::QuantFormat::Q4K {
        return Err(format!(
            "--vindex fast Shannon path requires Q4K, found {:?}",
            cfg.quant
        )
        .into());
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(vindex, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex, &mut cb)?;
    index.load_attn_kquant(vindex)?;
    index.load_interleaved_kquant(vindex)?;
    let _ = index.load_lm_head_kquant(vindex);
    // `larql_compute::default_backend()` always returns CPU since the
    // GPU-backend extraction (ADR-019) — GPU selection is the caller's
    // responsibility. The fused Q4 forced-token scorer
    // (`stream_forced_full_logits`) requires Metal, so build it directly here
    // when `--metal` is set, mirroring `walk_cmd.rs` and
    // `bench/local_runtime.rs`. The previous `default_backend()` call silently
    // fell through to CPU and then errored out at "forced Shannon logits
    // require a fused Q4 backend", making the `encode`/`decode` --metal path
    // unreachable on every machine.
    let backend: Box<dyn larql_compute::ComputeBackend> = metal_backend_box()?;
    if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        return Err("Metal/Q4 backend is not available".into());
    }
    eprintln!(
        "loaded vindex. {} layers, hidden_size={}, backend={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        backend.name(),
        start.elapsed().as_secs_f64()
    );

    Ok(VindexShannonRuntime {
        weights,
        tokenizer,
        index,
        backend,
    })
}

pub(super) fn run_encode_vindex(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex = args.vindex.as_ref().ok_or("--vindex missing")?;
    let text = read_text(&args.input, args.bytes)?;
    let mut rt = load_vindex_runtime(vindex, args.metal)?;
    let ids = encode_prompt(&rt.tokenizer, &*rt.weights.arch, &text)?;
    if ids.len() < 2 {
        return Err("input must tokenize to at least one encoded token".into());
    }

    // Diagnostic: run two forced passes in ONE process and compare the
    // per-step quantized frequency tables. Distinguishes per-dispatch GPU
    // non-determinism (in-process passes disagree) from cross-process-only
    // drift (in-process agree). Gated so it never runs in the demo path.
    if std::env::var("LARQL_SHANNON_SELFTEST").is_ok() {
        return run_encode_vindex_selftest(&mut rt, &ids);
    }

    eprintln!(
        "encoding {} bytes as {} target tokens with KV-cached vindex blocks...",
        text.len(),
        ids.len() - 1
    );
    let pb = progress_bar((ids.len() - 1) as u64, "encoding");
    let mut blocks = Vec::new();
    let mut prefill_ms = 0.0;
    let mut decode_ms = Vec::new();
    let mut start = 0usize;
    while start + 1 < ids.len() {
        let end = (start + VINDEX_BLOCK_TARGET_TOKENS + 1).min(ids.len());
        let block_ids = &ids[start..end];
        let mut encoder = ArithmeticEncoder::new();
        let forced = larql_inference::layer_graph::generate::stream_forced_full_logits(
            &mut rt.weights,
            block_ids[0],
            block_ids.len() - 1,
            &rt.index,
            rt.backend.as_ref(),
            |step, logits| {
                let target = block_ids[step + 1];
                let counts =
                    quantized_counts(logits).map_err(|e| format!("quantize logits: {e}"))?;
                let (low, high) =
                    interval_for_symbol(&counts, target).map_err(|e| format!("interval: {e}"))?;
                encoder.encode(low, high, FREQ_TOTAL);
                pb.inc(1);
                Ok(target)
            },
        )?;
        prefill_ms += forced.prefill_ms;
        decode_ms.extend(forced.decode_ms);
        blocks.push(VindexShannonBlock {
            first_token: block_ids[0],
            target_tokens: (block_ids.len() - 1) as u64,
            payload: encoder.finish(),
        });
        start = end - 1;
    }
    pb.finish_and_clear();

    let payload = encode_vindex_blocks(&blocks);
    let blob = ShannonFile {
        // The vindex fast path is full-context within the GPU KV cache. Use
        // u32::MAX so old CPU decode treats this as "effectively unlimited"
        // for normal demo-sized files.
        context: u32::MAX,
        first_token: ids[0],
        target_tokens: (ids.len() - 1) as u64,
        original_bytes: text.len() as u64,
        payload,
    };
    let bytes = blob.to_bytes();
    fs::write(&args.out, &bytes)?;

    let chars = text.chars().count().max(1) as f64;
    println!("original:        {:>10} bytes", text.len());
    println!("payload:         {:>10} bytes", blob.payload.len());
    println!("file:            {:>10} bytes", bytes.len());
    println!("tokens:          {:>10}", ids.len() - 1);
    println!(
        "ratio(payload):  {:>10.2}x",
        text.len() as f64 / blob.payload.len().max(1) as f64
    );
    println!(
        "bits/char:       {:>10.3}",
        blob.payload.len() as f64 * 8.0 / chars
    );
    println!("blocks:          {:>10}", blocks.len());
    println!("prefill total:   {:>10.1} ms", prefill_ms);
    if !decode_ms.is_empty() {
        let avg = decode_ms.iter().sum::<f64>() / decode_ms.len() as f64;
        println!("decode avg:      {:>10.1} ms/token", avg);
    }
    println!("wrote: {}", args.out.display());
    Ok(())
}

/// Run two forced passes over the first block in one process and compare the
/// per-step quantized frequency tables. The arithmetic coder desyncs at the
/// first step whose count table differs, so this reports exactly where (and
/// by how much) the GPU forward drifts. See the `--metal` round-trip notes in
/// `docs/replay/shannon-transformers-the-same.md`.
pub(super) fn run_encode_vindex_selftest(
    rt: &mut VindexShannonRuntime,
    ids: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let n = ids.len().min(VINDEX_BLOCK_TARGET_TOKENS + 1);
    let block_ids = ids[..n].to_vec();
    eprintln!(
        "[selftest] two in-process forced passes over {} forced tokens",
        block_ids.len() - 1
    );

    // Per step: (bits at the forced target, FNV fingerprint of the full count
    // table, cumulative-low of the target symbol).
    fn run_pass(
        rt: &mut VindexShannonRuntime,
        block_ids: &[u32],
    ) -> Result<Vec<(f64, u64, u32)>, String> {
        let mut per_step = Vec::with_capacity(block_ids.len());
        larql_inference::layer_graph::generate::stream_forced_full_logits(
            &mut rt.weights,
            block_ids[0],
            block_ids.len() - 1,
            &rt.index,
            rt.backend.as_ref(),
            |step, logits| {
                let target = block_ids[step + 1];
                let counts = quantized_counts(logits).map_err(|e| e.to_string())?;
                let (low, _high) =
                    interval_for_symbol(&counts, target).map_err(|e| e.to_string())?;
                let bits = bits_for_target(logits, target).map_err(|e| e.to_string())?;
                let mut fp: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
                for (i, &c) in counts.iter().enumerate() {
                    fp ^= (c as u64).wrapping_mul((i as u64).wrapping_add(1));
                    fp = fp.wrapping_mul(0x100000001b3);
                }
                per_step.push((bits, fp, low));
                Ok(target)
            },
        )?;
        Ok(per_step)
    }

    let a = run_pass(rt, &block_ids)?;
    let b = run_pass(rt, &block_ids)?;

    let mut first_div: Option<usize> = None;
    let mut max_bits_delta = 0.0_f64;
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        max_bits_delta = max_bits_delta.max((pa.0 - pb.0).abs());
        if first_div.is_none() && pa.1 != pb.1 {
            first_div = Some(i);
        }
    }

    eprintln!("[selftest] steps compared:                 {}", a.len());
    eprintln!(
        "[selftest] max |Δ bits(target)| across steps: {:.6}",
        max_bits_delta
    );
    match first_div {
        Some(i) => {
            eprintln!(
                "[selftest] first step with DIFFERING count table: {} of {}",
                i,
                a.len()
            );
            eprintln!(
                "[selftest]   cum_low(target) A={} B={}  Δbits={:.6}",
                a[i].2,
                b[i].2,
                (a[i].0 - b[i].0).abs()
            );
            eprintln!(
                "[selftest] VERDICT: per-dispatch non-determinism — two passes in ONE process"
            );
            eprintln!("[selftest]          disagree, so the coder cannot round-trip on this path.");
        }
        None => {
            eprintln!(
                "[selftest] count tables IDENTICAL at every step across two in-process passes."
            );
            eprintln!(
                "[selftest] VERDICT: in-process is deterministic; drift is cross-process only"
            );
            eprintln!("[selftest]          (buffer init / dispatch geometry) and may be fixable.");
        }
    }
    Ok(())
}

pub(super) fn run_decode_vindex(args: DecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex = args.vindex.as_ref().ok_or("--vindex missing")?;
    let mut raw = Vec::new();
    fs::File::open(&args.input)?.read_to_end(&mut raw)?;
    let blob = ShannonFile::from_bytes(&raw)?;
    let mut rt = load_vindex_runtime(vindex, args.metal)?;
    let blocks = parse_vindex_blocks(&blob.payload)?.unwrap_or_else(|| {
        vec![VindexShannonBlock {
            first_token: blob.first_token,
            target_tokens: blob.target_tokens,
            payload: blob.payload.clone(),
        }]
    });

    eprintln!(
        "decoding {} target tokens with KV-cached vindex blocks...",
        blob.target_tokens
    );
    let pb = progress_bar(blob.target_tokens, "decoding");
    let mut ids = Vec::with_capacity(blob.target_tokens as usize + 1);
    let mut prefill_ms = 0.0;
    let mut decode_ms = Vec::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        let mut decoder = ArithmeticDecoder::new(&block.payload);
        let forced = larql_inference::layer_graph::generate::stream_forced_full_logits(
            &mut rt.weights,
            block.first_token,
            block.target_tokens as usize,
            &rt.index,
            rt.backend.as_ref(),
            |_step, logits| {
                let counts =
                    quantized_counts(logits).map_err(|e| format!("quantize logits: {e}"))?;
                let value = decoder.scaled_value(FREQ_TOTAL);
                let (symbol, low, high) =
                    symbol_for_value(&counts, value).map_err(|e| format!("decode symbol: {e}"))?;
                decoder.decode(low, high, FREQ_TOTAL);
                pb.inc(1);
                Ok(symbol)
            },
        )?;
        if block_idx == 0 {
            ids.push(block.first_token);
        }
        ids.extend_from_slice(&forced.forced_tokens);
        prefill_ms += forced.prefill_ms;
        decode_ms.extend(forced.decode_ms);
    }
    pb.finish_and_clear();

    let text = rt
        .tokenizer
        .decode(&ids, true)
        .map_err(|e| format!("decode error: {e}"))?;
    fs::write(&args.out, text.as_bytes())?;
    println!("decoded:         {:>10} bytes", text.len());
    println!("expected:        {:>10} bytes", blob.original_bytes);
    println!("blocks:          {:>10}", blocks.len());
    println!("prefill total:   {:>10.1} ms", prefill_ms);
    if !decode_ms.is_empty() {
        let avg = decode_ms.iter().sum::<f64>() / decode_ms.len() as f64;
        println!("decode avg:      {:>10.1} ms/token", avg);
    }
    println!("wrote: {}", args.out.display());
    Ok(())
}
