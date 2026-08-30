//! Per-layer diagnostics: layer/slot/repeat scoring and their summaries.

use super::*;

pub(super) fn run_layers(args: LayersArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, args.stride)?;
    let text = read_text(&args.corpus, args.bytes)?;
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }
    let weights = model.weights();
    let n_layers = weights.num_layers;
    let n_captures = n_layers + 1;

    eprintln!(
        "scoring {} target tokens over {} bytes across {} layers...",
        ids.len() - 1,
        text.len(),
        n_layers,
    );

    let mut layer_summaries: Vec<LayerSummary> =
        (0..n_captures).map(|_| LayerSummary::default()).collect();

    let pb = progress_bar((ids.len() - 1) as u64, "layers");
    let mut target_start = 1usize;
    while target_start < ids.len() {
        let target_end = (target_start + args.stride).min(ids.len());
        let prefix_start = target_end
            .saturating_sub(args.context)
            .min(target_start.saturating_sub(1));
        let chunk_ids = &ids[prefix_start..target_end];

        let captures = forward_hidden_all_layers(weights, chunk_ids)?;
        if captures.len() != n_captures {
            return Err(format!("expected {} captures, got {}", n_captures, captures.len()).into());
        }

        let row_start = target_start - prefix_start - 1;
        let row_end = target_end - prefix_start - 1;
        let n_targets = target_end - target_start;

        // Final log-probs at scoring positions, used as the KL reference.
        let final_normed = final_norm(weights, captures.last().unwrap());
        let final_rows = final_normed.slice(s![row_start..row_end, ..]);
        let final_raw = dot_proj(&final_rows, &weights.lm_head);
        let final_log_probs: Vec<Vec<f32>> = (0..n_targets)
            .map(|t| compute_log_probs_row(weights, final_raw.row(t)))
            .collect();

        for (layer_idx, hidden) in captures.iter().enumerate() {
            let normed = final_norm(weights, hidden);
            let rows = normed.slice(s![row_start..row_end, ..]);
            let raw = dot_proj(&rows, &weights.lm_head);
            for offset in 0..n_targets {
                let target = ids[target_start + offset] as usize;
                let layer_lp = compute_log_probs_row(weights, raw.row(offset));
                if target >= layer_lp.len() {
                    return Err(format!("target token {target} out of vocab").into());
                }
                let bits = -(layer_lp[target] as f64) / LN_2;
                let final_lp = &final_log_probs[offset];
                let mut kl_nats = 0.0_f64;
                for v in 0..layer_lp.len() {
                    let lp_l = layer_lp[v] as f64;
                    if !lp_l.is_finite() {
                        continue;
                    }
                    let p_l = lp_l.exp();
                    if p_l <= 0.0 || !p_l.is_finite() {
                        continue;
                    }
                    let lp_f = final_lp[v] as f64;
                    if !lp_f.is_finite() {
                        continue;
                    }
                    kl_nats += p_l * (lp_l - lp_f);
                }
                layer_summaries[layer_idx].total_bits += bits;
                layer_summaries[layer_idx].total_kl_bits += kl_nats / LN_2;
                layer_summaries[layer_idx].n_tokens += 1;
            }
        }

        pb.inc(n_targets as u64);
        target_start = target_end;
    }
    pb.finish_and_clear();

    print_layers_summary(&layer_summaries, text.len(), text.chars().count());
    Ok(())
}

pub(super) fn run_slot(args: SlotArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, 1)?;
    let model = load_model(&args.model)?;
    let full = format!("{}{}", args.prefix, args.answer);
    let prefix_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &args.prefix)?;
    let full_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &full)?;
    ensure_token_prefix(&prefix_ids, &full_ids)?;

    if prefix_ids.len() == full_ids.len() {
        return Err("answer did not add any tokens; check --prefix and --answer".into());
    }

    let range = prefix_ids.len()..full_ids.len();
    let summary = score_token_range(
        model.weights(),
        &full_ids,
        range.clone(),
        args.context,
        range.len().max(1),
        None,
    )?;

    println!("prefix bytes: {}", args.prefix.len());
    println!("answer: {:?}", args.answer);
    println!("answer tokens: {}", range.len());
    println!("bits: {:.3}", summary.total_bits);
    println!("bits/token: {:.3}", summary.bits_per_token());
    println!(
        "bits/char: {:.3}",
        summary.total_bits / args.answer.chars().count().max(1) as f64
    );

    let first_prefix_start = prefix_ids.len().saturating_sub(args.context);
    let prefix_window = &full_ids[first_prefix_start..prefix_ids.len()];
    let logits = logits_for_last_token(model.weights(), prefix_window)?;
    let target = full_ids[prefix_ids.len()];
    let prob = prob_for_target(&logits, target)?;
    let first_bits = -prob.log2();
    let target_text = decode_one(model.tokenizer(), target);
    println!(
        "first token: id={} text={:?} prob={:.6} bits={:.3}",
        target, target_text, prob, first_bits
    );
    print_top_k(model.tokenizer(), &logits, args.top_k);
    Ok(())
}

pub(super) fn run_repeat(args: RepeatArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, 1)?;
    if args.needle.is_empty() {
        return Err("--needle must not be empty".into());
    }
    let text = read_text(&args.text, args.bytes)?;
    let matches: Vec<(usize, &str)> = text.match_indices(&args.needle).collect();
    if matches.is_empty() {
        return Err(format!("needle {:?} not found", args.needle).into());
    }

    let model = load_model(&args.model)?;
    println!(
        "{:<8} {:>10} {:>10} {:>12}  text",
        "occ", "byte", "tokens", "bits"
    );
    println!("{}", "-".repeat(60));
    for (i, (offset, matched)) in matches.iter().enumerate() {
        let prefix = &text[..*offset];
        let full = format!("{prefix}{matched}");
        let prefix_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, prefix)?;
        let full_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &full)?;
        ensure_token_prefix(&prefix_ids, &full_ids)?;
        let range = prefix_ids.len()..full_ids.len();
        let summary = score_token_range(
            model.weights(),
            &full_ids,
            range.clone(),
            args.context,
            range.len().max(1),
            None,
        )?;
        println!(
            "{:<8} {:>10} {:>10} {:>12.3}  {:?}",
            i + 1,
            offset,
            range.len(),
            summary.total_bits,
            matched
        );
    }
    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct LayerSummary {
    pub(super) total_bits: f64,
    pub(super) total_kl_bits: f64,
    pub(super) n_tokens: usize,
}

impl LayerSummary {
    pub(super) fn bits_per_token(&self) -> f64 {
        if self.n_tokens == 0 {
            0.0
        } else {
            self.total_bits / self.n_tokens as f64
        }
    }

    pub(super) fn kl_per_token(&self) -> f64 {
        if self.n_tokens == 0 {
            0.0
        } else {
            self.total_kl_bits / self.n_tokens as f64
        }
    }
}

pub(super) fn layer_label(idx: usize) -> String {
    if idx == 0 {
        "embed".to_string()
    } else {
        format!("L{:02}", idx - 1)
    }
}

pub(super) fn print_layers_summary(layer_summaries: &[LayerSummary], bytes: usize, chars: usize) {
    let n = layer_summaries.len();
    let scored = layer_summaries.first().map(|s| s.n_tokens).unwrap_or(0);
    println!("done.");
    println!("tokens scored:  {:>10}", scored);
    println!("bytes:          {:>10}", bytes);
    println!("chars:          {:>10}", chars);
    println!();
    println!("per-layer bit contribution (final-norm lens):");
    println!();
    println!(
        "  {:<6} {:<6}  {:>11}  {:>11}  {:>11}",
        "from", "to", "bits saved", "bits/token", "KL->final"
    );
    println!("  {:-<55}", "");

    let mut layers_only_total = 0.0_f64;
    for to_idx in 1..n {
        let from = &layer_summaries[to_idx - 1];
        let to = &layer_summaries[to_idx];
        let bits_saved = from.bits_per_token() - to.bits_per_token();
        let kl_reduction = from.kl_per_token() - to.kl_per_token();
        println!(
            "  {:<6} {:<6}  {:>11.3}  {:>11.3}  {:>11.3}",
            layer_label(to_idx - 1),
            layer_label(to_idx),
            bits_saved,
            to.bits_per_token(),
            kl_reduction,
        );
        if to_idx > 1 {
            // Skip the embed -> L0 transition: that's lens warm-up, not layer
            // labour. Match exp 34's `summary_layers_only` view.
            layers_only_total += bits_saved;
        }
    }

    if let (Some(first), Some(last)) = (layer_summaries.first(), layer_summaries.last()) {
        println!();
        println!(
            "embed bits/token: {:>10.3}    final bits/token: {:>10.3}",
            first.bits_per_token(),
            last.bits_per_token()
        );
    }
    println!(
        "total layers-only bits saved: {:>8.2} / token  (excludes embed -> L0)",
        layers_only_total
    );
}

pub(super) fn finite_max(values: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".into())
}

pub(super) fn print_top_k(tokenizer: &tokenizers::Tokenizer, logits: &[f32], top_k: usize) {
    let max_logit = match finite_max(logits) {
        Ok(v) => v,
        Err(_) => return,
    };
    let exp_sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max_logit) as f64).exp())
        .sum();
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("top predictions before slot:");
    for (rank, (id, logit)) in indexed.into_iter().take(top_k).enumerate() {
        let prob = (((logit - max_logit) as f64).exp() / exp_sum).max(0.0);
        println!(
            "  {:>2}. id={:<8} text={:?} prob={:.6} bits={:.3}",
            rank + 1,
            id,
            decode_one(tokenizer, id as u32),
            prob,
            -prob.log2()
        );
    }
}

pub(super) fn decode_one(tokenizer: &tokenizers::Tokenizer, id: u32) -> String {
    tokenizer
        .decode(&[id], true)
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| tokenizer.id_to_token(id))
        .unwrap_or_else(|| format!("[{id}]"))
}
