pub(super) fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max_logit = logits
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = logits
        .iter()
        .map(|&v| ((v as f64) - max_logit).exp())
        .sum::<f64>();
    let log_z = max_logit + sum_exp.ln();
    logits.iter().map(|&v| (v as f64) - log_z).collect()
}

pub(super) fn kl_logp(p_logp: &[f64], q_logp: &[f64]) -> f64 {
    p_logp
        .iter()
        .zip(q_logp.iter())
        .map(|(&lp, &lq)| {
            let p = lp.exp();
            p * (lp - lq)
        })
        .sum()
}

pub(super) fn token_prob(logp: &[f64], token_id: u32) -> f64 {
    logp.get(token_id as usize)
        .map(|value| value.exp())
        .unwrap_or(0.0)
}

pub(super) fn argmax_usize(values: &[usize]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by_key(|(_, value)| *value)
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

pub(super) fn code_mass(counts: &[usize], code: usize) -> f64 {
    let total = counts.iter().sum::<usize>();
    if total == 0 {
        0.0
    } else {
        counts.get(code).copied().unwrap_or(0) as f64 / total as f64
    }
}

pub(super) fn entropy_bits(counts: &[usize]) -> f64 {
    let total = counts.iter().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn kl_counts_to_probs_bits(counts: &[usize], probs: &[f64]) -> f64 {
    let total = counts.iter().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    counts
        .iter()
        .zip(probs.iter())
        .filter(|(&count, _)| count > 0)
        .map(|(&count, &q)| {
            let p = count as f64 / total as f64;
            p * (p / q.max(1e-12)).log2()
        })
        .sum()
}

pub(super) fn js_divergence_bits(a: &[usize], b: &[usize]) -> f64 {
    let total_a = a.iter().sum::<usize>();
    let total_b = b.iter().sum::<usize>();
    if total_a == 0 || total_b == 0 {
        return 0.0;
    }
    let levels = a.len().max(b.len());
    let mut midpoint = vec![0.0; levels];
    for (idx, value) in midpoint.iter_mut().enumerate() {
        let pa = a.get(idx).copied().unwrap_or(0) as f64 / total_a as f64;
        let pb = b.get(idx).copied().unwrap_or(0) as f64 / total_b as f64;
        *value = 0.5 * (pa + pb);
    }
    0.5 * kl_counts_to_probs_bits(a, &midpoint) + 0.5 * kl_counts_to_probs_bits(b, &midpoint)
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as f64) - (y as f64)).abs())
        .fold(0.0, f64::max)
}

pub(super) fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

pub(super) fn top_k_indices(values: &[f32], k: usize) -> Vec<u32> {
    let mut pairs: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    let take = k.min(pairs.len());
    pairs.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(take);
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().map(|(idx, _)| idx as u32).collect()
}

pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub(super) fn bool_rate(values: impl Iterator<Item = bool>) -> f64 {
    let mut total = 0usize;
    let mut hits = 0usize;
    for value in values {
        total += 1;
        if value {
            hits += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

pub(super) fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((values.len() - 1) as f64 * p).ceil() as usize;
    values[rank.min(values.len() - 1)]
}
