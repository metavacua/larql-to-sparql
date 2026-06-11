/// Compute a boolean on-shell mask for features in one layer.
/// Features whose c_score ranks in the top `top_fraction` are on-shell.
/// `top_fraction = 0.15` selects the top 15% factual features.
///
/// Returns a Vec<bool> of length equal to `c_scores`.
/// Empty input returns an empty Vec.
pub fn compute_onshell_mask(c_scores: &[f32], top_fraction: f32) -> Vec<bool> {
    if c_scores.is_empty() {
        return Vec::new();
    }
    let n = c_scores.len();
    // At least 1 feature is always on-shell
    let k = ((n as f32 * top_fraction).ceil() as usize).max(1).min(n);
    // k-th largest threshold (descending sort)
    let mut sorted: Vec<f32> = c_scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    // Ties are included — may yield slightly more than k on-shell features
    c_scores.iter().map(|&s| s >= threshold).collect()
}

/// Count on-shell features across all layers; returns (on_shell_count, total) per layer.
pub fn onshell_stats(c_scores_per_layer: &[Vec<f32>], top_fraction: f32) -> Vec<(usize, usize)> {
    c_scores_per_layer
        .iter()
        .map(|cscores| {
            let mask = compute_onshell_mask(cscores, top_fraction);
            let on_shell = mask.iter().filter(|&&b| b).count();
            (on_shell, cscores.len())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_15pct_of_10_is_2() {
        // 15% of 10 = 1.5 → ceil = 2
        let scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mask = compute_onshell_mask(&scores, 0.15);
        let on_shell_indices: Vec<usize> =
            mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
        assert_eq!(on_shell_indices.len(), 2);
        assert!(on_shell_indices.contains(&8));
        assert!(on_shell_indices.contains(&9));
    }

    #[test]
    fn empty_scores_returns_empty_mask() {
        assert!(compute_onshell_mask(&[], 0.15).is_empty());
    }

    #[test]
    fn single_feature_always_on_shell() {
        let mask = compute_onshell_mask(&[0.5], 0.15);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn mask_length_matches_input() {
        let scores: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let mask = compute_onshell_mask(&scores, 0.15);
        assert_eq!(mask.len(), 100);
    }

    #[test]
    fn on_shell_count_is_at_least_top_fraction() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mask = compute_onshell_mask(&scores, 0.15);
        let count = mask.iter().filter(|&&b| b).count();
        // 15% of 20 = 3
        assert_eq!(count, 3);
    }

    #[test]
    fn all_same_scores_all_on_shell() {
        let scores = vec![1.0f32; 10];
        let mask = compute_onshell_mask(&scores, 0.15);
        assert!(mask.iter().all(|&b| b));
    }

    #[test]
    fn onshell_stats_counts_per_layer() {
        let layers = vec![
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![],
        ];
        let stats = onshell_stats(&layers, 0.15);
        assert_eq!(stats.len(), 2);
        let (on, total) = stats[0];
        assert_eq!(total, 10);
        assert_eq!(on, 2);
        let (on2, total2) = stats[1];
        assert_eq!(total2, 0);
        assert_eq!(on2, 0);
    }
}
