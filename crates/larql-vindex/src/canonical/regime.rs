use crate::canonical::types::Regime;

const ACTIVE_FLOOR: f32 = 0.1;

/// Classify the regime for a single layer.
/// density = fraction of features with c_score > ACTIVE_FLOOR.
///   - density > 0.5  → Wave
///   - density < 0.05 → Particle
///   - otherwise      → Wavelet
/// Empty layer → (Wavelet, 0.0).
pub fn classify_layer_regime(c_scores: &[f32]) -> (Regime, f32) {
    if c_scores.is_empty() {
        return (Regime::Wavelet, 0.0);
    }
    let active = c_scores.iter().filter(|&&s| s > ACTIVE_FLOOR).count();
    let density = active as f32 / c_scores.len() as f32;
    let regime = if density > 0.5 {
        Regime::Wave
    } else if density < 0.05 {
        Regime::Particle
    } else {
        Regime::Wavelet
    };
    (regime, density)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layer_is_wave() {
        // 80% active
        let scores: Vec<f32> = (0..10).map(|i| if i < 8 { 0.5 } else { 0.0 }).collect();
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wave);
        assert!((density - 0.8).abs() < 1e-5);
    }

    #[test]
    fn sparse_layer_is_particle() {
        // 2% active
        let mut scores = vec![0.0f32; 100];
        scores[0] = 0.5;
        scores[1] = 0.5;
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Particle);
        assert!((density - 0.02).abs() < 1e-5);
    }

    #[test]
    fn mid_density_is_wavelet() {
        // 20% active
        let mut scores = vec![0.0f32; 10];
        for i in 0..2 { scores[i] = 0.5; }
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wavelet);
        assert!((density - 0.2).abs() < 1e-5);
    }

    #[test]
    fn empty_layer_is_wavelet_density_zero() {
        let (regime, density) = classify_layer_regime(&[]);
        assert_eq!(regime, Regime::Wavelet);
        assert_eq!(density, 0.0);
    }

    #[test]
    fn boundary_at_0_5_is_wave() {
        // density == 0.5, NOT > 0.5, so Wavelet
        let scores: Vec<f32> = (0..10).map(|i| if i < 5 { 0.5 } else { 0.0 }).collect();
        let (regime, _) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wavelet, "density=0.5 is not > 0.5, should be Wavelet");
    }

    #[test]
    fn boundary_at_0_05_is_wavelet() {
        // density = 0.05, NOT < 0.05, so Wavelet
        let mut scores = vec![0.0f32; 20];
        scores[0] = 0.5; // 1/20 = 0.05
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wavelet, "density=0.05 is not < 0.05, should be Wavelet");
        assert!((density - 0.05).abs() < 1e-5);
    }
}
