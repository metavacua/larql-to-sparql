use std::collections::HashMap;

use super::types::PqConfig;

#[derive(Debug, Clone)]
pub(super) struct PqCodebook {
    pub(super) config: PqConfig,
    pub(super) centroids: Vec<Vec<Vec<f64>>>,
    pub(super) stratum_centroids: HashMap<String, HashMap<usize, Vec<Vec<f64>>>>,
}

impl PqCodebook {
    pub(super) fn quantize_indices_for_stratum(&self, coords: &[f64], stratum: &str) -> Vec<usize> {
        let group_dim = self.config.k / self.config.groups;
        (0..self.config.groups)
            .map(|group| {
                let start = group * group_dim;
                let end = start + group_dim;
                nearest_centroid_index(
                    &coords[start..end],
                    self.centroids_for_group(stratum, group),
                )
            })
            .collect()
    }

    pub(super) fn quantize_from_indices_for_stratum(
        &self,
        indices: &[usize],
        stratum: &str,
    ) -> Vec<f64> {
        let group_dim = self.config.k / self.config.groups;
        let mut out = vec![0.0; self.config.k];
        for (group, &index) in indices.iter().take(self.config.groups).enumerate() {
            let start = group * group_dim;
            let end = start + group_dim;
            let centroid = &self.centroids_for_group(stratum, group)[index];
            out[start..end].copy_from_slice(centroid);
        }
        out
    }

    fn centroids_for_group(&self, stratum: &str, group: usize) -> &[Vec<f64>] {
        self.stratum_centroids
            .get(stratum)
            .and_then(|groups| groups.get(&group))
            .unwrap_or(&self.centroids[group])
    }
}

#[derive(Debug, Clone)]
pub(super) struct ModeDTable {
    pub(super) static_delta_by_position: Vec<Vec<f32>>,
    pub(super) static_global_delta: Vec<f32>,
    pub(super) group_tables: Vec<Vec<Vec<f32>>>,
    pub(super) stratum_group_tables: HashMap<String, HashMap<usize, Vec<Vec<f32>>>>,
}

impl ModeDTable {
    pub(super) fn delta_for_position_codes_with_stratum(
        &self,
        position: usize,
        codes: &[usize],
        stratum: &str,
    ) -> Vec<f32> {
        let mut out = self
            .static_delta_by_position
            .get(position)
            .unwrap_or(&self.static_global_delta)
            .clone();
        for (group, &code) in codes.iter().enumerate() {
            let table = &self.table_for_group(stratum, group)[code];
            for (dst, &value) in out.iter_mut().zip(table.iter()) {
                *dst += value;
            }
        }
        out
    }

    fn table_for_group(&self, stratum: &str, group: usize) -> &[Vec<f32>] {
        self.stratum_group_tables
            .get(stratum)
            .and_then(|groups| groups.get(&group))
            .unwrap_or(&self.group_tables[group])
    }
}

pub(super) fn kmeans_centroids(samples: &[Vec<f64>], k: usize, iterations: usize) -> Vec<Vec<f64>> {
    if samples.is_empty() {
        return vec![Vec::new(); k];
    }
    let dim = samples[0].len();
    let mut centroids = (0..k)
        .map(|idx| samples[(idx * samples.len()) / k].clone())
        .collect::<Vec<_>>();
    let mut assignments = vec![0usize; samples.len()];
    for _ in 0..iterations {
        let mut changed = false;
        for (sample_idx, sample) in samples.iter().enumerate() {
            let nearest = nearest_centroid_index(sample, &centroids);
            if assignments[sample_idx] != nearest {
                assignments[sample_idx] = nearest;
                changed = true;
            }
        }
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (sample, &cluster) in samples.iter().zip(assignments.iter()) {
            counts[cluster] += 1;
            for (dst, &value) in sums[cluster].iter_mut().zip(sample.iter()) {
                *dst += value;
            }
        }
        for cluster in 0..k {
            if counts[cluster] == 0 {
                continue;
            }
            let inv = 1.0 / counts[cluster] as f64;
            for value in &mut sums[cluster] {
                *value *= inv;
            }
            centroids[cluster] = sums[cluster].clone();
        }
        if !changed {
            break;
        }
    }
    centroids
}

pub(super) fn nearest_centroid_index(sample: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, centroid) in centroids.iter().enumerate() {
        let dist = sample
            .iter()
            .zip(centroid.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx
}
