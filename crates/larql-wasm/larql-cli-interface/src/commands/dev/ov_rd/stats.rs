use std::collections::HashMap;

use super::reports::FinishedHeadStats;

#[derive(Debug)]
pub(super) struct RunningHeadStats {
    count: u64,
    sum: Vec<f64>,
    sum_sq_norm: f64,
}

impl RunningHeadStats {
    pub(super) fn new(head_dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; head_dim],
            sum_sq_norm: 0.0,
        }
    }

    pub(super) fn add(&mut self, values: &[f32]) {
        self.count += 1;
        let mut sq = 0.0f64;
        for (dst, &v) in self.sum.iter_mut().zip(values.iter()) {
            let vf = v as f64;
            *dst += vf;
            sq += vf * vf;
        }
        self.sum_sq_norm += sq;
    }

    pub(super) fn finish(&self) -> FinishedHeadStats {
        if self.count == 0 {
            return FinishedHeadStats {
                count: 0,
                mean_norm_sq: 0.0,
                second_moment: 0.0,
                variance: 0.0,
                rms_norm: 0.0,
            };
        }
        let n = self.count as f64;
        let mean_norm_sq = self
            .sum
            .iter()
            .map(|v| {
                let m = *v / n;
                m * m
            })
            .sum::<f64>();
        let second_moment = self.sum_sq_norm / n;
        let variance = (second_moment - mean_norm_sq).max(0.0);
        FinishedHeadStats {
            count: self.count,
            mean_norm_sq,
            second_moment,
            variance,
            rms_norm: second_moment.sqrt(),
        }
    }
}

#[derive(Debug, Clone)]
struct MeanAccumulator {
    count: u64,
    sum: Vec<f64>,
}

impl MeanAccumulator {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; dim],
        }
    }

    fn add(&mut self, values: &[f32]) {
        self.count += 1;
        for (dst, &value) in self.sum.iter_mut().zip(values.iter()) {
            *dst += value as f64;
        }
    }

    fn mean(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.sum.len()];
        }
        let n = self.count as f64;
        self.sum.iter().map(|v| (*v / n) as f32).collect()
    }
}

#[derive(Debug)]
pub(super) struct StaticHeadAccumulator {
    global: MeanAccumulator,
    positions: Vec<MeanAccumulator>,
    strata: HashMap<String, MeanAccumulator>,
    position_strata: HashMap<String, Vec<MeanAccumulator>>,
}

impl StaticHeadAccumulator {
    pub(super) fn new(head_dim: usize) -> Self {
        Self {
            global: MeanAccumulator::new(head_dim),
            positions: Vec::new(),
            strata: HashMap::new(),
            position_strata: HashMap::new(),
        }
    }

    pub(super) fn add(&mut self, position: usize, stratum: &str, values: &[f32]) {
        self.global.add(values);
        while self.positions.len() <= position {
            self.positions
                .push(MeanAccumulator::new(self.global.sum.len()));
        }
        self.positions[position].add(values);
        self.strata
            .entry(stratum.to_string())
            .or_insert_with(|| MeanAccumulator::new(self.global.sum.len()))
            .add(values);
        let by_position = self.position_strata.entry(stratum.to_string()).or_default();
        while by_position.len() <= position {
            by_position.push(MeanAccumulator::new(self.global.sum.len()));
        }
        by_position[position].add(values);
    }

    pub(super) fn finish(&self) -> StaticHeadMeans {
        StaticHeadMeans {
            count: self.global.count,
            head_dim: self.global.sum.len(),
            global: self.global.mean(),
            positions: self.positions.iter().map(MeanAccumulator::mean).collect(),
            strata: self
                .strata
                .iter()
                .map(|(key, value)| (key.clone(), value.mean()))
                .collect(),
            position_strata: self
                .position_strata
                .iter()
                .map(|(key, values)| {
                    (
                        key.clone(),
                        values.iter().map(MeanAccumulator::mean).collect(),
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct StaticHeadMeans {
    pub(super) count: u64,
    pub(super) head_dim: usize,
    pub(super) global: Vec<f32>,
    pub(super) positions: Vec<Vec<f32>>,
    pub(super) strata: HashMap<String, Vec<f32>>,
    pub(super) position_strata: HashMap<String, Vec<Vec<f32>>>,
}
