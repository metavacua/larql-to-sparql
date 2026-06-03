use std::collections::HashMap;

use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::s;

use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PromptRecord};

#[derive(Debug)]
pub(super) struct WoRoundtripBasis {
    pub(super) head_dim: usize,
    gram: Vec<Vec<f64>>,
    vectors: Vec<Vec<f64>>,
    sigmas: Vec<f64>,
    pub(super) sigma_max: f64,
    pub(super) sigma_min_retained: f64,
    pub(super) sigma_rel_cutoff: f64,
}

impl WoRoundtripBasis {
    pub(super) fn rank_retained(&self) -> usize {
        self.vectors.len()
    }

    pub(super) fn project(&self, y: &[f32]) -> Vec<f32> {
        self.project_with_rank(y, self.vectors.len())
    }

    pub(super) fn project_with_rank(&self, y: &[f32], k: usize) -> Vec<f32> {
        let mut out = vec![0.0f64; self.head_dim];
        for v in self.vectors.iter().take(k.min(self.vectors.len())) {
            let coeff = v
                .iter()
                .zip(y.iter())
                .map(|(&vi, &yi)| vi * yi as f64)
                .sum::<f64>();
            for (dst, &vi) in out.iter_mut().zip(v.iter()) {
                *dst += coeff * vi;
            }
        }
        out.into_iter().map(|value| value as f32).collect()
    }

    pub(super) fn residual_to_z(&self, residual: &[f32]) -> Vec<f64> {
        self.vectors
            .iter()
            .zip(self.sigmas.iter())
            .map(|(v, &sigma)| {
                sigma
                    * v.iter()
                        .zip(residual.iter())
                        .map(|(&vi, &ri)| vi * ri as f64)
                        .sum::<f64>()
            })
            .collect()
    }

    pub(super) fn z_to_residual(&self, z: &[f64]) -> Vec<f32> {
        let mut residual = vec![0.0f64; self.head_dim];
        for ((v, &sigma), &zi) in self.vectors.iter().zip(self.sigmas.iter()).zip(z.iter()) {
            if sigma == 0.0 {
                continue;
            }
            let coeff = zi / sigma;
            for (dst, &vi) in residual.iter_mut().zip(v.iter()) {
                *dst += coeff * vi;
            }
        }
        residual.into_iter().map(|value| value as f32).collect()
    }

    pub(super) fn visible_sq_norm(&self, delta: &[f64]) -> f64 {
        let mut total = 0.0;
        for i in 0..self.head_dim {
            let mut row = 0.0;
            for j in 0..self.head_dim {
                row += self.gram[i][j] * delta[j];
            }
            total += delta[i] * row;
        }
        total.max(0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RoundtripPatchMetrics {
    pub(super) pre_wo_l2: f64,
    pub(super) wo_visible_l2: f64,
}

#[derive(Debug)]
pub(super) struct ZPcaBasis {
    pub(super) vectors: Vec<Vec<f64>>,
}

impl ZPcaBasis {
    pub(super) fn rank(&self) -> usize {
        self.vectors.len()
    }

    pub(super) fn coordinates_with_rank(&self, z: &[f64], k: usize) -> Vec<f64> {
        self.vectors
            .iter()
            .take(k.min(self.vectors.len()))
            .map(|v| v.iter().zip(z.iter()).map(|(&vi, &zi)| vi * zi).sum())
            .collect()
    }

    pub(super) fn reconstruct_from_coordinates(&self, coords: &[f64]) -> Vec<f64> {
        let dim = self.vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut out = vec![0.0; dim];
        for (coord, v) in coords.iter().zip(self.vectors.iter()) {
            for (dst, &vi) in out.iter_mut().zip(v.iter()) {
                *dst += coord * vi;
            }
        }
        out
    }

    pub(super) fn project_with_rank(&self, z: &[f64], k: usize) -> Vec<f64> {
        let coords = self.coordinates_with_rank(z, k);
        self.reconstruct_from_coordinates(&coords)
    }
}

pub(super) fn build_roundtrip_bases(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
    sigma_rel_cutoff: f64,
) -> Result<HashMap<HeadId, WoRoundtripBasis>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut bases = HashMap::new();
    for (layer, layer_heads) in heads_by_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let w_o = weights
            .tensors
            .get(&weights.arch.attn_o_key(layer))
            .ok_or_else(|| format!("missing W_O tensor at layer {layer}"))?;
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for head in layer_heads {
            let start = head.head * head_dim;
            let end = start + head_dim;
            let w_o_head = w_o.slice(s![.., start..end]);
            let basis = build_wo_roundtrip_basis(&w_o_head, sigma_rel_cutoff)?;
            bases.insert(head, basis);
        }
        remove_layer_tensors(weights, inserted);
    }

    Ok(bases)
}

#[derive(Debug)]
struct ZPcaAccumulator {
    count: u64,
    sum: Vec<f64>,
    sum_outer: Vec<Vec<f64>>,
}

impl ZPcaAccumulator {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; dim],
            sum_outer: vec![vec![0.0; dim]; dim],
        }
    }

    fn add(&mut self, z: &[f64]) {
        self.count += 1;
        for (dst, &value) in self.sum.iter_mut().zip(z.iter()) {
            *dst += value;
        }
        for i in 0..z.len() {
            for j in i..z.len() {
                self.sum_outer[i][j] += z[i] * z[j];
            }
        }
    }

    fn finish(mut self) -> ZPcaBasis {
        let dim = self.sum.len();
        if self.count == 0 {
            return ZPcaBasis {
                vectors: Vec::new(),
            };
        }
        for i in 0..dim {
            for j in 0..i {
                self.sum_outer[i][j] = self.sum_outer[j][i];
            }
        }
        let n = self.count as f64;
        let mut covariance = self.sum_outer;
        for i in 0..dim {
            for j in 0..dim {
                covariance[i][j] = covariance[i][j] / n - (self.sum[i] / n) * (self.sum[j] / n);
            }
        }
        let (eigenvalues, eigenvectors) = jacobi_symmetric_eigen(&covariance, 100, 1e-8);
        let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues.into_iter().zip(eigenvectors).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ZPcaBasis {
            vectors: pairs
                .into_iter()
                .filter(|(value, _)| *value > 0.0)
                .map(|(_, vector)| vector)
                .collect(),
        }
    }
}

pub(super) fn fit_z_pca_bases(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
) -> Result<HashMap<HeadId, ZPcaBasis>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut accumulators: HashMap<HeadId, ZPcaAccumulator> = HashMap::new();
    for head in heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing W_O basis for L{} H{}", head.layer, head.head))?;
        accumulators.insert(*head, ZPcaAccumulator::new(basis.rank_retained()));
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  pca-fit [{}/{}] {}", prompt_idx + 1, prompts.len(), label);
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).expect("basis pre-created for PCA fit");
                    let head_means = means.get(head).expect("means pre-created for PCA fit");
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let acc = accumulators.get_mut(head).expect("PCA accumulator missing");
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during PCA fit")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        acc.add(&z);
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    Ok(accumulators
        .into_iter()
        .map(|(head, acc)| (head, acc.finish()))
        .collect())
}

fn build_wo_roundtrip_basis(
    w_o_head: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    sigma_rel_cutoff: f64,
) -> Result<WoRoundtripBasis, Box<dyn std::error::Error>> {
    let hidden = w_o_head.nrows();
    let head_dim = w_o_head.ncols();
    let mut gram = vec![vec![0.0f64; head_dim]; head_dim];
    for row in 0..hidden {
        for i in 0..head_dim {
            let wi = w_o_head[[row, i]] as f64;
            for j in i..head_dim {
                gram[i][j] += wi * w_o_head[[row, j]] as f64;
            }
        }
    }
    for i in 0..head_dim {
        for j in 0..i {
            gram[i][j] = gram[j][i];
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_symmetric_eigen(&gram, 100, 1e-10);
    let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues.into_iter().zip(eigenvectors).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let sigma_max = pairs
        .first()
        .map(|(value, _)| value.max(0.0).sqrt())
        .unwrap_or(0.0);
    let cutoff = sigma_max * sigma_rel_cutoff;
    let mut vectors = Vec::new();
    let mut sigmas = Vec::new();
    let mut sigma_min_retained: f64 = 0.0;
    for (value, vector) in pairs {
        let sigma = value.max(0.0).sqrt();
        if sigma > cutoff {
            sigma_min_retained = if sigma_min_retained == 0.0 {
                sigma
            } else {
                sigma_min_retained.min(sigma)
            };
            sigmas.push(sigma);
            vectors.push(vector);
        }
    }
    if vectors.is_empty() && sigma_max > 0.0 {
        return Err("W_O roundtrip retained zero singular directions".into());
    }

    Ok(WoRoundtripBasis {
        head_dim,
        gram,
        vectors,
        sigmas,
        sigma_max,
        sigma_min_retained,
        sigma_rel_cutoff,
    })
}

pub(super) fn jacobi_symmetric_eigen(
    input: &[Vec<f64>],
    max_sweeps: usize,
    tolerance: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = input.len();
    let mut a = input.to_vec();
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..max_sweeps {
        let mut max_value = 0.0;
        let mut p = 0;
        let mut q = 1.min(n.saturating_sub(1));
        for i in 0..n {
            for j in (i + 1)..n {
                let value = a[i][j].abs();
                if value > max_value {
                    max_value = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max_value < tolerance || n < 2 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        if apq == 0.0 {
            continue;
        }
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                let new_kp = c * akp - s * akq;
                let new_kq = s * akp + c * akq;
                a[k][p] = new_kp;
                a[p][k] = new_kp;
                a[k][q] = new_kq;
                a[q][k] = new_kq;
            }
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for row in &mut v {
            let vip = row[p];
            let viq = row[q];
            row[p] = c * vip - s * viq;
            row[q] = s * vip + c * viq;
        }
    }

    let eigenvalues = (0..n).map(|i| a[i][i]).collect::<Vec<_>>();
    let eigenvectors = (0..n)
        .map(|col| (0..n).map(|row| v[row][col]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    (eigenvalues, eigenvectors)
}
