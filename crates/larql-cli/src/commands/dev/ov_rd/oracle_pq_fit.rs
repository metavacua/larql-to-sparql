use std::collections::HashMap;

use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::s;

use super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::pq::{kmeans_centroids, PqCodebook};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PqConfig, PromptRecord};

pub(super) fn fit_pq_codebooks(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    configs: &[PqConfig],
    iterations: usize,
    stratum_conditioned_groups: &[usize],
) -> Result<HashMap<(HeadId, PqConfig), PqCodebook>, Box<dyn std::error::Error>> {
    let max_k = configs.iter().map(|c| c.k).max().unwrap_or(0);
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut samples: HashMap<HeadId, Vec<Vec<f64>>> = HashMap::new();
    let mut samples_by_stratum: HashMap<(HeadId, String), Vec<Vec<f64>>> = HashMap::new();
    for head in heads {
        samples.insert(*head, Vec::new());
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  pq-fit [{}/{}] {}", prompt_idx + 1, prompts.len(), label);
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).expect("basis pre-created for PQ fit");
                    let head_means = means.get(head).expect("means pre-created for PQ fit");
                    let pca_basis = pca_bases.get(head).expect("PCA pre-created for PQ fit");
                    if pca_basis.rank() < max_k {
                        return Err(format!(
                            "PCA rank {} is below requested K {} for L{}H{}",
                            pca_basis.rank(),
                            max_k,
                            head.layer,
                            head.head
                        )
                        .into());
                    }
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_samples = samples.get_mut(head).expect("PQ samples missing");
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during PQ fit")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        let coords = pca_basis.coordinates_with_rank(&z, max_k);
                        head_samples.push(coords.clone());
                        if !stratum_conditioned_groups.is_empty() {
                            samples_by_stratum
                                .entry((*head, stratum.to_string()))
                                .or_default()
                                .push(coords);
                        }
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

    let mut codebooks = HashMap::new();
    for head in heads {
        let head_samples = samples
            .get(head)
            .ok_or_else(|| format!("missing PQ samples for L{}H{}", head.layer, head.head))?;
        for &config in configs {
            let levels = 1usize << config.bits_per_group;
            let group_dim = config.k / config.groups;
            let mut centroids = Vec::with_capacity(config.groups);
            for group in 0..config.groups {
                let start = group * group_dim;
                let group_samples = head_samples
                    .iter()
                    .map(|sample| sample[start..start + group_dim].to_vec())
                    .collect::<Vec<_>>();
                centroids.push(kmeans_centroids(&group_samples, levels, iterations));
            }
            let mut stratum_centroids: HashMap<String, HashMap<usize, Vec<Vec<f64>>>> =
                HashMap::new();
            for &group in stratum_conditioned_groups {
                let start = group * group_dim;
                for ((sample_head, stratum), stratum_samples) in samples_by_stratum.iter() {
                    if sample_head != head {
                        continue;
                    }
                    let group_samples = stratum_samples
                        .iter()
                        .map(|sample| sample[start..start + group_dim].to_vec())
                        .collect::<Vec<_>>();
                    stratum_centroids
                        .entry(stratum.clone())
                        .or_default()
                        .insert(group, kmeans_centroids(&group_samples, levels, iterations));
                }
            }
            codebooks.insert(
                (*head, config),
                PqCodebook {
                    config,
                    centroids,
                    stratum_centroids,
                },
            );
        }
    }

    Ok(codebooks)
}
