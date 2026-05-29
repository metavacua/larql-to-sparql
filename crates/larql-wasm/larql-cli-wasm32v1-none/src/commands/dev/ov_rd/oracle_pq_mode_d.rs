use std::collections::HashMap;

use larql_vindex::VectorIndex;
use ndarray::s;

use super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::pq::{ModeDTable, PqCodebook};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PqConfig};

pub(super) fn corruption_keep_values(groups: usize) -> Vec<usize> {
    [0usize, 4, 8, 12, 16, 24, 32, 40, groups]
        .into_iter()
        .filter(|value| *value <= groups)
        .collect()
}

pub(super) fn materialize_mode_d_tables(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    stratum_conditioned_groups: &[usize],
) -> Result<HashMap<(HeadId, PqConfig), ModeDTable>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut tables = HashMap::new();
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
            let head_means = means
                .get(&head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let static_global_delta = project_head_vector_to_hidden(&w_o_head, &head_means.global);
            let static_delta_by_position = head_means
                .positions
                .iter()
                .map(|mean| project_head_vector_to_hidden(&w_o_head, mean))
                .collect::<Vec<_>>();
            let basis = bases
                .get(&head)
                .ok_or_else(|| format!("missing W_O basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(&head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;

            for ((codebook_head, config), codebook) in codebooks.iter() {
                if *codebook_head != head {
                    continue;
                }
                let group_dim = config.k / config.groups;
                let mut group_tables = Vec::with_capacity(config.groups);
                for group in 0..config.groups {
                    let mut table = Vec::with_capacity(codebook.centroids[group].len());
                    for centroid in &codebook.centroids[group] {
                        let mut coords = vec![0.0; config.k];
                        let start_coord = group * group_dim;
                        coords[start_coord..start_coord + group_dim].copy_from_slice(centroid);
                        let z_part = pca_basis.reconstruct_from_coordinates(&coords);
                        let residual_part = basis.z_to_residual(&z_part);
                        table.push(project_head_vector_to_hidden(&w_o_head, &residual_part));
                    }
                    group_tables.push(table);
                }
                let mut stratum_group_tables: HashMap<String, HashMap<usize, Vec<Vec<f32>>>> =
                    HashMap::new();
                for (stratum, groups) in &codebook.stratum_centroids {
                    for &group in stratum_conditioned_groups {
                        let Some(centroids) = groups.get(&group) else {
                            continue;
                        };
                        let mut table = Vec::with_capacity(centroids.len());
                        for centroid in centroids {
                            let mut coords = vec![0.0; config.k];
                            let start_coord = group * group_dim;
                            coords[start_coord..start_coord + group_dim].copy_from_slice(centroid);
                            let z_part = pca_basis.reconstruct_from_coordinates(&coords);
                            let residual_part = basis.z_to_residual(&z_part);
                            table.push(project_head_vector_to_hidden(&w_o_head, &residual_part));
                        }
                        stratum_group_tables
                            .entry(stratum.clone())
                            .or_default()
                            .insert(group, table);
                    }
                }
                tables.insert(
                    (head, *config),
                    ModeDTable {
                        static_delta_by_position: static_delta_by_position.clone(),
                        static_global_delta: static_global_delta.clone(),
                        group_tables,
                        stratum_group_tables,
                    },
                );
            }
        }
        remove_layer_tensors(weights, inserted);
    }
    Ok(tables)
}

fn project_head_vector_to_hidden(
    w_o_head: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    values: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; w_o_head.nrows()];
    for row in 0..w_o_head.nrows() {
        let mut sum = 0.0f32;
        for col in 0..w_o_head.ncols() {
            sum += values[col] * w_o_head[[row, col]];
        }
        out[row] = sum;
    }
    out
}
