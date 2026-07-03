use std::collections::{HashMap, HashSet};

use crate::index::VectorIndex;
use ndarray::Array1;

/// Signed top-k `(layer, feature)` a subject routes to, across all provided layers,
/// reusing the engine's signed `gate_knn` (the single canonical ranking).
pub fn routed_features(
    index: &VectorIndex,
    residual_by_layer: &HashMap<usize, Array1<f32>>,
    per_layer_k: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (&layer, resid) in residual_by_layer {
        for (feat, _score) in index.gate_knn(layer, resid, per_layer_k) {
            out.push((layer, feat));
        }
    }
    out
}

/// Label features for one relation using frame-subtraction.
/// `routed`: per subject, the (layer,feat) it routes to (signed top-k across all layers — computed by the caller).
/// `down_ids`: (layer,feat) -> the token-id set of its down_meta top-K tokens.
/// `obj_ids`: object string -> its token-id set (the caller tokenizes the object,
/// including the leading-space BPE variant). `pairs`: (subject, object).
///
/// A feature is labeled `rel` if it is subject-specific (not in the relation
/// frame) AND its down-meta top-K token-id set intersects that subject's object
/// token-id set. This is the validated Phase-0 mechanism: a single down-meta
/// top-token STRING can never equal a multi-token object, so matching is done at
/// the token-id-set level over the feature's top-K down tokens.
/// `frame_frac`: a feature routed by > frame_frac * n subjects is frame and excluded.
pub fn label_relation_from_routed(
    routed: &[(String, Vec<(usize, usize)>)],
    down_ids: &HashMap<(usize, usize), HashSet<u32>>,
    obj_ids: &HashMap<String, HashSet<u32>>,
    pairs: &[(String, String)],
    rel: &str,
    frame_frac: f32,
) -> Vec<((usize, usize), String)> {
    let n = routed.len().max(1);
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for (_, fs) in routed {
        for &f in fs {
            *count.entry(f).or_default() += 1;
        }
    }
    let frame: HashSet<(usize, usize)> = count
        .iter()
        .filter(|(_, &c)| c as f32 > frame_frac * n as f32)
        .map(|(&f, _)| f)
        .collect();
    let obj: HashMap<&str, &str> = pairs.iter().map(|(s, o)| (s.as_str(), o.as_str())).collect();
    let mut out = Vec::new();
    for (subj, fs) in routed {
        let Some(&o) = obj.get(subj.as_str()) else { continue };
        for f in fs.iter().filter(|f| !frame.contains(f)) {
            if let (Some(fids), Some(oids)) = (down_ids.get(f), obj_ids.get(o)) {
                if !fids.is_disjoint(oids) {
                    out.push((*f, rel.to_string()));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_index() -> crate::index::VectorIndex {
        let num_layers = 2;
        let hidden = 4;
        let mut gate0 = ndarray::Array2::<f32>::zeros((8, hidden));
        for f in 0..8 {
            gate0[[f, f % 4]] = 1.0;
        }
        let gate1 = gate0.clone();
        let gate = vec![Some(gate0), Some(gate1)];
        let down = vec![None, None];
        crate::index::VectorIndex::new(gate, down, num_layers, hidden)
    }

    #[test]
    fn routed_features_uses_signed_gate_knn_across_layers() {
        let index = synth_index();
        let q = ndarray::Array1::from_vec(vec![0.0f32, 0.0, 1.0, 0.0]); // e_2
        let resid: std::collections::HashMap<usize, ndarray::Array1<f32>> =
            [(0usize, q.clone()), (1usize, q.clone())].into_iter().collect();
        let routed = routed_features(&index, &resid, 2);
        // e_2 dot-products to 1.0 with features 2 and 6 (f % 4 == 2) in each layer; signed top-2 returns them.
        assert!(routed.contains(&(0, 2)) && routed.contains(&(0, 6)), "layer 0 routes to 2 and 6: {:?}", routed);
        assert!(routed.contains(&(1, 2)) && routed.contains(&(1, 6)), "layer 1 routes to 2 and 6: {:?}", routed);
    }

    fn ids(xs: &[u32]) -> HashSet<u32> {
        xs.iter().copied().collect()
    }

    #[test]
    fn labels_subject_specific_feature_after_frame_subtraction() {
        // Two subjects. Feature (5,0) fires for BOTH (frame). (5,1) fires only for FR
        // and its down-meta top-K token-id set intersects FR's object "Paris" id set
        // → must be labeled; (5,0) must not. Feature (5,3) fires only for FR but its
        // down ids are DISJOINT from "Paris" → must NOT be labeled.
        let routed = vec![
            ("FR".to_string(), vec![(5usize, 0usize), (5, 1), (5, 3)]),
            ("JP".to_string(), vec![(5usize, 0usize), (5, 2)]),
        ];
        // Down token-id sets per feature (e.g. (5,1)'s top-K contains id 101).
        let down_ids: std::collections::HashMap<(usize, usize), HashSet<u32>> = [
            ((5, 0), ids(&[1])),     // "the"
            ((5, 1), ids(&[101])),   // Paris
            ((5, 2), ids(&[202])),   // Tokyo
            ((5, 3), ids(&[999])),   // disjoint from any object
        ]
        .into_iter()
        .collect();
        // Object string → token-id set (as the caller's tokenizer would supply).
        let obj_ids: std::collections::HashMap<String, HashSet<u32>> = [
            ("Paris".to_string(), ids(&[101])),
            ("Tokyo".to_string(), ids(&[202])),
        ]
        .into_iter()
        .collect();
        let pairs = vec![("FR".to_string(), "Paris".to_string()), ("JP".to_string(), "Tokyo".to_string())];
        let labels = label_relation_from_routed(&routed, &down_ids, &obj_ids, &pairs, "capital", 0.5);
        assert!(labels.contains(&((5, 1), "capital".to_string())), "FR's specific feature (Paris) labeled");
        assert!(labels.contains(&((5, 2), "capital".to_string())), "JP's specific feature (Tokyo) labeled");
        assert!(!labels.iter().any(|(lf, _)| *lf == (5, 0)), "frame feature (5,0) NOT labeled");
        assert!(!labels.iter().any(|(lf, _)| *lf == (5, 3)), "disjoint-id feature (5,3) NOT labeled");
    }
}
