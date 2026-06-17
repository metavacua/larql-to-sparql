use std::collections::{HashMap, HashSet};

/// Label features for one relation using frame-subtraction.
/// `routed`: per subject, the (layer,feat) it routes to (signed top-k across all layers — computed by the caller).
/// `down`: (layer,feat) -> its down_meta top token. `pairs`: (subject, object).
/// A feature is labeled `rel` if it is subject-specific (not in the relation frame) AND its
/// down_meta top token equals that subject's object. `frame_frac`: a feature routed by
/// > frame_frac * n subjects is considered frame and excluded.
pub fn label_relation_from_routed(
    routed: &[(String, Vec<(usize, usize)>)],
    down: &HashMap<(usize, usize), String>,
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
            if down.get(f).map(|t| t.trim().eq_ignore_ascii_case(o)).unwrap_or(false) {
                out.push((*f, rel.to_string()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_subject_specific_feature_after_frame_subtraction() {
        // Two subjects. Feature (5,0) fires for BOTH (frame). (5,1) fires only for FR
        // and its down_meta top token == FR's object "Paris" → must be labeled; (5,0) must not.
        let routed = vec![
            ("FR".to_string(), vec![(5usize, 0usize), (5, 1)]),
            ("JP".to_string(), vec![(5usize, 0usize), (5, 2)]),
        ];
        let down: std::collections::HashMap<(usize, usize), String> =
            [((5, 0), "the".to_string()), ((5, 1), "Paris".to_string()), ((5, 2), "Tokyo".to_string())]
                .into_iter().collect();
        let pairs = vec![("FR".to_string(), "Paris".to_string()), ("JP".to_string(), "Tokyo".to_string())];
        let labels = label_relation_from_routed(&routed, &down, &pairs, "capital", 0.5);
        assert!(labels.contains(&((5, 1), "capital".to_string())), "FR's specific feature (Paris) labeled");
        assert!(labels.contains(&((5, 2), "capital".to_string())), "JP's specific feature (Tokyo) labeled");
        assert!(!labels.iter().any(|(lf, _)| *lf == (5, 0)), "frame feature (5,0) NOT labeled");
    }
}
