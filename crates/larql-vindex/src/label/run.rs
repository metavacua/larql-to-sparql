//! Producer orchestration for `larql label`: load captured residuals and
//! drive the (already-tested) routing + frame-subtraction + match pieces to
//! emit `((layer, feature), relation)` labels.
//!
//! Two pure, model-free entry points live here so the producer's logic is
//! testable without loading a model:
//!
//! - [`load_subject_residuals`] — read a `residuals.vectors.jsonl` into a
//!   `subject -> layer -> vector` map.
//! - [`label_catalog`] — compose [`crate::label::contrastive`] over a whole
//!   [`Catalog`] of relations.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array1;
use serde_json::Value;

use crate::index::VectorIndex;
use crate::label::catalog::Catalog;
use crate::label::contrastive::{label_relation_from_routed, routed_features};

/// Read a `residuals.vectors.jsonl` (as written by the inference capturer)
/// into `subject -> (layer -> residual vector)`.
///
/// Each data line carries `id` ("<subject>_L<layer>"), `layer`, and `vector`.
/// The subject is `id` with its trailing `_L<layer>` stripped (using the
/// `layer` field, so subjects that themselves contain `_L` survive).
///
/// Only the `_header` line is skipped (detected by its `_header` field, which
/// also supplies the expected `dimension`). Every other non-empty line is a
/// data record that must fully parse: a line that is not valid JSON, lacks
/// `id`/`layer`/`vector`, has a non-numeric vector element, or whose vector
/// length disagrees with the expected dimension is a malformed/truncated
/// record and yields an `io::Error` (InvalidData) naming the offending id.
/// This refuses to let a silently-shortened vector escape into routing, where
/// the gate matmul would panic on a dimension mismatch. If no `_header`
/// dimension is present, the first valid record's length sets the expectation.
pub fn load_subject_residuals(
    path: &Path,
) -> std::io::Result<HashMap<String, HashMap<usize, Array1<f32>>>> {
    use std::io::{Error, ErrorKind};

    let text = std::fs::read_to_string(path)?;
    let mut out: HashMap<String, HashMap<usize, Array1<f32>>> = HashMap::new();
    let mut expected_dim: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| {
            Error::new(ErrorKind::InvalidData, format!("malformed residual line: {e}"))
        })?;
        // The header is the only intentionally skipped line; capture its
        // declared dimension to validate every data record's length.
        if v.get("_header").is_some() {
            if let Some(dim) = v.get("dimension").and_then(Value::as_u64) {
                expected_dim = Some(dim as usize);
            }
            continue;
        }
        let (Some(id), Some(layer), Some(vec_json)) = (
            v.get("id").and_then(Value::as_str),
            v.get("layer").and_then(Value::as_u64),
            v.get("vector"),
        ) else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("residual record missing id/layer/vector: {line}"),
            ));
        };
        let Some(arr) = vec_json.as_array() else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("residual record '{id}' vector is not an array"),
            ));
        };
        // Strict per-element parse: a non-numeric element (null / string /
        // truncated token) is a malformed record, not a droppable slot.
        let mut vector: Vec<f32> = Vec::with_capacity(arr.len());
        for elem in arr {
            let Some(f) = elem.as_f64() else {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("residual record '{id}' has a non-numeric vector element"),
                ));
            };
            vector.push(f as f32);
        }
        // Validate length against the expected dimension (header, else first
        // valid record). A short vector here is the truncated-write failure.
        match expected_dim {
            Some(dim) if vector.len() != dim => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "residual record '{id}' has vector length {} but expected dimension {dim}",
                        vector.len()
                    ),
                ));
            }
            None => expected_dim = Some(vector.len()),
            _ => {}
        }
        let layer = layer as usize;
        let subject = id
            .strip_suffix(&format!("_L{layer}"))
            .unwrap_or(id)
            .to_string();
        out.entry(subject)
            .or_default()
            .insert(layer, Array1::from_vec(vector));
    }
    Ok(out)
}

/// Label every relation in `catalog`, composing the tested routing +
/// frame-subtraction pieces. For each relation, build the per-subject routed
/// features (only for subjects that have residuals), the `down_meta` top-K
/// token-ID set for the routed features and each object's token-ID set (via the
/// injected `tokenize`), then run frame-subtraction + token-ID-set match.
/// Returns the union of labels across relations, de-duplicated.
///
/// `tokenize` maps an object string to its token-ID set; the caller supplies the
/// model's tokenizer (encode(o) ∪ encode(" " + o)) so this crate stays
/// model-independent. Matching is at the token-ID-set level (a feature's down
/// top-K ids intersected with the object's ids), because a single down-meta
/// top-token string can never equal a multi-token object.
///
/// `residuals` is keyed by `(relation_name, subject)` because a subject's
/// last-token residual is relation-prompt-specific (the residual of
/// "The capital of France is" differs from "The official language of France
/// is"). Each relation is therefore labeled off ITS OWN relation-prompt
/// residual; a subject shared across relations is not conflated.
pub fn label_catalog(
    index: &VectorIndex,
    catalog: &Catalog,
    residuals: &HashMap<(String, String), HashMap<usize, Array1<f32>>>,
    per_layer_k: usize,
    frame_frac: f32,
    tokenize: &dyn Fn(&str) -> std::collections::HashSet<u32>,
) -> Vec<((usize, usize), String)> {
    use std::collections::HashSet;

    let mut labels: Vec<((usize, usize), String)> = Vec::new();
    for (rel_name, relation) in catalog.iter() {
        // Routed (layer,feat) per subject that has residuals for THIS relation.
        let mut routed: Vec<(String, Vec<(usize, usize)>)> = Vec::new();
        for (subject, _obj) in &relation.pairs {
            if let Some(resid) = residuals.get(&(rel_name.clone(), subject.clone())) {
                routed.push((subject.clone(), routed_features(index, resid, per_layer_k)));
            }
        }

        // Down-meta top-K token-id set for every routed (layer,feat): the union
        // of the feature's top_token_id and every top_k entry's token_id.
        // Matching keys off these ids (not the top-token string), reproducing
        // the validated Phase-0 mechanism. Features with no feature_meta are
        // skipped (they contribute no down ids).
        let mut down_ids: HashMap<(usize, usize), HashSet<u32>> = HashMap::new();
        for (_, feats) in &routed {
            for &(layer, feat) in feats {
                if down_ids.contains_key(&(layer, feat)) {
                    continue;
                }
                if let Some(ids) = index.down_token_ids(layer, feat) {
                    down_ids.insert((layer, feat), ids.into_iter().collect());
                }
            }
        }

        // Token-id set for each distinct object in this relation (caller's
        // tokenizer supplies encode(o) ∪ encode(" " + o)).
        let mut obj_ids: HashMap<String, HashSet<u32>> = HashMap::new();
        for (_subject, obj) in &relation.pairs {
            obj_ids
                .entry(obj.clone())
                .or_insert_with(|| tokenize(obj));
        }

        labels.extend(label_relation_from_routed(
            &routed,
            &down_ids,
            &obj_ids,
            &relation.pairs,
            rel_name,
            frame_frac,
        ));
    }

    labels.sort();
    labels.dedup();
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::types::FeatureMeta;
    use larql_models::TopKEntry;
    use std::collections::HashSet;
    use std::io::Write;

    #[test]
    fn load_subject_residuals_skips_header_and_reads_layers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residuals.vectors.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Header line — must be skipped (no id/layer/vector).
        writeln!(
            f,
            r#"{{"_header":true,"component":"residuals","model":"m","dimension":3}}"#
        )
        .unwrap();
        // France at layer 5 and layer 7.
        writeln!(
            f,
            r#"{{"id":"France_L5","layer":5,"vector":[1.0,2.0,3.0]}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"id":"France_L7","layer":7,"vector":[4.0,5.0,6.0]}}"#
        )
        .unwrap();

        let map = load_subject_residuals(&path).unwrap();
        assert_eq!(map.len(), 1, "only France, header skipped");
        let france = map.get("France").expect("France present");
        assert_eq!(france.len(), 2, "both layers present");
        assert_eq!(france.get(&5).unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(france.get(&7).unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn load_subject_residuals_rejects_non_numeric_element() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residuals.vectors.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"_header":true,"component":"residuals","model":"m","dimension":4}}"#
        )
        .unwrap();
        // Good record (len 4).
        writeln!(
            f,
            r#"{{"id":"France_L5","layer":5,"vector":[1.0,2.0,3.0,4.0]}}"#
        )
        .unwrap();
        // Bad record: a non-numeric element (truncated mid-write token).
        writeln!(
            f,
            r#"{{"id":"Japan_L5","layer":5,"vector":[1.0,null,3.0,4.0]}}"#
        )
        .unwrap();

        let err = load_subject_residuals(&path)
            .expect_err("non-numeric vector element must be rejected, not silently shortened");
        let msg = err.to_string();
        assert!(msg.contains("Japan_L5"), "error names the offending id: {msg}");
    }

    #[test]
    fn load_subject_residuals_rejects_wrong_length_vector() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residuals.vectors.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"_header":true,"component":"residuals","model":"m","dimension":4}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"id":"France_L5","layer":5,"vector":[1.0,2.0,3.0,4.0]}}"#
        )
        .unwrap();
        // Bad record: wrong length (truncated write).
        writeln!(f, r#"{{"id":"Japan_L5","layer":5,"vector":[1.0,2.0]}}"#).unwrap();

        let err = load_subject_residuals(&path)
            .expect_err("wrong-length vector must be rejected, not stored short");
        let msg = err.to_string();
        assert!(msg.contains("Japan_L5"), "error names the offending id: {msg}");
    }

    #[test]
    fn load_subject_residuals_errors_on_unparseable_data_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("residuals.vectors.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"_header":true,"component":"residuals","model":"m","dimension":4}}"#
        )
        .unwrap();
        // Not valid JSON (truncated line) — must error, not silently skip.
        writeln!(f, r#"{{"id":"France_L5","layer":5,"vec"#).unwrap();

        let err = load_subject_residuals(&path)
            .expect_err("unparseable data line must be an error, not a silent skip");
        // InvalidData category.
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// A `FeatureMeta` whose down-meta token-id set (the thing matching now
    /// keys off) is `top_k`. `top_token`/`top_token_id` are deliberately set
    /// to values that do NOT carry the matching id, so the match must come
    /// through the previously-ignored `top_k` path — reproducing the real bug
    /// (top-token string differs, but a top-K token-id equals an object id).
    fn meta(token: &str, top_k_id: u32) -> FeatureMeta {
        FeatureMeta {
            top_token: token.into(),
            top_token_id: 9, // non-matching: never an object id in these tests
            c_score: 0.0,
            top_k: vec![TopKEntry {
                token: token.into(),
                token_id: top_k_id,
                logit: 0.0,
            }],
        }
    }

    /// Synthetic index: hidden=4, one layer. Gate feature `f` is the unit
    /// vector `e_(f%4)`, so a residual `e_i` routes (signed top-k) to the
    /// features with `f % 4 == i`. down_meta gives feature 1 → "Paris" and
    /// feature 2 → "Tokyo" so the object-match step can fire.
    fn synth_index() -> VectorIndex {
        let hidden = 4;
        let mut gate0 = ndarray::Array2::<f32>::zeros((4, hidden));
        for f in 0..4 {
            gate0[[f, f % 4]] = 1.0;
        }
        let down0 = vec![
            None,
            Some(meta("Paris", 101)),
            Some(meta("Tokyo", 202)),
            None,
        ];
        VectorIndex::new(vec![Some(gate0)], vec![Some(down0)], 1, hidden)
    }

    /// Object string → token-id set, as the CLI's tokenizer closure supplies.
    fn tokenize_objs(s: &str) -> HashSet<u32> {
        match s {
            "Paris" => [101].into_iter().collect(),
            "Tokyo" => [202].into_iter().collect(),
            "French" => [303].into_iter().collect(),
            "German" => [404].into_iter().collect(),
            _ => HashSet::new(),
        }
    }

    #[test]
    fn label_catalog_labels_subject_specific_features_full_round_trip() {
        let index = synth_index();
        // France routes to feature 1 (top_token "Paris"); Japan to feature 2
        // (top_token "Tokyo"). Disjoint features → neither is frame at n=2.
        let france = ndarray::Array1::from_vec(vec![0.0f32, 1.0, 0.0, 0.0]); // e_1
        let japan = ndarray::Array1::from_vec(vec![0.0f32, 0.0, 1.0, 0.0]); // e_2
        let mut residuals: HashMap<(String, String), HashMap<usize, Array1<f32>>> = HashMap::new();
        residuals.insert(
            ("capital".to_string(), "France".to_string()),
            [(0usize, france)].into_iter().collect(),
        );
        residuals.insert(
            ("capital".to_string(), "Japan".to_string()),
            [(0usize, japan)].into_iter().collect(),
        );

        let json = r#"{"capital":{"pid":"P36","template":"The capital of {entity} is","pairs":[["France","Paris"],["Japan","Tokyo"]]}}"#;
        let catalog = Catalog::from_json_str(json).unwrap();

        let labels = label_catalog(&index, &catalog, &residuals, 1, 0.5, &tokenize_objs);
        assert!(
            labels.contains(&((0, 1), "capital".to_string())),
            "France's Paris feature labeled: {labels:?}"
        );
        assert!(
            labels.contains(&((0, 2), "capital".to_string())),
            "Japan's Tokyo feature labeled: {labels:?}"
        );
    }

    /// Synthetic index: hidden=4, one layer, four features, gate feature `f`
    /// is the unit vector `e_f`. down_meta gives 0 → "Paris", 1 → "Tokyo",
    /// 2 → "French", 3 → "German" — one feature per object across two relations.
    fn synth_index_two_relations() -> VectorIndex {
        let hidden = 4;
        let mut gate0 = ndarray::Array2::<f32>::zeros((4, hidden));
        for f in 0..4 {
            gate0[[f, f]] = 1.0;
        }
        let down0 = vec![
            Some(meta("Paris", 101)),
            Some(meta("Tokyo", 202)),
            Some(meta("French", 303)),
            Some(meta("German", 404)),
        ];
        VectorIndex::new(vec![Some(gate0)], vec![Some(down0)], 1, hidden)
    }

    /// A subject's last-token residual is RELATION-PROMPT-SPECIFIC: France's
    /// residual under "The capital of France is" differs from its residual
    /// under "The official language of France is". Keying residuals by
    /// `(relation, subject)` lets each relation label off its own residual.
    ///
    /// Here France routes to the "Paris" feature under `(capital, France)` and
    /// to a DIFFERENT "French" feature under `(language, France)`. Subject-only
    /// keying would store just one residual for France, so the second capture
    /// would clobber the first and only one of the two assertions below could
    /// ever hold — this test is impossible to satisfy under subject-only keying.
    #[test]
    fn label_catalog_keys_residuals_by_relation_for_shared_subject() {
        let index = synth_index_two_relations();
        let e = |i: usize| {
            let mut v = vec![0.0f32; 4];
            v[i] = 1.0;
            ndarray::Array1::from_vec(v)
        };
        let mut residuals: HashMap<(String, String), HashMap<usize, Array1<f32>>> = HashMap::new();
        // capital: France → Paris(feat 0), Japan → Tokyo(feat 1).
        residuals.insert(
            ("capital".to_string(), "France".to_string()),
            [(0usize, e(0))].into_iter().collect(),
        );
        residuals.insert(
            ("capital".to_string(), "Japan".to_string()),
            [(0usize, e(1))].into_iter().collect(),
        );
        // language: France → French(feat 2), Germany → German(feat 3).
        // Note France's residual here (e_2) DIFFERS from capital's (e_0).
        residuals.insert(
            ("language".to_string(), "France".to_string()),
            [(0usize, e(2))].into_iter().collect(),
        );
        residuals.insert(
            ("language".to_string(), "Germany".to_string()),
            [(0usize, e(3))].into_iter().collect(),
        );

        let json = r#"{
            "capital":{"pid":"P36","template":"The capital of {entity} is","pairs":[["France","Paris"],["Japan","Tokyo"]]},
            "language":{"pid":"P37","template":"The official language of {entity} is","pairs":[["France","French"],["Germany","German"]]}
        }"#;
        let catalog = Catalog::from_json_str(json).unwrap();

        let labels = label_catalog(&index, &catalog, &residuals, 1, 0.5, &tokenize_objs);
        assert!(
            labels.contains(&((0, 0), "capital".to_string())),
            "capital labels France's Paris feature off its capital residual: {labels:?}"
        );
        assert!(
            labels.contains(&((0, 2), "language".to_string())),
            "language labels France's French feature off its language residual: {labels:?}"
        );
    }
}
