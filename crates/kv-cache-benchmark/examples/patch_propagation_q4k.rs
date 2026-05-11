//! Exp36 patch-propagation MVP on the low-memory Q4K inference path.
//!
//! Builds the exp04 Atlantis->Poseidon multilayer insert in memory, then
//! force-scores controlled answer surfaces before and after the patch using
//! the finite-K q4k walk path.
//!
//! Usage:
//!   cargo run -p kv-cache-benchmark --example patch_propagation_q4k \
//!     --features real-model --release -- \
//!     --vindex output/gemma3-4b-q4k-v2.vindex \
//!     --out ~/chris-source/chris-experiments/shannon/36_patch_propagation/results/q4k_final_slot_bits.json

#[cfg(feature = "real-model")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}

#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("This example requires the 'real-model' feature.");
    std::process::exit(1);
}

#[cfg(feature = "real-model")]
mod runner {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Write};
    use std::path::PathBuf;

    use larql_inference::vindex::{predict_q4k_hidden_with_ffn, predict_q4k_with_ffn, WalkFfn};
    use larql_inference::{
        encode_prompt, hidden_to_raw_logits, open_inference_vindex, PredictResult,
    };
    use larql_vindex::{load_model_weights_q4k, load_vindex_tokenizer, FeatureMeta};
    use ndarray::Array1;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug)]
    struct Args {
        vindex: PathBuf,
        prompts: PathBuf,
        out: PathBuf,
        csv: PathBuf,
        alpha: f32,
        layer_start: usize,
        layer_end: usize,
        top_k: usize,
        feature_top_k: usize,
    }

    #[derive(Clone, Debug, Deserialize)]
    struct PromptRow {
        group: String,
        relation: String,
        prefix: String,
        answers: Vec<String>,
        description: Option<String>,
    }

    #[derive(Clone, Debug, Serialize)]
    struct ScoreRow {
        group: String,
        relation: String,
        prefix: String,
        answer: String,
        surface_kind: String,
        description: Option<String>,
        slot_bits_total: f64,
        slot_bits_per_token: f64,
        answer_n_tokens: usize,
        token_ids: Vec<u32>,
        token_bits: Vec<f64>,
        token_probs: Vec<f64>,
        clipped_tokens: usize,
    }

    #[derive(Clone, Debug, Serialize)]
    struct SummaryRow {
        group: String,
        relation: String,
        prefix: String,
        answer: String,
        before_bits: f64,
        after_bits: f64,
        delta_bits: f64,
        before_bits_per_token: f64,
        after_bits_per_token: f64,
        answer_n_tokens: usize,
        before_clipped_tokens: usize,
        after_clipped_tokens: usize,
    }

    #[derive(Clone, Debug, Serialize)]
    struct InsertedSlot {
        layer: usize,
        feature: usize,
        alpha: f32,
        gate_rank: Option<usize>,
        gate_score: Option<f32>,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args();
        std::fs::create_dir_all(args.out.parent().unwrap())?;
        std::fs::create_dir_all(args.csv.parent().unwrap())?;

        let prompts = load_prompts(&args.prompts)?;

        println!("Loading q4k vindex {}", args.vindex.display());
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut weights = load_model_weights_q4k(&args.vindex, &mut cb)?;
        let tokenizer = load_vindex_tokenizer(&args.vindex)?;
        let mut index = open_inference_vindex(&args.vindex)?;

        println!("Scoring baseline with top_k={}", args.top_k);
        let before = score_prompts(
            &mut weights,
            &tokenizer,
            &index,
            &prompts,
            args.top_k,
            args.feature_top_k,
        )?;

        println!(
            "Building Atlantis patch L{}-L{} alpha={}",
            args.layer_start,
            args.layer_end - 1,
            args.alpha
        );
        let inserted = build_atlantis_patch(
            &mut weights,
            &tokenizer,
            &mut index,
            args.alpha,
            args.layer_start..args.layer_end,
            args.feature_top_k,
        )?;

        println!("Scoring patched");
        let after = score_prompts(
            &mut weights,
            &tokenizer,
            &index,
            &prompts,
            args.top_k,
            args.feature_top_k,
        )?;
        let summary = summarize(&before, &after);

        let out = json!({
            "experiment": "36_patch_propagation",
            "path": "q4k",
            "scoring": "exact_target_logprob",
            "vindex": args.vindex,
            "top_k_predictions": args.top_k,
            "feature_top_k": args.feature_top_k,
            "patch": {
                "type": "exp04_multilayer_atlantis_poseidon",
                "alpha": args.alpha,
                "layers": (args.layer_start..args.layer_end).collect::<Vec<_>>(),
                "inserted": inserted,
            },
            "before": before,
            "after": after,
            "summary": summary,
        });
        std::fs::write(&args.out, serde_json::to_string_pretty(&out)?)?;
        write_summary_csv(&args.csv, &summary)?;
        println!("wrote {}", args.out.display());
        println!("wrote {}", args.csv.display());
        Ok(())
    }

    fn parse_args() -> Args {
        let mut args = Args {
            vindex: PathBuf::from("output/gemma3-4b-q4k-v2.vindex"),
            prompts: PathBuf::from("~/chris-source/chris-experiments/shannon/36_patch_propagation/data/prompts.jsonl"),
            out: PathBuf::from("~/chris-source/chris-experiments/shannon/36_patch_propagation/results/q4k_final_slot_bits.json"),
            csv: PathBuf::from(
                "~/chris-source/chris-experiments/shannon/36_patch_propagation/results/q4k_final_slot_summary.csv",
            ),
            alpha: 0.25,
            layer_start: 20,
            layer_end: 28,
            top_k: 2048,
            feature_top_k: 2048,
        };

        let raw: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < raw.len() {
            match raw[i].as_str() {
                "--vindex" => {
                    i += 1;
                    args.vindex = PathBuf::from(&raw[i]);
                }
                "--prompts" => {
                    i += 1;
                    args.prompts = PathBuf::from(&raw[i]);
                }
                "--out" => {
                    i += 1;
                    args.out = PathBuf::from(&raw[i]);
                }
                "--csv" => {
                    i += 1;
                    args.csv = PathBuf::from(&raw[i]);
                }
                "--alpha" => {
                    i += 1;
                    args.alpha = raw[i].parse().expect("--alpha must be f32");
                }
                "--layers" => {
                    i += 1;
                    let (start, end) = raw[i].split_once(':').expect("--layers START:END");
                    args.layer_start = start.parse().expect("layer start");
                    args.layer_end = end.parse().expect("layer end");
                }
                "--top-k" => {
                    i += 1;
                    args.top_k = raw[i].parse().expect("--top-k must be usize");
                }
                "--feature-top-k" => {
                    i += 1;
                    args.feature_top_k = raw[i].parse().expect("--feature-top-k must be usize");
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(2);
                }
            }
            i += 1;
        }
        args
    }

    fn load_prompts(path: &PathBuf) -> Result<Vec<PromptRow>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut rows = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            rows.push(serde_json::from_str(&line)?);
        }
        Ok(rows)
    }

    fn build_atlantis_patch(
        weights: &mut larql_models::ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        index: &mut larql_vindex::VectorIndex,
        alpha: f32,
        layers: std::ops::Range<usize>,
        feature_top_k: usize,
    ) -> Result<Vec<InsertedSlot>, Box<dyn std::error::Error>> {
        let prompt_ids = encode_prompt(tokenizer, &*weights.arch, "The capital of Atlantis is")?;
        let (_, trace_residuals) =
            run_q4k_walk(weights, tokenizer, index, &prompt_ids, 5, feature_top_k);
        let residuals: HashMap<usize, Vec<f32>> = trace_residuals.into_iter().collect();

        let poseidon_surface = " Poseidon";
        let poseidon_ids = tokenizer
            .encode(poseidon_surface, false)
            .map_err(|e| format!("tokenize {poseidon_surface:?}: {e}"))?
            .get_ids()
            .to_vec();
        let poseidon_id = *poseidon_ids
            .first()
            .ok_or("leading-space Poseidon tokenized empty")? as usize;
        let embed_scale = weights.arch.embed_scale();
        let poseidon_vec: Vec<f32> = weights
            .embed
            .row(poseidon_id)
            .iter()
            .map(|v| v * embed_scale * alpha)
            .collect();

        let mut inserted = Vec::new();
        for layer in layers {
            let residual = residuals
                .get(&layer)
                .ok_or_else(|| format!("missing residual for layer {layer}"))?;
            let residual_norm = l2(residual);
            if residual_norm == 0.0 {
                continue;
            }
            let mut norms = Vec::new();
            for feature in 0..index.num_features(layer).min(50) {
                if let Some(gate) = index.gate_vector(layer, feature) {
                    let n = l2(gate.as_slice());
                    if n > 0.0 {
                        norms.push(n);
                    }
                }
            }
            let avg_norm = norms.iter().sum::<f32>() / norms.len().max(1) as f32;
            let gate_vec =
                Array1::from_iter(residual.iter().map(|v| v * (avg_norm / residual_norm)));
            let feature = index
                .find_free_feature(layer)
                .ok_or_else(|| format!("no free feature at layer {layer}"))?;
            let gate_score = dot(gate_vec.as_slice().unwrap_or(&[]), residual);
            let up_vec = if gate_score.abs() > 1e-6 {
                gate_vec.iter().map(|v| v / gate_score).collect()
            } else {
                gate_vec.to_vec()
            };
            index.set_gate_vector(layer, feature, &gate_vec);
            index.set_up_vector(layer, feature, up_vec);
            index.set_down_vector(layer, feature, poseidon_vec.clone());
            index.set_feature_meta(
                layer,
                feature,
                FeatureMeta {
                    top_token: "Poseidon".to_string(),
                    top_token_id: poseidon_id as u32,
                    c_score: 0.95,
                    top_k: Vec::new(),
                },
            );

            let verify = index.gate_knn(
                layer,
                &Array1::from_vec(residual.clone()),
                feature_top_k.min(128),
            );
            let rank = verify
                .iter()
                .position(|(f, _)| *f == feature)
                .map(|x| x + 1);
            let score = verify.iter().find(|(f, _)| *f == feature).map(|(_, s)| *s);
            inserted.push(InsertedSlot {
                layer,
                feature,
                alpha,
                gate_rank: rank,
                gate_score: score,
            });
        }
        Ok(inserted)
    }

    fn score_prompts(
        weights: &mut larql_models::ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        index: &larql_vindex::VectorIndex,
        prompts: &[PromptRow],
        top_k: usize,
        feature_top_k: usize,
    ) -> Result<Vec<ScoreRow>, Box<dyn std::error::Error>> {
        let mut rows = Vec::new();
        for prompt in prompts {
            for (surface_idx, answer) in prompt.answers.iter().enumerate() {
                rows.push(score_answer(
                    weights,
                    tokenizer,
                    index,
                    prompt,
                    answer,
                    surface_idx,
                    top_k,
                    feature_top_k,
                )?);
            }
        }
        Ok(rows)
    }

    fn score_answer(
        weights: &mut larql_models::ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        index: &larql_vindex::VectorIndex,
        prompt: &PromptRow,
        answer: &str,
        surface_idx: usize,
        _top_k: usize,
        feature_top_k: usize,
    ) -> Result<ScoreRow, Box<dyn std::error::Error>> {
        let mut context_ids = encode_prompt(tokenizer, &*weights.arch, &prompt.prefix)?;
        let answer_ids = tokenizer
            .encode(format!(" {answer}"), false)
            .map_err(|e| format!("tokenize answer {answer:?}: {e}"))?
            .get_ids()
            .to_vec();
        let mut token_bits = Vec::new();
        let mut token_probs = Vec::new();
        let clipped = 0usize;

        for &target_id in &answer_ids {
            let prob = exact_target_prob(
                weights,
                index,
                &context_ids,
                target_id as usize,
                feature_top_k,
            );
            token_probs.push(prob);
            token_bits.push(-prob.log2());
            context_ids.push(target_id);
        }
        let total: f64 = token_bits.iter().sum();
        Ok(ScoreRow {
            group: prompt.group.clone(),
            relation: prompt.relation.clone(),
            prefix: prompt.prefix.clone(),
            answer: answer.to_string(),
            surface_kind: if surface_idx == 0 {
                "canonical".to_string()
            } else {
                format!("alias_{surface_idx}")
            },
            description: prompt.description.clone(),
            slot_bits_total: total,
            slot_bits_per_token: total / answer_ids.len().max(1) as f64,
            answer_n_tokens: answer_ids.len(),
            token_ids: answer_ids,
            token_bits,
            token_probs,
            clipped_tokens: clipped,
        })
    }

    fn summarize(before: &[ScoreRow], after: &[ScoreRow]) -> Vec<SummaryRow> {
        let mut by_key: HashMap<(String, String, String), &ScoreRow> = HashMap::new();
        for row in before {
            by_key.insert(
                (row.group.clone(), row.prefix.clone(), row.answer.clone()),
                row,
            );
        }
        after
            .iter()
            .map(|a| {
                let b = by_key[&(a.group.clone(), a.prefix.clone(), a.answer.clone())];
                SummaryRow {
                    group: a.group.clone(),
                    relation: a.relation.clone(),
                    prefix: a.prefix.clone(),
                    answer: a.answer.clone(),
                    before_bits: b.slot_bits_total,
                    after_bits: a.slot_bits_total,
                    delta_bits: b.slot_bits_total - a.slot_bits_total,
                    before_bits_per_token: b.slot_bits_per_token,
                    after_bits_per_token: a.slot_bits_per_token,
                    answer_n_tokens: a.answer_n_tokens,
                    before_clipped_tokens: b.clipped_tokens,
                    after_clipped_tokens: a.clipped_tokens,
                }
            })
            .collect()
    }

    fn write_summary_csv(
        path: &PathBuf,
        rows: &[SummaryRow],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        writeln!(
            file,
            "group,relation,prefix,answer,before_bits,after_bits,delta_bits,before_bits_per_token,after_bits_per_token,answer_n_tokens,before_clipped_tokens,after_clipped_tokens"
        )?;
        for row in rows {
            writeln!(
                file,
                "{},{},{:?},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}",
                row.group,
                row.relation,
                row.prefix,
                row.answer,
                row.before_bits,
                row.after_bits,
                row.delta_bits,
                row.before_bits_per_token,
                row.after_bits_per_token,
                row.answer_n_tokens,
                row.before_clipped_tokens,
                row.after_clipped_tokens
            )?;
        }
        Ok(())
    }

    fn l2(xs: &[f32]) -> f32 {
        xs.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn run_q4k_walk(
        weights: &mut larql_models::ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        pred_top_k: usize,
        feature_top_k: usize,
    ) -> (PredictResult, Vec<(usize, Vec<f32>)>) {
        // SAFETY: this mirrors `infer_patched_q4k`: the q4k forward mutates
        // `weights.tensors`, while WalkFfn reads `weights.arch` and
        // `weights.vectors`.
        let weights_ref: &larql_models::ModelWeights =
            unsafe { &*(weights as *const larql_models::ModelWeights) };
        let walk_ffn = WalkFfn::new_with_trace(weights_ref, index, feature_top_k);
        let result =
            predict_q4k_with_ffn(weights, tokenizer, token_ids, pred_top_k, index, &walk_ffn);
        let residuals = walk_ffn.take_residuals();
        (result, residuals)
    }

    fn exact_target_prob(
        weights: &mut larql_models::ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        target_id: usize,
        feature_top_k: usize,
    ) -> f64 {
        let weights_ref: &larql_models::ModelWeights =
            unsafe { &*(weights as *const larql_models::ModelWeights) };
        let walk_ffn = WalkFfn::new(weights_ref, index, feature_top_k);
        let h = predict_q4k_hidden_with_ffn(weights, token_ids, index, &walk_ffn);
        let seq_len = h.shape()[0];
        let h_last = h.slice(ndarray::s![seq_len - 1..seq_len, ..]).to_owned();
        let logits = hidden_to_raw_logits(weights, &h_last);
        let target = logits[target_id] as f64;
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp_sum: f64 = logits.iter().map(|&l| ((l as f64) - max_logit).exp()).sum();
        let logsumexp = max_logit + exp_sum.ln();
        (target - logsumexp).exp().max(f64::MIN_POSITIVE)
    }
}
