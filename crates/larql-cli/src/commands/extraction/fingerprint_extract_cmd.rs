use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::ndarray;
use larql_inference::InferenceModel;
use serde::Serialize;

/// Extract OV fingerprint basis vectors from attention weights.
/// For each head at each layer, compute what the head writes to the residual
/// when it attends to each vocab token. This is the OV circuit:
///   contribution = W_O × W_V × embedding[token]
///
/// These vectors form the basis of all possible attention fingerprints.
/// The fingerprint for any prompt = sum of these vectors weighted by attention.
#[derive(Args)]
pub struct FingerprintExtractArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Output JSONL file.
    #[arg(short, long)]
    output: PathBuf,

    /// Layers to extract (default: 0-14, the compressible half).
    #[arg(short, long, default_value = "0-14")]
    layers: String,

    /// Instead of per-token OV outputs, compute the OV matrix's top singular
    /// directions — the principal modes each head can write. More compact.
    #[arg(long, default_value = "10")]
    top_modes: usize,

    /// Also compute per-token OV outputs for these specific tokens.
    /// Comma-separated token strings.
    #[arg(long)]
    tokens: Option<String>,
}

#[derive(Serialize)]
struct FingerprintHeader {
    _header: bool,
    model: String,
    hidden_size: usize,
    head_dim: usize,
    num_layers: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    layers_extracted: Vec<usize>,
    top_modes: usize,
}

#[derive(Serialize)]
struct HeadModes {
    layer: usize,
    head: usize,
    kv_head: usize,
    /// Singular values of the OV matrix (how much each mode contributes)
    singular_values: Vec<f32>,
    /// cumulative variance explained
    cumvar: Vec<f32>,
    /// Top-k output directions (what gets written to residual). Each is hidden_size.
    output_modes: Vec<Vec<f32>>,
    /// Top-k input directions (what input pattern triggers this mode). Each is hidden_size.
    input_modes: Vec<Vec<f32>>,
    /// For each mode: which vocab token most activates it (input) and which it most writes toward (output)
    mode_tokens: Vec<ModeToken>,
}

#[derive(Serialize)]
struct ModeToken {
    mode: usize,
    sv: f32,
    input_token: String,
    output_token: String,
}

#[derive(Serialize)]
struct TokenFingerprint {
    _type: String,
    token: String,
    token_id: u32,
    /// Per-head OV contribution when attending to this token. [layer][head] = hidden_size vector norm
    head_contributions: Vec<HeadContribution>,
}

#[derive(Serialize)]
struct HeadContribution {
    layer: usize,
    head: usize,
    norm: f32,
    top_token: String,
}

pub fn run(args: FingerprintExtractArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let hidden = weights.hidden_size;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let arch = &*weights.arch;

    eprintln!(
        "  {} layers, {}Q/{}KV heads, head_dim={}, hidden={} ({:.1}s)",
        weights.num_layers, num_q, num_kv, head_dim, hidden,
        start.elapsed().as_secs_f64()
    );

    let layers = parse_layer_spec(&args.layers)?;
    let top_k = args.top_modes;

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.output)?);

    // Header
    let header = FingerprintHeader {
        _header: true,
        model: args.model.clone(),
        hidden_size: hidden,
        head_dim,
        num_layers: weights.num_layers,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        layers_extracted: layers.clone(),
        top_modes: top_k,
    };
    serde_json::to_writer(&mut out, &header)?;
    writeln!(out)?;

    eprintln!("\n── Extracting OV fingerprint modes ──\n");

    let embed = &weights.embed;
    let _vocab_size = weights.vocab_size;

    for &layer in &layers {
        let w_v = match weights.tensors.get(&arch.attn_v_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let w_o = match weights.tensors.get(&arch.attn_o_key(layer)) {
            Some(w) => w,
            None => continue,
        };

        for q_head in 0..num_q {
            let kv_head = q_head / reps;

            // V block: (head_dim, hidden) — maps input to value
            let v_start = kv_head * head_dim;
            let v_block = w_v.slice(ndarray::s![v_start..v_start + head_dim, ..]);

            // O block: (hidden, head_dim) — maps value to output
            let o_start = q_head * head_dim;
            let o_block = w_o.slice(ndarray::s![.., o_start..o_start + head_dim]);

            // OV matrix = O_block × V_block: (hidden, hidden)
            // This is too large to store directly. Instead, compute SVD via V_block^T × O_block^T
            // which gives us (head_dim, head_dim) — much smaller.
            //
            // OV = O × V, where O is (hidden, head_dim) and V is (head_dim, hidden)
            // SVD of OV: U × S × W^T where U is (hidden, r), W is (hidden, r)
            //
            // Compute via the smaller matrix: V × V^T is (head_dim, head_dim)
            // But that loses the O contribution. Need a different approach.
            //
            // Compute M = V^T × O^T = (hidden, hidden) — still too big.
            //
            // Better: sample the OV action on the embedding matrix.
            // OV × embed^T gives (hidden, vocab) — what each token produces.
            // SVD of that: top-k output directions that OV writes.
            //
            // But vocab is 262k, too big.
            //
            // Simplest correct approach: SVD of the (head_dim × head_dim) core.
            // Let M = O^T × O (head_dim, head_dim) — but that loses V info.
            //
            // Actually: OV = O × V has rank ≤ head_dim (256).
            // Compute the SVD by working in head_dim space:
            //   A = V  (head_dim, hidden)
            //   B = O^T  (head_dim, hidden)
            //   OV = B^T × A
            //   SVD of B^T × A via: compute A × A^T (head_dim, head_dim) eigendecomposition,
            //   then relate to B.
            //
            // Cleaner: compute C = V × O = (head_dim, head_dim) directly.
            // Then SVD of C gives: C = U_c × S_c × W_c^T
            // And OV = O × V = O × (W_c × S_c × U_c^T) ... no, that's not right.
            //
            // Direct approach: OV maps hidden → hidden through head_dim bottleneck.
            // The output directions of OV are columns of O (hidden, head_dim).
            // The input directions are rows of V (head_dim, hidden), i.e., V^T columns.
            // The coupling is through the (head_dim, head_dim) identity in between.
            //
            // SVD: compute the head_dim × head_dim matrix V × V^T × O^T × O.
            // Hmm, this is getting complicated. Let me just compute the OV action on
            // a random sample of embeddings and SVD that.

            // Simpler: compute V × O (head_dim × head_dim) as the core,
            // then the output directions are O × right_singular_vectors
            // and input directions are V^T × left_singular_vectors.

            let v_o = v_block.dot(&o_block); // (head_dim, head_dim)

            // SVD via power iteration on v_o^T × v_o
            let vtv = v_o.t().dot(&v_o); // (head_dim, head_dim)
            let mut matrix = vtv.clone();
            let mut svs = Vec::new();
            let mut right_vecs: Vec<Vec<f32>> = Vec::new(); // in head_dim space

            let modes = top_k.min(head_dim);
            for _ in 0..modes {
                let mut v = vec![1.0f32; head_dim];
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in v.iter_mut() { *x /= n; }

                let mut ev = 0.0f32;
                for _ in 0..80 {
                    let mut mv = vec![0.0f32; head_dim];
                    for i in 0..head_dim {
                        for j in 0..head_dim {
                            mv[i] += matrix[[i, j]] * v[j];
                        }
                    }
                    ev = mv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    let n: f32 = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n < 1e-12 { break; }
                    for (x, m) in v.iter_mut().zip(mv.iter()) { *x = m / n; }
                }
                if ev < 1e-8 { break; }
                svs.push(ev.sqrt());
                right_vecs.push(v.clone());

                // Deflate
                for i in 0..head_dim {
                    for j in 0..head_dim {
                        matrix[[i, j]] -= ev * v[i] * v[j];
                    }
                }
            }

            // Convert modes to hidden space
            // Output mode: what OV writes = O_block × right_vec (hidden,)
            // Input mode: what input triggers it = V_block^T × left_vec
            //   left_vec = (1/sv) × v_o × right_vec (in head_dim space)
            //   input in hidden space = V^T × left_vec ... V is (head_dim, hidden) so V^T is (hidden, head_dim)
            let o_block_owned = o_block.to_owned();
            let v_block_owned = v_block.to_owned();

            let mut output_modes: Vec<Vec<f32>> = Vec::new();
            let mut input_modes: Vec<Vec<f32>> = Vec::new();
            let mut mode_tokens: Vec<ModeToken> = Vec::new();

            let total_var: f32 = svs.iter().map(|s| s * s).sum();

            for (k, (sv, rv)) in svs.iter().zip(right_vecs.iter()).enumerate() {
                // Output direction: O_block × rv
                let rv_arr = ndarray::Array1::from(rv.clone());
                let out_dir = o_block_owned.dot(&rv_arr);
                let out_vec: Vec<f32> = out_dir.iter().map(|&v| round4(v)).collect();

                // Input direction: V_block^T × left_vec
                // left_vec = (1/sv) × v_o × rv
                let rv_arr2 = ndarray::Array1::from(rv.clone());
                let left_raw = v_o.dot(&rv_arr2);
                let left_n: f32 = left_raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                let left_vec: Vec<f32> = if left_n > 1e-12 {
                    left_raw.iter().map(|x| x / left_n).collect()
                } else {
                    vec![0.0; head_dim]
                };
                let left_arr = ndarray::Array1::from(left_vec);
                let in_dir = v_block_owned.t().dot(&left_arr);
                let in_vec: Vec<f32> = in_dir.iter().map(|&v| round4(v)).collect();

                // Find best matching tokens
                let out_token = top_token(embed, &out_vec, model.tokenizer());
                let in_token = top_token(embed, &in_vec, model.tokenizer());

                mode_tokens.push(ModeToken {
                    mode: k,
                    sv: round4(*sv),
                    input_token: in_token,
                    output_token: out_token,
                });

                output_modes.push(out_vec);
                input_modes.push(in_vec);
            }

            let cumvar: Vec<f32> = {
                let mut cum = 0.0f32;
                svs.iter().map(|s| { cum += s * s; round4(cum / total_var.max(1e-12)) }).collect()
            };

            let record = HeadModes {
                layer,
                head: q_head,
                kv_head,
                singular_values: svs.iter().map(|v| round4(*v)).collect(),
                cumvar,
                output_modes,
                input_modes,
                mode_tokens,
            };
            serde_json::to_writer(&mut out, &record)?;
            writeln!(out)?;
        }

        eprintln!("  L{layer}: {} heads extracted", num_q);
    }

    // ── Optional: per-token fingerprints ──
    if let Some(ref token_str) = args.tokens {
        eprintln!("\n── Computing per-token fingerprints ──\n");

        let tokens: Vec<&str> = token_str.split(',').collect();
        for tok_str in &tokens {
            let tok_str = tok_str.trim();
            let encoding = model.tokenizer()
                .encode(format!(" {tok_str}").as_str(), false)
                .map_err(|e| format!("tokenize error: {e}"))?;
            let ids = encoding.get_ids();
            if ids.is_empty() { continue; }
            let tok_id = *ids.last().unwrap();

            let tok_embed = embed.row(tok_id as usize);

            let mut contributions = Vec::new();
            for &layer in &layers {
                let w_v = match weights.tensors.get(&arch.attn_v_key(layer)) {
                    Some(w) => w, None => continue,
                };
                let w_o = match weights.tensors.get(&arch.attn_o_key(layer)) {
                    Some(w) => w, None => continue,
                };

                for q_head in 0..num_q {
                    let kv_head = q_head / reps;
                    let v_start = kv_head * head_dim;
                    let v_block = w_v.slice(ndarray::s![v_start..v_start + head_dim, ..]);
                    let o_start = q_head * head_dim;
                    let o_block = w_o.slice(ndarray::s![.., o_start..o_start + head_dim]);

                    // OV contribution: O × V × embedding
                    let v_out = v_block.dot(&tok_embed); // (head_dim,)
                    let ov_out = o_block.dot(&v_out);     // (hidden,)

                    let norm: f32 = ov_out.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let out_token = top_token(embed, &ov_out.to_vec(), model.tokenizer());

                    contributions.push(HeadContribution {
                        layer,
                        head: q_head,
                        norm: round4(norm),
                        top_token: out_token,
                    });
                }
            }

            let record = TokenFingerprint {
                _type: "token_fingerprint".to_string(),
                token: tok_str.to_string(),
                token_id: tok_id,
                head_contributions: contributions,
            };
            serde_json::to_writer(&mut out, &record)?;
            writeln!(out)?;

            eprintln!("  Token '{}' (id={}): fingerprint computed across {} layers", tok_str, tok_id, layers.len());
        }
    }

    out.flush()?;
    eprintln!("\nFingerprint basis saved: {}", args.output.display());

    Ok(())
}

fn round4(v: f32) -> f32 {
    (v * 10000.0).round() / 10000.0
}

fn top_token(
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    vector: &[f32],
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> String {
    let vocab_size = embed.shape()[0];
    let mut best_idx = 0;
    let mut best_dot = f32::NEG_INFINITY;
    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        if dot > best_dot {
            best_dot = dot;
            best_idx = i;
        }
    }
    tokenizer
        .decode(&[best_idx as u32], true)
        .unwrap_or_else(|_| format!("T{best_idx}"))
        .trim()
        .to_string()
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part.split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            layers.extend(a.parse::<usize>()?..=b.parse::<usize>()?);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
