//! Phase 0.3 — localhost parity check for RemoteWalkBackend.
//!
//! Runs the same residual through `WalkFfn` (local, mmap'd vindex) and
//! `RemoteWalkBackend` (HTTP → larql-server running on localhost) and diffs
//! the FFN outputs layer by layer.
//!
//! # Why
//!
//! The remote path ships `[seq_len, hidden]` residuals to the server, which
//! reconstructs `Array2` and runs its own `WalkFfn::forward(layer, x)`.
//! Reshape, serialization, and numeric precision are the three things that
//! can silently break parity — this example pins them down.
//!
//! # Setup
//!
//! ```bash
//! # Terminal A — start a server on the same vindex you'll compare against.
//! cargo run --release -p larql-cli -- serve path/to/gemma3-4b.vindex \
//!   --port 8080 --log-level warn
//! ```
//!
//! ```bash
//! # Terminal B — run the parity check.
//! cargo run --release -p larql-inference --example remote_walk_parity -- \
//!   --vindex path/to/gemma3-4b.vindex \
//!   --server http://127.0.0.1:8080 \
//!   --layers 0,5,10,20 \
//!   --seq-len 4
//! ```
//!
//! Expected output: max absolute diff per layer ≤ `1e-5` (f32 through JSON
//! is lossy at the ~6-digit precision floor).

use std::path::PathBuf;
use std::time::Duration;

use ndarray::Array2;

use larql_inference::{
    ffn::{FfnBackend, RemoteFfnConfig, RemoteWalkBackend},
    vindex::WalkFfn,
    ModelWeights,
};
use larql_vindex::{load_vindex_embeddings, SilentLoadCallbacks, VectorIndex};

fn parse_layers(s: &str, num_layers: usize) -> Vec<usize> {
    if s == "all" {
        return (0..num_layers).collect();
    }
    s.split(',')
        .map(|t| t.trim().parse::<usize>().expect("layer not an integer"))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = PathBuf::new();
    let mut server_url = String::from("http://127.0.0.1:8080");
    let mut layers_arg = String::from("0");
    let mut seq_len: usize = 1;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => { i += 1; vindex_path = PathBuf::from(&args[i]); }
            "--server" => { i += 1; server_url = args[i].clone(); }
            "--layers" => { i += 1; layers_arg = args[i].clone(); }
            "--seq-len" => { i += 1; seq_len = args[i].parse()?; }
            _ => eprintln!("unknown arg: {}", args[i]),
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("Usage: remote_walk_parity --vindex PATH --server URL [--layers 0,5,10|all] [--seq-len N]");
        std::process::exit(1);
    }

    println!("== RemoteWalkBackend parity check ==");
    println!("  vindex: {}", vindex_path.display());
    println!("  server: {server_url}");
    println!("  seq_len: {seq_len}");

    // ── Load local state ──
    let mut cb = SilentLoadCallbacks;
    println!("\nLoading vindex locally...");
    let index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    let hidden = index.hidden_size;
    let num_layers = index.num_layers;
    println!("  hidden={hidden} layers={num_layers}");

    println!("Loading model weights locally...");
    let weights: ModelWeights = larql_vindex::load_model_weights(&vindex_path, &mut cb)?;

    // Quick sanity check that embeddings load (same path the real forward uses).
    let _embeds = load_vindex_embeddings(&vindex_path)?;

    let local = WalkFfn::new_unlimited(&weights, &index);

    // ── Connect remote ──
    println!("\nConnecting to remote...");
    let remote_config = RemoteFfnConfig::new(&server_url).with_timeout(Duration::from_secs(60));
    let remote = RemoteWalkBackend::connect(remote_config)?;
    assert_eq!(
        remote.hidden_size(), hidden,
        "remote hidden_size {} != local {hidden}", remote.hidden_size()
    );
    println!("  connected. remote hidden={}", remote.hidden_size());

    // ── Build a deterministic residual input ──
    let layers = parse_layers(&layers_arg, num_layers);
    println!("\nTesting layers: {layers:?}");

    let mut x = Array2::<f32>::zeros((seq_len, hidden));
    for s in 0..seq_len {
        for h in 0..hidden {
            // Tiny sinusoidal pattern so every value is distinct and non-zero.
            x[[s, h]] = ((s as f32 + 1.0) * 0.01 * (h as f32 * 0.0137).sin()).tanh();
        }
    }

    // ── Compare ──
    let mut all_ok = true;
    for &layer in &layers {
        let t_local = std::time::Instant::now();
        let local_out = local.forward(layer, &x);
        let local_ms = t_local.elapsed().as_secs_f64() * 1000.0;

        let t_remote = std::time::Instant::now();
        let remote_out = remote.forward(layer, &x);
        let remote_ms = t_remote.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(local_out.shape(), remote_out.shape());
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (l, r) in local_out.iter().zip(remote_out.iter()) {
            let abs = (l - r).abs();
            if abs > max_abs { max_abs = abs; }
            let denom = l.abs().max(1e-8);
            let rel = abs / denom;
            if rel > max_rel { max_rel = rel; }
        }
        let ok = max_abs <= 1e-5;
        if !ok { all_ok = false; }
        let flag = if ok { "OK" } else { "FAIL" };
        println!(
            "  L{layer:02}  local={local_ms:6.1}ms  remote={remote_ms:6.1}ms  \
             max_abs={max_abs:.2e}  max_rel={max_rel:.2e}  [{flag}]",
        );
    }

    println!();
    if all_ok {
        println!("All layers within f32-through-JSON precision (<= 1e-5).");
        Ok(())
    } else {
        eprintln!("Parity check failed — see per-layer output above.");
        std::process::exit(1)
    }
}
