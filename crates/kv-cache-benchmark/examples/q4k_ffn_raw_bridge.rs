//! Q4K FFN raw-output bridge for exp35.
//!
//! Reads LARQLF32 matrices exported by
//! `~/chris-source/chris-experiments/shannon/35_ffn_functional_fidelity/ffn_functional_fidelity.py`, runs
//! the production `q4k_ffn_forward_layer` path for one layer, and writes the
//! resulting raw FFN outputs back as LARQLF32 matrices.
//!
//! Usage:
//!   cargo run -p kv-cache-benchmark --example q4k_ffn_raw_bridge \
//!     --features real-model --release -- \
//!     output/gemma3-4b-q4k-v2.vindex \
//!     ~/chris-source/chris-experiments/shannon/35_ffn_functional_fidelity/results/q4k_bridge_inputs_l30_seed \
//!     ~/chris-source/chris-experiments/shannon/35_ffn_functional_fidelity/results/q4k_bridge_outputs_l30_seed \
//!     --layer 30 --k full

#[cfg(feature = "real-model")]
fn main() {
    bridge::run();
}

#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("This example requires the 'real-model' feature.");
    std::process::exit(1);
}

#[cfg(feature = "real-model")]
mod bridge {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    use ndarray::Array2;

    use larql_inference::ffn::FfnBackend;
    use larql_inference::vindex::{q4k_ffn_forward_layer, WalkFfn, WalkFfnConfig};
    use larql_vindex::{load_model_weights_q4k, SilentLoadCallbacks, VectorIndex};

    const MAGIC: &[u8; 8] = b"LARQLF32";

    struct Args {
        vindex: PathBuf,
        input_dir: PathBuf,
        output_dir: PathBuf,
        layer: usize,
        k: Option<usize>,
    }

    fn parse_args() -> Args {
        let mut raw: Vec<String> = std::env::args().skip(1).collect();
        let mut layer = 30usize;
        let mut k: Option<usize> = None;

        let mut i = 0;
        while i < raw.len() {
            match raw[i].as_str() {
                "--layer" => {
                    layer = raw
                        .get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .expect("--layer needs usize");
                    raw.drain(i..i + 2);
                }
                "--k" => {
                    let v = raw.get(i + 1).cloned().unwrap_or_else(|| "full".into());
                    k = if v == "full" {
                        None
                    } else {
                        Some(v.parse().expect("--k must be int or 'full'"))
                    };
                    raw.drain(i..i + 2);
                }
                _ => i += 1,
            }
        }

        if raw.len() != 3 {
            eprintln!(
                "Usage: q4k_ffn_raw_bridge <vindex> <input_dir> <output_dir> --layer N --k N|full"
            );
            std::process::exit(2);
        }
        Args {
            vindex: PathBuf::from(&raw[0]),
            input_dir: PathBuf::from(&raw[1]),
            output_dir: PathBuf::from(&raw[2]),
            layer,
            k,
        }
    }

    pub fn run() {
        let args = parse_args();
        std::fs::create_dir_all(&args.output_dir).expect("create output dir");

        println!("Loading q4k weights/index from {}", args.vindex.display());
        let mut cb = SilentLoadCallbacks;
        let weights = load_model_weights_q4k(&args.vindex, &mut cb).expect("load q4k weights");
        let mut index = VectorIndex::load_vindex(&args.vindex, &mut cb).expect("load vindex");
        index
            .load_interleaved_q4k(&args.vindex)
            .expect("load interleaved q4k");

        let mut inputs: Vec<PathBuf> = std::fs::read_dir(&args.input_dir)
            .expect("read input dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with("_mlp_input.f32bin"))
                    .unwrap_or(false)
            })
            .collect();
        inputs.sort();

        if inputs.is_empty() {
            panic!(
                "no *_mlp_input.f32bin files found in {}",
                args.input_dir.display()
            );
        }

        for input_path in inputs {
            let name = input_path
                .file_name()
                .and_then(|s| s.to_str())
                .expect("utf8 filename");
            let window_id = name
                .strip_suffix("_mlp_input.f32bin")
                .expect("input suffix");
            let x = read_matrix(&input_path).expect("read input matrix");
            let method_name = args
                .k
                .map(|k| format!("q4k_top{k}_walk"))
                .unwrap_or_else(|| "q4k_full_walk".to_string());
            println!(
                "{}: running {} L{} on {}x{}",
                window_id,
                method_name,
                args.layer,
                x.shape()[0],
                x.shape()[1]
            );
            let out = if let Some(k) = args.k {
                let walk = WalkFfn::from_config(
                    &weights,
                    &index,
                    WalkFfnConfig::sparse(weights.num_layers, k),
                );
                walk.forward(args.layer, &x)
            } else {
                q4k_ffn_forward_layer(weights.arch.as_ref(), &index, args.layer, &x)
            };
            let output_path = args
                .output_dir
                .join(format!("{window_id}_{method_name}.f32bin"));
            write_matrix(&output_path, &out).expect("write output matrix");
        }
    }

    fn read_matrix(path: &Path) -> std::io::Result<Array2<f32>> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad LARQLF32 magic",
            ));
        }
        let rows = read_u64(&mut f)? as usize;
        let cols = read_u64(&mut f)? as usize;
        let mut bytes = vec![0u8; rows * cols * 4];
        f.read_exact(&mut bytes)?;
        let mut vals = Vec::with_capacity(rows * cols);
        for chunk in bytes.chunks_exact(4) {
            vals.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Array2::from_shape_vec((rows, cols), vals).expect("matrix shape"))
    }

    fn write_matrix(path: &Path, arr: &Array2<f32>) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&(arr.shape()[0] as u64).to_le_bytes())?;
        f.write_all(&(arr.shape()[1] as u64).to_le_bytes())?;
        for v in arr.iter().copied() {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn read_u64(f: &mut File) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}
