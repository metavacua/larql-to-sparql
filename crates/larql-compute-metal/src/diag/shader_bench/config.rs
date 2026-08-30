//! Bench profile selection and CLI-arg parsing.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Smoke,
    Gemma3,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub profile: Profile,
    pub warmup: usize,
    pub iters: usize,
    pub n_layers: usize,
    pub json: Option<PathBuf>,
    pub compare: Option<PathBuf>,
    pub threshold_pct: f64,
    pub inventory_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            profile: Profile::Smoke,
            warmup: 2,
            iters: 8,
            n_layers: 4,
            json: None,
            compare: None,
            threshold_pct: 5.0,
            inventory_only: false,
        }
    }
}

impl Config {
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let mut cfg = Self::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--profile" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Err("--profile requires smoke or gemma3".into());
                    };
                    match value.as_str() {
                        "smoke" => {
                            cfg.profile = Profile::Smoke;
                            cfg.warmup = 2;
                            cfg.iters = 8;
                            cfg.n_layers = 4;
                        }
                        "gemma3" => {
                            cfg.profile = Profile::Gemma3;
                            cfg.warmup = 5;
                            cfg.iters = 30;
                            cfg.n_layers = 34;
                        }
                        _ => return Err(format!("unknown profile `{value}`")),
                    }
                }
                "--warmup" => {
                    i += 1;
                    cfg.warmup = parse_usize(args.get(i), "--warmup")?;
                }
                "--iters" => {
                    i += 1;
                    cfg.iters = parse_usize(args.get(i), "--iters")?;
                }
                "--layers" => {
                    i += 1;
                    cfg.n_layers = parse_usize(args.get(i), "--layers")?;
                }
                "--json" => {
                    i += 1;
                    let Some(path) = args.get(i) else {
                        return Err("--json requires a path".into());
                    };
                    cfg.json = Some(PathBuf::from(path));
                }
                "--compare" => {
                    i += 1;
                    let Some(path) = args.get(i) else {
                        return Err("--compare requires a path".into());
                    };
                    cfg.compare = Some(PathBuf::from(path));
                }
                "--threshold" => {
                    i += 1;
                    cfg.threshold_pct = parse_f64(args.get(i), "--threshold")?;
                }
                "--inventory-only" => cfg.inventory_only = true,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown argument `{other}`")),
            }
            i += 1;
        }
        if cfg.warmup == 0 || cfg.iters == 0 || cfg.n_layers == 0 {
            return Err("--warmup, --iters, and --layers must be non-zero".into());
        }
        if !cfg.threshold_pct.is_finite() || cfg.threshold_pct < 0.0 {
            return Err("--threshold must be a non-negative percentage".into());
        }
        Ok(cfg)
    }
}

pub fn usage() -> String {
    "Usage: cargo run --release --features gpu -p larql-compute --example diag_shader_bench -- [--profile smoke|gemma3] [--warmup N] [--iters N] [--layers N] [--inventory-only] [--json PATH] [--compare PATH] [--threshold PCT]".into()
}

pub(crate) fn parse_usize(value: Option<&String>, flag: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<usize>()
        .map_err(|_| format!("{flag} requires a positive integer"))
}

pub(crate) fn parse_f64(value: Option<&String>, flag: &str) -> Result<f64, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<f64>()
        .map_err(|_| format!("{flag} requires a number"))
}
