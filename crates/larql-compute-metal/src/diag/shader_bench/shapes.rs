//! The shape a bench runs at, and the result rows it produces.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) label: &'static str,
    pub(crate) hidden: usize,
    pub(crate) inter: usize,
    pub(crate) q_rows: usize,
    pub(crate) kv_rows: usize,
    pub(crate) lm_rows: usize,
}

impl Shape {
    pub(crate) fn for_profile(profile: Profile) -> Self {
        match profile {
            Profile::Smoke => Self {
                label: "smoke",
                hidden: 512,
                inter: 2048,
                q_rows: 1024,
                kv_rows: 512,
                lm_rows: 8192,
            },
            Profile::Gemma3 => Self {
                label: "gemma3-4b",
                hidden: 2560,
                inter: 10240,
                q_rows: 8192,
                kv_rows: GEMMA3_4B_KV_ROWS,
                // Full Gemma 3 vocab would allocate ~2.7GB for f32
                // lm_head input alone. Keep shader bench usable by
                // capping the synthetic f32/f16 gemv case while other
                // kernels use production layer shapes.
                lm_rows: 32768,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: &'static str,
    pub family: &'static str,
    pub status: &'static str,
    pub shape: String,
    pub rows_per_tg: Option<u64>,
    pub threads_per_tg: Option<u64>,
    pub bytes_per_call: u64,
    pub isolated_ms: Option<f64>,
    pub isolated_sd_ms: Option<f64>,
    pub batched_ms: Option<f64>,
    pub batched_gbs: Option<f64>,
    pub output_nonzero: Option<usize>,
    pub sanity: &'static str,
    pub note: &'static str,
}

pub(crate) struct InventoryItem {
    pub(crate) name: &'static str,
    pub(crate) family: &'static str,
    pub(crate) status: &'static str,
    pub(crate) note: &'static str,
}
