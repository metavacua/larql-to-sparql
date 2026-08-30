//! Tests for [`super`].
//!
//! Split out of `shader_bench.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use crate::MetalBackend;

use super::*;

fn args(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn compare_json_parser_reads_batched_ms() {
    let path = std::env::temp_dir().join(format!(
        "larql-shader-bench-compare-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &path,
        r#"[
  {"name":"q4k_matvec","family":"q4k-matvec","batched_ms":0.025000,"batched_gbs":147.7},
  {"name":"f16_gemv","family":"lm-head","batched_ms":null}
]"#,
    )
    .unwrap();

    let parsed = load_baseline(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let q4k = parsed.get("q4k_matvec").unwrap();
    assert_eq!(q4k.family, "q4k-matvec");
    assert_eq!(q4k.batched_ms, Some(0.025));
    assert_eq!(parsed.get("f16_gemv").unwrap().batched_ms, None);
}

#[test]
fn config_default_uses_smoke_profile_with_baseline_iters() {
    let c = Config::default();
    assert_eq!(c.profile, Profile::Smoke);
    assert_eq!(c.warmup, 2);
    assert_eq!(c.iters, 8);
    assert_eq!(c.n_layers, 4);
    assert!(!c.inventory_only);
    assert!(c.json.is_none());
    assert!(c.compare.is_none());
}

#[test]
fn config_from_args_parses_smoke_profile_flag() {
    let cfg = Config::from_args(&args(&["--profile", "smoke"])).unwrap();
    assert_eq!(cfg.profile, Profile::Smoke);
    assert_eq!(cfg.warmup, 2);
    assert_eq!(cfg.iters, 8);
    assert_eq!(cfg.n_layers, 4);
}

#[test]
fn config_from_args_parses_gemma3_profile_flag() {
    let cfg = Config::from_args(&args(&["--profile", "gemma3"])).unwrap();
    assert_eq!(cfg.profile, Profile::Gemma3);
    assert_eq!(cfg.warmup, 5);
    assert_eq!(cfg.iters, 30);
    assert_eq!(cfg.n_layers, 34);
}

#[test]
fn config_from_args_parses_numeric_flags() {
    let cfg = Config::from_args(&args(&[
        "--warmup",
        "7",
        "--iters",
        "11",
        "--layers",
        "3",
        "--threshold",
        "12.5",
    ]))
    .unwrap();
    assert_eq!(cfg.warmup, 7);
    assert_eq!(cfg.iters, 11);
    assert_eq!(cfg.n_layers, 3);
    assert!((cfg.threshold_pct - 12.5).abs() < f64::EPSILON);
}

#[test]
fn config_from_args_parses_path_flags_and_inventory_only() {
    let cfg = Config::from_args(&args(&[
        "--json",
        "/tmp/x.json",
        "--compare",
        "/tmp/baseline.json",
        "--inventory-only",
    ]))
    .unwrap();
    assert_eq!(
        cfg.json.as_deref().unwrap().to_string_lossy(),
        "/tmp/x.json"
    );
    assert_eq!(
        cfg.compare.as_deref().unwrap().to_string_lossy(),
        "/tmp/baseline.json"
    );
    assert!(cfg.inventory_only);
}

#[test]
fn config_from_args_unknown_arg_errors() {
    assert!(Config::from_args(&args(&["--bogus"])).is_err());
}

#[test]
fn config_from_args_help_returns_usage_as_err() {
    let err = Config::from_args(&args(&["--help"])).unwrap_err();
    assert!(err.contains("diag_shader_bench"));
}

#[test]
fn config_from_args_rejects_unknown_profile_value() {
    assert!(Config::from_args(&args(&["--profile", "oversized"])).is_err());
}

#[test]
fn config_from_args_rejects_missing_flag_values() {
    assert!(Config::from_args(&args(&["--profile"])).is_err());
    assert!(Config::from_args(&args(&["--warmup"])).is_err());
    assert!(Config::from_args(&args(&["--iters"])).is_err());
    assert!(Config::from_args(&args(&["--layers"])).is_err());
    assert!(Config::from_args(&args(&["--json"])).is_err());
    assert!(Config::from_args(&args(&["--compare"])).is_err());
    assert!(Config::from_args(&args(&["--threshold"])).is_err());
}

#[test]
fn config_from_args_rejects_zero_warmup_iters_or_layers() {
    assert!(Config::from_args(&args(&["--warmup", "0"])).is_err());
    assert!(Config::from_args(&args(&["--iters", "0"])).is_err());
    assert!(Config::from_args(&args(&["--layers", "0"])).is_err());
}

#[test]
fn config_from_args_rejects_negative_threshold() {
    assert!(Config::from_args(&args(&["--threshold", "-1.0"])).is_err());
    assert!(Config::from_args(&args(&["--threshold", "nan"])).is_err());
}

#[test]
fn config_from_args_rejects_non_numeric_warmup_iters_layers() {
    assert!(Config::from_args(&args(&["--warmup", "abc"])).is_err());
    assert!(Config::from_args(&args(&["--iters", "x"])).is_err());
    assert!(Config::from_args(&args(&["--layers", "y"])).is_err());
    assert!(Config::from_args(&args(&["--threshold", "z"])).is_err());
}

#[test]
fn usage_string_mentions_the_example_entry_point() {
    let u = usage();
    assert!(u.contains("diag_shader_bench"));
}

#[test]
fn shape_for_smoke_and_gemma3_have_distinct_dimensions() {
    let smoke = Shape::for_profile(Profile::Smoke);
    let gemma3 = Shape::for_profile(Profile::Gemma3);
    assert_eq!(smoke.label, "smoke");
    assert_eq!(gemma3.label, "gemma3-4b");
    assert!(gemma3.hidden > smoke.hidden);
    assert!(gemma3.inter > smoke.inter);
}

#[test]
fn inventory_results_include_or_exclude_benched_entries() {
    let with_bench = inventory_results(true);
    let no_bench = inventory_results(false);
    assert!(!with_bench.is_empty());
    assert!(no_bench.len() < with_bench.len());
    assert!(no_bench.iter().all(|r| r.status != "bench"));
}

/// Smoke-run the full bench loop at the small `Smoke` profile
/// with warmup=1 iters=1 layers=1. Drives every `bench_*` helper,
/// `print_results`, and the `--compare` baseline pipeline so the
/// bench-dispatch code path isn't dead from the unit-test
/// perspective. Skips silently if no Metal device is available.
#[test]
fn run_smoke_profile_drives_bench_dispatchers_end_to_end() {
    if MetalBackend::new().is_none() {
        return;
    }
    let json_path = std::env::temp_dir().join(format!(
        "larql-shader-bench-smoke-json-{}.json",
        std::process::id()
    ));
    let compare_path = std::env::temp_dir().join(format!(
        "larql-shader-bench-smoke-cmp-{}.json",
        std::process::id()
    ));
    let cfg = Config {
        profile: Profile::Smoke,
        warmup: 1,
        iters: 1,
        n_layers: 1,
        json: Some(json_path.clone()),
        compare: None,
        threshold_pct: 5.0,
        inventory_only: false,
    };
    let results = run(&cfg).expect("smoke run should succeed on Metal-capable host");
    assert!(results.iter().any(|r| r.status == "bench"));
    // Use the JSON we just wrote as the compare baseline so the
    // `print_compare` branch runs end-to-end on real data.
    std::fs::copy(&json_path, &compare_path).unwrap();
    let cfg_cmp = Config {
        profile: Profile::Smoke,
        warmup: 1,
        iters: 1,
        n_layers: 1,
        json: None,
        compare: Some(compare_path.clone()),
        threshold_pct: 5.0,
        inventory_only: false,
    };
    let _ = run(&cfg_cmp).expect("compare run should succeed");
    std::fs::remove_file(&json_path).ok();
    std::fs::remove_file(&compare_path).ok();
}

#[test]
fn run_with_inventory_only_writes_json_and_returns_results() {
    let path = std::env::temp_dir().join(format!(
        "larql-shader-bench-inv-{}.json",
        std::process::id()
    ));
    let cfg = Config {
        inventory_only: true,
        json: Some(path.clone()),
        ..Config::default()
    };
    let results = run(&cfg).expect("inventory-only run should succeed");
    assert!(!results.is_empty());
    let json = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(json.starts_with('['));
    assert!(json.contains("\"name\":"));
}

#[test]
fn to_json_handles_optional_fields_and_escapes_quotes() {
    let r = BenchResult {
        name: "k\"k",
        family: "fam",
        status: "bench",
        shape: "x".into(),
        rows_per_tg: Some(4),
        threads_per_tg: Some(64),
        bytes_per_call: 1024,
        isolated_ms: Some(1.25),
        isolated_sd_ms: None,
        batched_ms: Some(0.5),
        batched_gbs: None,
        output_nonzero: Some(7),
        sanity: "ok",
        note: "",
    };
    let s = to_json(&[r]);
    assert!(s.contains(r#""name":"k\"k""#));
    assert!(s.contains("\"isolated_sd_ms\":null"));
    assert!(s.contains("\"batched_gbs\":null"));
    assert!(s.contains("\"isolated_ms\":1.250000"));
    assert!(s.contains("\"bytes_per_call\":1024"));
}

#[test]
fn opt_helpers_round_trip_some_and_null_for_none() {
    assert_eq!(opt_u64(Some(42)), "42");
    assert_eq!(opt_u64(None), "null");
    assert_eq!(opt_usize(Some(7)), "7");
    assert_eq!(opt_usize(None), "null");
    assert_eq!(opt_f64(Some(1.5)), "1.500000");
    assert_eq!(opt_f64(None), "null");
}

#[test]
fn json_escape_handles_quotes_and_backslashes() {
    assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
}

#[test]
fn json_field_helpers_handle_missing_and_null() {
    assert_eq!(json_field_string("{\"x\":\"y\"}", "x"), Some("y".into()));
    assert_eq!(json_field_string("{}", "missing"), None);
    assert_eq!(json_field_number("{\"v\":1.25}", "v"), Some(1.25));
    assert_eq!(json_field_number("{\"v\":null}", "v"), None);
    assert_eq!(json_field_number("{}", "v"), None);
}

#[test]
fn load_baseline_errors_on_empty_array() {
    let path = std::env::temp_dir().join(format!(
        "larql-shader-bench-empty-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, "[]").unwrap();
    let err = load_baseline(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(err.contains("did not contain"));
}

#[test]
fn load_baseline_errors_when_path_missing() {
    let p = std::env::temp_dir().join(format!(
        "larql-shader-bench-missing-{}.json",
        std::process::id()
    ));
    assert!(load_baseline(&p).is_err());
}

#[test]
fn print_compare_summarises_regression_vs_baseline() {
    // Smoke test: builds a fake baseline + current and walks the
    // verdict branches (improved / flat / regressed / missing).
    let path = std::env::temp_dir().join(format!(
        "larql-shader-bench-cmp-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &path,
        r#"[
  {"name":"a","family":"fam","batched_ms":1.000000},
  {"name":"b","family":"fam","batched_ms":1.000000},
  {"name":"c","family":"fam","batched_ms":1.000000},
  {"name":"e","family":"fam","batched_ms":null},
  {"name":"f","family":"fam","batched_ms":0.000000}
]"#,
    )
    .unwrap();
    let baseline = load_baseline(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let mk = |name: &'static str, ms: Option<f64>| BenchResult {
        name,
        family: "fam",
        status: "bench",
        shape: String::new(),
        rows_per_tg: None,
        threads_per_tg: None,
        bytes_per_call: 0,
        isolated_ms: None,
        isolated_sd_ms: None,
        batched_ms: ms,
        batched_gbs: None,
        output_nonzero: None,
        sanity: "ok",
        note: "",
    };
    let current = vec![
        mk("a", Some(0.5)), // improved
        mk("b", Some(1.0)), // flat
        mk("c", Some(1.5)), // regressed
        mk("d", Some(1.0)), // missing from baseline
        mk("e", Some(1.0)), // baseline.batched_ms = None → missing
        mk("f", Some(1.0)), // baseline.batched_ms = 0.0 → missing
        mk("g", None),      // skipped, no cur_ms
    ];
    print_compare(
        &current,
        &baseline,
        std::path::Path::new("baseline.json"),
        5.0,
    );
}
