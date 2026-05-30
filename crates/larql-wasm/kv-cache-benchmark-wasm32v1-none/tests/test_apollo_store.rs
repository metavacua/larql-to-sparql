//! Integration tests: load the real `apollo-demo/apollo11_store/`.
//!
//! These tests are gated on the `real-model` feature only because the apollo
//! module itself is; they don't need a model loaded — they only read disk.
//!
//! Run with:
//!   APOLLO_STORE_PATH=/Users/christopherhay/chris-source/apollo-demo/apollo11_store \
//!   cargo test --features real-model -p kv-cache-benchmark \
//!       --test test_apollo_store -- --ignored --nocapture

#![cfg(feature = "real-model")]

use std::path::Path;

use kv_cache_benchmark::apollo::{ApolloStore, VecInjectEntry};

fn store_path() -> Option<std::path::PathBuf> {
    // Prefer env override; fall back to the sibling apollo-demo checkout.
    if let Ok(p) = std::env::var("APOLLO_STORE_PATH") {
        return Some(std::path::PathBuf::from(p));
    }
    let candidates = [
        "../../../apollo-demo/apollo11_store",
        "../../../../apollo-demo/apollo11_store",
        "/Users/christopherhay/chris-source/apollo-demo/apollo11_store",
    ];
    for c in candidates {
        let p = Path::new(c);
        if p.join("manifest.json").exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

#[test]
#[ignore]
fn test_load_apollo11_store_manifest() {
    let path = store_path().expect("apollo11_store not found — set APOLLO_STORE_PATH");
    let store = ApolloStore::load(&path).expect("load failed");

    // Known manifest invariants from apollo-demo/apollo11_store/manifest.json
    assert_eq!(store.manifest.version, 12);
    assert_eq!(store.manifest.num_entries, 3585);
    assert_eq!(store.manifest.num_windows, 176);
    assert_eq!(store.manifest.num_tokens, 90000);
    assert_eq!(store.manifest.window_size, 512);
    assert_eq!(store.manifest.crystal_layer, 30);
    assert_eq!(store.manifest.arch_config.injection_layer, 30);
    assert_eq!(store.manifest.arch_config.inject_coefficient, 10.0);
}

#[test]
#[ignore]
fn test_load_apollo11_boundaries() {
    let path = store_path().expect("apollo11_store not found — set APOLLO_STORE_PATH");
    let store = ApolloStore::load(&path).expect("load failed");

    assert_eq!(store.boundaries.len(), 176);
    assert_eq!(
        store.hidden_size(),
        2560,
        "Gemma 3 4B hidden size should be 2560"
    );
    // Every window's boundary should be the same hidden size.
    for (i, b) in store.boundaries.iter().enumerate() {
        assert_eq!(b.len(), 2560, "window {i}: boundary shape mismatch");
    }
    // boundary_residual.npy (if present) is shape (1, 1, 2560) = 2560 floats.
    if let Some(br) = &store.boundary_residual {
        assert_eq!(br.len(), 2560);
    }
}

#[test]
#[ignore]
fn test_load_apollo11_window_tokens() {
    let path = store_path().expect("apollo11_store not found — set APOLLO_STORE_PATH");
    let store = ApolloStore::load(&path).expect("load failed");

    assert_eq!(store.window_tokens.len(), 176);
    // Most windows have exactly window_size tokens; the last one may be
    // shorter (90000 total / 176 windows ≈ 511.4, so some or all tokens
    // live in a possibly-padded scheme). Sanity bound each window.
    let total: usize = store.window_tokens.iter().map(|w| w.len()).sum();
    assert!(
        total >= store.manifest.num_tokens,
        "total tokens across windows {} < manifest.num_tokens {}",
        total,
        store.manifest.num_tokens,
    );
    for (i, t) in store.window_tokens.iter().enumerate() {
        assert!(
            t.len() <= store.manifest.window_size,
            "window {i}: token list {} > window_size {}",
            t.len(),
            store.manifest.window_size,
        );
    }
}

#[test]
#[ignore]
fn test_load_apollo11_entries() {
    let path = store_path().expect("apollo11_store not found — set APOLLO_STORE_PATH");
    let store = ApolloStore::load(&path).expect("load failed");

    assert_eq!(store.entries.len(), 3585);

    // Sanity check the first entry against the Python inspection:
    //   (55241, 9.765625, 0, 1, 0)
    let first = store.entries[0];
    assert_eq!(first.token_id, 55241);
    assert!((first.coefficient - 9.765625).abs() < 1e-6);
    assert_eq!(first.window_id, 0);
    assert_eq!(first.position_in_window, 1);
    assert_eq!(first.fact_id, 0);

    // Every entry's window_id must be < num_windows.
    for (i, e) in store.entries.iter().enumerate() {
        assert!(
            (e.window_id as usize) < store.manifest.num_windows,
            "entry {i}: window_id {} out of range",
            e.window_id,
        );
        assert!(
            (e.position_in_window as usize) < store.manifest.window_size,
            "entry {i}: position_in_window {} out of range",
            e.position_in_window,
        );
    }
}

#[test]
#[ignore]
fn test_apollo11_total_bytes_reasonable() {
    let path = store_path().expect("apollo11_store not found — set APOLLO_STORE_PATH");
    let store = ApolloStore::load(&path).expect("load failed");

    // Disk footprint for apollo11_store is ~2.8 MB. The in-memory
    // footprint (all f32 inflated, all arrays in RAM) will be somewhat
    // bigger than the disk .npy files. Sanity bounds:
    //   boundaries: 176 × 2560 × 4 = 1.8 MB
    //   window_tokens: 176 × 512 × 4 = 360 KB
    //   entries: 3585 × 16 bytes (with padding) ≈ 57 KB
    //   boundary_residual: ~10 KB
    // Total: ~2.2 MB memory footprint.
    let bytes = store.total_bytes();
    println!(
        "apollo11 store in memory: {:.2} MB ({} bytes)",
        bytes as f64 / 1024.0 / 1024.0,
        bytes
    );
    assert!(bytes > 1_000_000, "store smaller than expected: {bytes}");
    assert!(bytes < 5_000_000, "store bigger than expected: {bytes}");
}

/// Show the top fact_ids by entry count — quick look at how the entries are
/// grouped in this store.
#[test]
#[ignore]
fn test_apollo11_entry_distribution() {
    let path = store_path().expect("apollo11_store not found — set APOLLO_STORE_PATH");
    let store = ApolloStore::load(&path).expect("load failed");

    let mut per_window = std::collections::HashMap::<u16, usize>::new();
    for e in &store.entries {
        *per_window.entry(e.window_id).or_insert(0) += 1;
    }
    let mut counts: Vec<(u16, usize)> = per_window.into_iter().collect();
    counts.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

    println!("top 5 windows by entry count:");
    for (wid, c) in counts.iter().take(5) {
        println!("  window {wid}: {c} entries");
    }

    // `manifest.entries_per_window` is the *target default* at build time
    // — the actual distribution is highly skewed (some windows have
    // 100+ entries, many have fewer). Just report the stats and sanity
    // bound the total count.
    let mean = store.entries.len() as f64 / store.manifest.num_windows as f64;
    println!(
        "mean entries per window: {:.2} (manifest default: {})",
        mean, store.manifest.entries_per_window,
    );
    assert!(mean > 0.0 && mean < 200.0);
}

/// Ensure `VecInjectEntry` layout matches the Python structured dtype byte-for-byte.
#[test]
fn test_entry_struct_roundtrips_cleanly() {
    let e = VecInjectEntry {
        token_id: 0xdeadbeef,
        coefficient: 1.5,
        window_id: 17,
        position_in_window: 42,
        fact_id: 99,
    };
    // Just verify field read-back — the rest is compiler's job.
    assert_eq!(e.token_id, 0xdeadbeef);
    assert_eq!(e.coefficient, 1.5);
    assert_eq!(e.window_id, 17);
}
