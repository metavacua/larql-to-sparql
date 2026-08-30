//! Tests for [`super`] — the env-gated decode diagnostics.
//!
//! Every function here is a no-op unless its variable names a
//! destination, which makes the *silence* the thing worth testing: a
//! dump that quietly wrote nothing, or wrote to the wrong path, looks
//! exactly like a dump that was switched off. So each case asserts both
//! directions — nothing written when the variable is unset, and the
//! specific files with the specific byte counts when it is set.
//!
//! Byte counts are asserted rather than just existence because these
//! files are read back as raw little-endian f32 by an out-of-tree
//! comparison script; a file of the wrong length is a silent
//! misalignment at the other end, not a loud failure.

use larql_compute::options::{ENV_KV_CACHE_DUMP_DIR, ENV_PERCALL_LAYER_DUMP_DIR};

use super::{dump_kv_caches, dump_percall_layers};
use crate::ops::kv_cache::KVCache;
use crate::MetalBackend;

/// Serialises the env mutation these tests do; `--test-threads=1` is the
/// crate's coverage default but the lock keeps them honest under a normal
/// parallel `cargo test` too.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env<T>(vars: &[(&'static str, Option<&str>)], f: impl FnOnce() -> T) -> T {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev: Vec<_> = vars
        .iter()
        .map(|(n, _)| (*n, std::env::var_os(n)))
        .collect();
    for (n, v) in vars {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    let out = f();
    for (n, v) in prev {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    out
}

const LAYERS: usize = 3;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const HIDDEN: usize = 16;
/// Elements per layer once `current_len` positions are populated.
const fn kv_elems(len: usize) -> usize {
    len * KV_HEADS * HEAD_DIM
}

/// A cache with `len` populated positions on every layer. `current_len` is
/// what the dumps read, so it is set directly rather than by appending —
/// the dumps' contract is about lengths, not about how they got there.
fn cache_with(metal: &MetalBackend, len: usize) -> KVCache {
    let mut kv = KVCache::new(&metal.bufs, LAYERS, 32, KV_HEADS, HEAD_DIM);
    for l in kv.layers.iter_mut() {
        l.current_len = len;
    }
    kv
}

fn count_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
}

#[test]
fn percall_dump_writes_nothing_when_unset() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let kv = cache_with(&metal, 4);
    let h = metal.bufs.output((HIDDEN * 4) as u64);
    let x = vec![0.5f32; HIDDEN];

    with_env(&[(ENV_PERCALL_LAYER_DUMP_DIR, None)], || {
        dump_percall_layers(&kv, &h, &x, HIDDEN, 7);
    });
    assert_eq!(
        count_files(tmp.path()),
        0,
        "an unset destination must write nothing at all"
    );
}

#[test]
fn percall_dump_writes_one_k_file_per_populated_layer_plus_hidden_and_input() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_str().unwrap().to_string();
    let len = 4;
    let kv = cache_with(&metal, len);
    let h = metal.bufs.output((HIDDEN * 4) as u64);
    let x = vec![0.5f32; HIDDEN];

    with_env(&[(ENV_PERCALL_LAYER_DUMP_DIR, Some(&dir))], || {
        dump_percall_layers(&kv, &h, &x, HIDDEN, 7);
    });

    for l in 0..LAYERS {
        let p = tmp.path().join(format!("metal_call007_L{l:02}_K.f32"));
        let meta = std::fs::metadata(&p).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        assert_eq!(
            meta.len() as usize,
            kv_elems(len) * 4,
            "layer {l}'s K dump must cover current_len x num_kv_heads x head_dim"
        );
    }
    let hidden_file = tmp.path().join("metal_call007_h_final.f32");
    assert_eq!(
        std::fs::metadata(&hidden_file).unwrap().len() as usize,
        HIDDEN * 4
    );
    let x_file = tmp.path().join("metal_call007_x_input.f32");
    assert_eq!(
        std::fs::metadata(&x_file).unwrap().len() as usize,
        HIDDEN * 4
    );
    assert_eq!(count_files(tmp.path()), LAYERS + 2);
}

/// The call index is in the filename precisely so successive calls do not
/// overwrite each other — that is the whole point of this dump versus the
/// K/V one below.
#[test]
fn percall_dump_keys_files_by_call_index() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_str().unwrap().to_string();
    let kv = cache_with(&metal, 2);
    let h = metal.bufs.output((HIDDEN * 4) as u64);
    let x = vec![0.25f32; HIDDEN];

    with_env(&[(ENV_PERCALL_LAYER_DUMP_DIR, Some(&dir))], || {
        dump_percall_layers(&kv, &h, &x, HIDDEN, 0);
        dump_percall_layers(&kv, &h, &x, HIDDEN, 1);
    });
    assert!(tmp.path().join("metal_call000_h_final.f32").exists());
    assert!(tmp.path().join("metal_call001_h_final.f32").exists());
}

/// A layer that has never been written has nothing to dump, and skipping
/// it must not shift the other layers' filenames.
#[test]
fn percall_dump_skips_empty_layers_without_renumbering() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_str().unwrap().to_string();
    let mut kv = cache_with(&metal, 4);
    kv.layers[1].current_len = 0;
    let h = metal.bufs.output((HIDDEN * 4) as u64);
    let x = vec![0.5f32; HIDDEN];

    with_env(&[(ENV_PERCALL_LAYER_DUMP_DIR, Some(&dir))], || {
        dump_percall_layers(&kv, &h, &x, HIDDEN, 3);
    });
    assert!(tmp.path().join("metal_call003_L00_K.f32").exists());
    assert!(
        !tmp.path().join("metal_call003_L01_K.f32").exists(),
        "an empty layer is skipped"
    );
    assert!(
        tmp.path().join("metal_call003_L02_K.f32").exists(),
        "layer 2 keeps its own index — skipping must not renumber"
    );
}

#[test]
fn kv_cache_dump_writes_nothing_when_unset() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let kv = cache_with(&metal, 3);
    with_env(&[(ENV_KV_CACHE_DUMP_DIR, None)], || {
        dump_kv_caches(&kv);
    });
    assert_eq!(count_files(tmp.path()), 0);
}

#[test]
fn kv_cache_dump_writes_a_k_and_v_file_per_populated_layer() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_str().unwrap().to_string();
    let len = 5;
    let mut kv = cache_with(&metal, len);
    kv.layers[2].current_len = 0;

    with_env(&[(ENV_KV_CACHE_DUMP_DIR, Some(&dir))], || {
        dump_kv_caches(&kv);
    });

    for l in 0..2 {
        for which in ["K", "V"] {
            let p = tmp.path().join(format!("metal_L{l:02}_{which}_cache.f32"));
            let meta = std::fs::metadata(&p).unwrap_or_else(|e| panic!("{p:?}: {e}"));
            assert_eq!(meta.len() as usize, kv_elems(len) * 4);
        }
    }
    assert_eq!(
        count_files(tmp.path()),
        4,
        "two populated layers x (K, V); the empty layer contributes nothing"
    );
}

/// These filenames carry no call index, so a later call overwrites an
/// earlier one — "last call wins" is the documented behaviour and the
/// reason the per-call dump above exists separately.
#[test]
fn kv_cache_dump_overwrites_rather_than_accumulating() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_str().unwrap().to_string();
    let kv_short = cache_with(&metal, 2);
    let kv_long = cache_with(&metal, 6);

    with_env(&[(ENV_KV_CACHE_DUMP_DIR, Some(&dir))], || {
        dump_kv_caches(&kv_short);
        dump_kv_caches(&kv_long);
    });
    let p = tmp.path().join("metal_L00_K_cache.f32");
    assert_eq!(
        std::fs::metadata(&p).unwrap().len() as usize,
        kv_elems(6) * 4,
        "the second call's longer cache must have replaced the first"
    );
    assert_eq!(count_files(tmp.path()), LAYERS * 2);
}

/// An unwritable destination is a diagnostic failure, not a decode
/// failure: it reports and returns rather than unwinding through a live
/// command buffer.
#[test]
fn a_bad_destination_is_reported_not_fatal() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let kv = cache_with(&metal, 2);
    let h = metal.bufs.output((HIDDEN * 4) as u64);
    let x = vec![1.0f32; HIDDEN];
    let missing = "/nonexistent-larql-diag-dir";

    with_env(&[(ENV_PERCALL_LAYER_DUMP_DIR, Some(missing))], || {
        dump_percall_layers(&kv, &h, &x, HIDDEN, 0);
    });
    with_env(&[(ENV_KV_CACHE_DUMP_DIR, Some(missing))], || {
        dump_kv_caches(&kv);
    });
}
