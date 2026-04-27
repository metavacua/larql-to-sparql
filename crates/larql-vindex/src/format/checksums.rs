//! Checksum utilities for vindex file integrity verification.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::VindexError;

/// Compute SHA256 checksum of a file. Returns hex string.
pub fn sha256_file(path: &Path) -> Result<String, VindexError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20]; // 1 MB buffer
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute checksums for all binary files in a vindex directory.
/// Returns a map of filename → SHA256 hex string.
pub fn compute_checksums(dir: &Path) -> Result<HashMap<String, String>, VindexError> {
    let mut checksums = HashMap::new();

    let files = [
        "gate_vectors.bin",
        "embeddings.bin",
        "down_meta.bin",
        "down_meta.jsonl",
        "attn_weights.bin",
        "up_weights.bin",
        "down_weights.bin",
        "norms.bin",
        "lm_head.bin",
    ];

    for filename in &files {
        let path = dir.join(filename);
        if path.exists() {
            checksums.insert(filename.to_string(), sha256_file(&path)?);
        }
    }

    Ok(checksums)
}

/// Verify checksums of a vindex directory against stored checksums.
/// Returns a list of (filename, status) pairs.
pub fn verify_checksums(
    dir: &Path,
    stored: &HashMap<String, String>,
) -> Result<Vec<(String, bool)>, VindexError> {
    let mut results = Vec::new();

    for (filename, expected) in stored {
        let path = dir.join(filename);
        if path.exists() {
            let actual = sha256_file(&path)?;
            results.push((filename.clone(), actual == *expected));
        } else {
            results.push((filename.clone(), false));
        }
    }

    Ok(results)
}
