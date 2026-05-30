/// FNV-1a 64-bit hash over arbitrary bytes.
/// Deterministic across Rust versions — suitable for codebook drift detection.
fn fnv64(data: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h = OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Stable fingerprint of a PQ codebook: FNV-1a over all centroid bytes in
/// deterministic order (group-major, code-minor, dimension-minor).
/// Returns a 16-char lowercase hex string.
pub fn codebook_fingerprint(centroids: &[Vec<Vec<f64>>]) -> String {
    let mut buf = Vec::with_capacity(
        centroids.len()
            * centroids
                .first()
                .and_then(|g| g.first())
                .map(|c| c.len())
                .unwrap_or(0)
            * 8,
    );
    for group in centroids {
        for centroid in group {
            for &v in centroid {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    format!("{:016x}", fnv64(&buf))
}
