//! Minimal lowercase hex encoding — avoids pulling in the `hex` crate
//! for the handful of digest-formatting call sites this crate needs
//! (`build_id` today; shard `sha256` in the MANIFEST stage later).

use std::fmt::Write;

/// Lowercase-hex-encode `bytes`.
pub fn encode(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(out, "{b:02x}").expect("writing to a String never fails");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_known_bytes() {
        assert_eq!(encode([0x00, 0xff, 0x0a]), "00ff0a");
    }

    #[test]
    fn encodes_empty_input() {
        assert_eq!(encode([]), "");
    }

    #[test]
    fn sha256_digest_encodes_to_two_hex_chars_per_byte() {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(b"larql");
        assert_eq!(encode(digest).len(), Sha256::output_size() * 2);
    }
}
