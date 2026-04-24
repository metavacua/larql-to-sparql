//! Per-window token-ID archive (COLD tier).
//!
//! Append-only; never evicted. Provides the raw token stream for replay.
//! Four bytes per token (u32), regardless of model size.

use std::collections::HashMap;

#[derive(Default)]
pub struct TokenArchive {
    tokens: HashMap<usize, Vec<u32>>,
    abs_offsets: HashMap<usize, usize>,
}

impl TokenArchive {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn archive(&mut self, window_id: usize, token_ids: Vec<u32>, abs_offset: usize) {
        self.tokens.insert(window_id, token_ids);
        self.abs_offsets.insert(window_id, abs_offset);
    }

    /// Return `(token_ids, abs_offset)` for a window. Offset is the absolute
    /// position of the first token in this window within the full document.
    pub fn retrieve(&self, window_id: usize) -> Option<(&[u32], usize)> {
        let toks = self.tokens.get(&window_id)?;
        let off = *self.abs_offsets.get(&window_id)?;
        Some((toks.as_slice(), off))
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn total_tokens(&self) -> usize {
        self.tokens.values().map(|t| t.len()).sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.tokens.values().map(|t| t.len() * 4).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_and_retrieve_roundtrip() {
        let mut archive = TokenArchive::new();
        archive.archive(0, vec![1, 2, 3, 4, 5], 0);
        archive.archive(1, vec![6, 7, 8], 5);

        let (t0, o0) = archive.retrieve(0).unwrap();
        assert_eq!(t0, &[1, 2, 3, 4, 5]);
        assert_eq!(o0, 0);

        let (t1, o1) = archive.retrieve(1).unwrap();
        assert_eq!(t1, &[6, 7, 8]);
        assert_eq!(o1, 5);
    }

    #[test]
    fn total_accounting() {
        let mut archive = TokenArchive::new();
        archive.archive(0, vec![0; 512], 0);
        archive.archive(1, vec![0; 512], 512);
        assert_eq!(archive.total_tokens(), 1024);
        assert_eq!(archive.total_bytes(), 1024 * 4);
    }

    #[test]
    fn retrieve_missing_returns_none() {
        let archive = TokenArchive::new();
        assert!(archive.retrieve(42).is_none());
    }
}
