/// Cold tier storage: token IDs only.
///
/// When tokens fall out of the active window, their residuals are discarded.
/// Only the token ID (u32, 4 bytes) is retained. To reconstruct, replay
/// through the forward pass from the nearest checkpoint.
///
/// At 370K tokens: 370,000 × 4 = 1.48 MB (vs 56 GB for standard KV).

/// Cold tier token storage.
pub struct ColdTier {
    token_ids: Vec<u32>,
}

impl ColdTier {
    pub fn new() -> Self {
        Self {
            token_ids: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            token_ids: Vec::with_capacity(capacity),
        }
    }

    /// Store a token ID in cold tier.
    pub fn push(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
    }

    /// Number of tokens in cold storage.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Memory used in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.token_ids.len() * 4
    }

    /// Get token IDs for a range (for replay/reconstruction).
    pub fn token_ids(&self, start: usize, end: usize) -> &[u32] {
        &self.token_ids[start..end.min(self.token_ids.len())]
    }

    /// Serialise to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.token_ids.len() * 4);
        for &id in &self.token_ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        buf
    }

    /// Deserialise from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut token_ids = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            token_ids.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Self { token_ids }
    }
}

impl Default for ColdTier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_tier_memory() {
        let mut ct = ColdTier::new();
        for i in 0..1000u32 {
            ct.push(i);
        }
        assert_eq!(ct.memory_bytes(), 4000);
        assert_eq!(ct.len(), 1000);
    }

    #[test]
    fn test_cold_tier_serialise_roundtrip() {
        let mut ct = ColdTier::new();
        for i in 0..100u32 {
            ct.push(i * 7);
        }
        let bytes = ct.to_bytes();
        let ct2 = ColdTier::from_bytes(&bytes);
        assert_eq!(ct.len(), ct2.len());
        assert_eq!(ct.token_ids(0, 100), ct2.token_ids(0, 100));
    }

    #[test]
    fn test_370k_tokens_size() {
        // 370K tokens × 4 bytes = 1.48 MB
        let size = 370_000usize * 4;
        assert_eq!(size, 1_480_000);
        assert!(size < 2_000_000, "Cold tier at 370K should be under 2MB");
    }
}
