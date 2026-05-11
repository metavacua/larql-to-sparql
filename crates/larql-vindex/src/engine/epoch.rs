use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct Epoch(AtomicU64);

impl Epoch {
    pub fn zero() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn value(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn advance(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed) + 1
    }
}

impl Default for Epoch {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_starts_at_zero() {
        let e = Epoch::zero();
        assert_eq!(e.value(), 0);
    }

    #[test]
    fn epoch_advances() {
        let e = Epoch::zero();
        assert_eq!(e.advance(), 1);
        assert_eq!(e.advance(), 2);
        assert_eq!(e.value(), 2);
    }
}
