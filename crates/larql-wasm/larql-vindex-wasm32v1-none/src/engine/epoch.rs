#[cfg(not(target_arch = "wasm32"))]
use portable_atomic::AtomicU64;
use core::sync::atomic::Ordering;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct Epoch(AtomicU64);

#[cfg(not(target_arch = "wasm32"))]
impl Epoch {
    #[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
