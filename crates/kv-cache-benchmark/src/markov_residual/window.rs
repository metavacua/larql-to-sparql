/// Sliding window over residual vectors.
///
/// Maintains the most recent `capacity` residuals. Older residuals
/// are evicted to cold tier (token IDs only).

pub struct ResidualWindow {
    pub capacity: usize,
    pub dim: usize,
    /// Ring buffer of residual vectors.
    buffer: Vec<Vec<f32>>,
    /// Write position in ring buffer.
    write_pos: usize,
    /// Total tokens seen (including evicted).
    total_tokens: usize,
}

impl ResidualWindow {
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            capacity,
            dim,
            buffer: Vec::with_capacity(capacity),
            write_pos: 0,
            total_tokens: 0,
        }
    }

    /// Push a new residual into the window. Returns evicted residual if window is full.
    pub fn push(&mut self, residual: Vec<f32>) -> Option<Vec<f32>> {
        assert_eq!(residual.len(), self.dim);
        self.total_tokens += 1;

        if self.buffer.len() < self.capacity {
            self.buffer.push(residual);
            None
        } else {
            let evicted = std::mem::replace(&mut self.buffer[self.write_pos], residual);
            self.write_pos = (self.write_pos + 1) % self.capacity;
            Some(evicted)
        }
    }

    /// Number of residuals currently in the window.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Total tokens seen (including those evicted to cold tier).
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get residuals in order (oldest to newest).
    pub fn residuals(&self) -> Vec<&Vec<f32>> {
        if self.buffer.len() < self.capacity {
            self.buffer.iter().collect()
        } else {
            let mut result = Vec::with_capacity(self.capacity);
            for i in 0..self.capacity {
                let idx = (self.write_pos + i) % self.capacity;
                result.push(&self.buffer[idx]);
            }
            result
        }
    }

    /// Memory used by the window in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.buffer.len() * self.dim * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_fill_and_evict() {
        let mut w = ResidualWindow::new(3, 4);

        assert!(w.push(vec![1.0; 4]).is_none());
        assert!(w.push(vec![2.0; 4]).is_none());
        assert!(w.push(vec![3.0; 4]).is_none());
        assert_eq!(w.len(), 3);

        // Fourth push should evict the first
        let evicted = w.push(vec![4.0; 4]);
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap(), vec![1.0; 4]);
        assert_eq!(w.total_tokens(), 4);
    }

    #[test]
    fn test_window_order() {
        let mut w = ResidualWindow::new(3, 2);
        w.push(vec![1.0, 0.0]);
        w.push(vec![2.0, 0.0]);
        w.push(vec![3.0, 0.0]);
        w.push(vec![4.0, 0.0]); // evicts [1,0]

        let residuals = w.residuals();
        assert_eq!(residuals[0][0], 2.0);
        assert_eq!(residuals[1][0], 3.0);
        assert_eq!(residuals[2][0], 4.0);
    }
}
