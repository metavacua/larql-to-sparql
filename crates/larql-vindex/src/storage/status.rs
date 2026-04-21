use std::fmt;

#[derive(Debug, Clone)]
pub struct CompactStatus {
    pub epoch: u64,
    pub l0_entries: usize,
    pub l0_tombstones: usize,
    pub l1_edges: usize,
    pub l1_layers_used: usize,
    pub l2_facts: usize,
    pub l2_cycles: usize,
    pub base_layers: usize,
    pub base_features_per_layer: usize,
    pub hidden_dim: usize,
    pub memit_supported: bool,
}

impl CompactStatus {
    pub fn l0_live(&self) -> usize {
        self.l0_entries.saturating_sub(self.l0_tombstones)
    }
}

impl fmt::Display for CompactStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Storage engine status (epoch {}):", self.epoch)?;
        writeln!(
            f,
            "  L0 (WAL/KNN):    {} entries ({} live, {} tombstones)",
            self.l0_entries,
            self.l0_live(),
            self.l0_tombstones,
        )?;
        writeln!(
            f,
            "  L1 (arch-A):     {} edges across {} layers",
            self.l1_edges, self.l1_layers_used,
        )?;
        if self.memit_supported {
            writeln!(
                f,
                "  L2 (MEMIT):      {} facts across {} cycles",
                self.l2_facts, self.l2_cycles,
            )?;
        } else {
            writeln!(
                f,
                "  L2 (MEMIT):      not available (hidden_dim={} < 1024)",
                self.hidden_dim,
            )?;
        }
        write!(
            f,
            "  Base model:      {} layers × {} features",
            self.base_layers, self.base_features_per_layer,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_with_memit() {
        let s = CompactStatus {
            epoch: 5,
            l0_entries: 47,
            l0_tombstones: 3,
            l1_edges: 230,
            l1_layers_used: 4,
            l2_facts: 4200,
            l2_cycles: 2,
            base_layers: 34,
            base_features_per_layer: 16384,
            hidden_dim: 2560,
            memit_supported: true,
        };
        let text = s.to_string();
        assert!(text.contains("44 live"));
        assert!(text.contains("4200 facts"));
        assert!(text.contains("epoch 5"));
    }

    #[test]
    fn display_without_memit() {
        let s = CompactStatus {
            epoch: 0,
            l0_entries: 10,
            l0_tombstones: 0,
            l1_edges: 0,
            l1_layers_used: 0,
            l2_facts: 0,
            l2_cycles: 0,
            base_layers: 20,
            base_features_per_layer: 2048,
            hidden_dim: 512,
            memit_supported: false,
        };
        let text = s.to_string();
        assert!(text.contains("not available"));
        assert!(text.contains("hidden_dim=512"));
    }
}
