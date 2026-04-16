/// Template cache for static attention head outputs.
///
/// For each known template (e.g., "The capital of X is"), stores the
/// cached attention output for all static heads. This is per-template,
/// not per-conversation — shared infrastructure.
///
/// Size per template: ~34 layers × ~9 static heads × 2560 × 2 bytes = ~1.5 MB
/// For 1000 templates: ~1.5 GB (shared across all conversations)

/// A cached template entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemplateCacheEntry {
    pub template_id: String,
    /// Per-layer: list of (head_index, cached_output_is_present).
    /// Static heads have cached outputs; dynamic heads are marked for real computation.
    pub layer_info: Vec<LayerCacheInfo>,
    /// Total memory for this template's cached outputs.
    pub memory_bytes: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerCacheInfo {
    pub layer: usize,
    pub static_head_count: usize,
    pub dynamic_head_count: usize,
}

/// The template cache.
#[derive(Debug, Default)]
pub struct TemplateAttnCache {
    pub entries: Vec<TemplateCacheEntry>,
}

impl TemplateAttnCache {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Estimated memory per template for Gemma 3-4B.
    pub fn bytes_per_template_gemma_4b() -> usize {
        // 34 layers × ~9 static heads per layer × 2560 hidden × 2 bytes (fp16)
        // ≈ 34 × 9 × 2560 × 2 = 1,566,720 ≈ 1.5 MB
        34 * 9 * 2560 * 2
    }

    /// Number of cached templates.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total memory for all cached templates.
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.memory_bytes).sum()
    }

    /// Look up a template.
    pub fn lookup(&self, template_id: &str) -> Option<&TemplateCacheEntry> {
        self.entries.iter().find(|e| e.template_id == template_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_cache_size() {
        let per_template = TemplateAttnCache::bytes_per_template_gemma_4b();
        // Should be ~1.5 MB
        assert!(per_template > 1_000_000, "Too small: {per_template}");
        assert!(per_template < 3_000_000, "Too large: {per_template}");
    }

    #[test]
    fn test_1000_templates_reasonable() {
        let per_template = TemplateAttnCache::bytes_per_template_gemma_4b();
        let total = per_template * 1000;
        // 1000 templates ≈ 1.5 GB — fits in RAM, cacheable on CDN
        assert!(total < 2_000_000_000, "1000 templates too large: {} GB", total / 1_000_000_000);
    }
}
