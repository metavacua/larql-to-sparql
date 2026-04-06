/// Template detection and pattern walk cache.
///
/// Templates are recurring query structures like "The capital of X is".
/// 99% of the residual trajectory is shared across entities within
/// a template (0.99 cosine similarity). Only ~1% is entity-specific.
///
/// The pattern walk cache stores the shared scaffolding:
/// which layers are critical, which feature ranges activate.
/// At inference, only the entity KNN needs to run.

/// A cached pattern walk for a known template.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternWalk {
    /// Template identifier (e.g., "capital-of").
    pub template_id: String,
    /// Critical layers where entity-dependent features activate.
    pub critical_layers: Vec<usize>,
    /// Feature index ranges per critical layer.
    pub feature_ranges: Vec<(usize, Vec<std::ops::Range<u32>>)>,
    /// Number of entities validated against this template.
    pub validated_count: usize,
    /// Mean cosine similarity across validated entities.
    pub mean_cosine: f32,
}

impl PatternWalk {
    /// Create a pattern walk for the "capital of X" template.
    /// Based on measured data: L13 task classifier, L15 confidence router,
    /// L24-L26 factual retrieval.
    pub fn capital_of() -> Self {
        Self {
            template_id: "capital-of".to_string(),
            critical_layers: vec![13, 15, 24, 25, 26],
            feature_ranges: vec![
                (13, vec![8000..8500]),  // Task classifier features
                (15, vec![3000..3200]),  // Confidence router
                (24, vec![5000..6000]),  // Factual retrieval
                (25, vec![5000..6000]),
                (26, vec![5000..6000]),
            ],
            validated_count: 100,
            mean_cosine: 0.993,
        }
    }

    /// Number of KNN lookups needed (one per critical layer).
    pub fn knn_lookups(&self) -> usize {
        self.critical_layers.len()
    }

    /// Estimated latency: ~20us per KNN lookup.
    pub fn estimated_latency_us(&self) -> f64 {
        self.knn_lookups() as f64 * 20.0
    }
}

/// Template cache: stores pattern walks for known templates.
#[derive(Debug, Default)]
pub struct TemplateCache {
    pub patterns: Vec<PatternWalk>,
}

impl TemplateCache {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Pre-populate with known templates.
    pub fn with_defaults() -> Self {
        Self {
            patterns: vec![PatternWalk::capital_of()],
        }
    }

    /// Look up a pattern walk by template ID.
    pub fn lookup(&self, template_id: &str) -> Option<&PatternWalk> {
        self.patterns.iter().find(|p| p.template_id == template_id)
    }

    /// Number of cached templates.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}
