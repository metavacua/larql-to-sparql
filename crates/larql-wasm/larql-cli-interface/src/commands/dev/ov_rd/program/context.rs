/// Field name constants used in predicate JSON — avoids magic strings.
pub mod fields {
    pub const STRATUM: &str = "stratum";
    pub const ATTENDS_BOS: &str = "attends_bos";
    pub const ATTENDS_PREV: &str = "attends_prev";
    pub const POSITION: &str = "position";
    pub const CURRENT_CODE: &str = "current_code";
    pub const ORIGINAL_CODE: &str = "original_code";
    pub const TOKEN_ID: &str = "token_id";
    pub const PREV_TOKEN_ID: &str = "prev_token_id";
    pub const PREV_LAYER_FFN_TOP1_ID: &str = "prev_layer_ffn_top1_id";
}

/// Well-known stratum identifiers. Predicates use these string values.
#[allow(dead_code)]
pub mod strata {
    pub const NATURAL_PROSE: &str = "natural_prose";
    pub const TRANSLATION: &str = "translation";
    pub const ARITHMETIC: &str = "arithmetic";
    pub const CAPITAL_CITY: &str = "capital_city";
}

/// Per-position features available to guard predicates at evaluation time.
#[derive(Debug, Clone)]
pub struct PositionContext {
    pub stratum: String,
    pub position: usize,
    pub token_id: u32,
    pub prev_token_id: Option<u32>,
    pub attends_bos: bool,
    pub attends_prev: bool,
    /// Oracle code before any stage transformation (set once, never updated).
    pub original_code: usize,
    /// Code after the stages applied so far (updated at each stage boundary).
    pub current_code: usize,
}
