#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmTarget {
    Wasm32V1None,
    Wasm32Unknown,
    Wasm32Wasi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixStatus {
    Pass,
    Partial(&'static str),
    Fail(&'static str),
    NotApplicable(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub struct MatrixEntry {
    pub language: &'static str,
    pub target: WasmTarget,
    pub status: MatrixStatus,
}

pub const COMPILATION_MATRIX: &[MatrixEntry] = &[];
