//! Native bounded kernels — arithmetic, datetime. Deterministic, pure,
//! hard-capped cost. See individual module docs for syntax.
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;
pub mod arithmetic;
pub mod datetime;
pub mod registry;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("kernel not registered: {0}")]
    NotFound(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("evaluation error: {0}")]
    Eval(String),
    #[error("out of range: {0}")]
    OutOfRange(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

/// A bounded compute kernel: expression string in, result string out.
pub trait Kernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn invoke(&self, expr: &str) -> Result<String, KernelError>;
}

pub use arithmetic::ArithmeticKernel;
pub use datetime::DateTimeKernel;
pub use registry::KernelRegistry;
