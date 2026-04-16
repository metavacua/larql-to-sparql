//! Kernel registry: name → kernel dispatch.

use std::collections::HashMap;

use super::{Kernel, KernelError};

#[derive(Default)]
pub struct KernelRegistry {
    kernels: HashMap<String, Box<dyn Kernel>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry preloaded with all V1 kernels: arithmetic, datetime.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register(Box::new(super::ArithmeticKernel));
        r.register(Box::new(super::DateTimeKernel));
        r
    }

    pub fn register(&mut self, kernel: Box<dyn Kernel>) {
        self.kernels.insert(kernel.name().to_string(), kernel);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Kernel> {
        self.kernels.get(name).map(|b| b.as_ref())
    }

    pub fn invoke(&self, name: &str, expr: &str) -> Result<String, KernelError> {
        self.get(name)
            .ok_or_else(|| KernelError::NotFound(name.into()))?
            .invoke(expr)
    }

    pub fn names(&self) -> Vec<&str> {
        self.kernels.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_have_arithmetic_and_datetime() {
        let r = KernelRegistry::with_defaults();
        let mut names = r.names();
        names.sort();
        assert_eq!(names, vec!["arithmetic", "datetime"]);
    }

    #[test]
    fn not_found_errors_clearly() {
        let r = KernelRegistry::with_defaults();
        let err = r.invoke("nonexistent", "whatever").unwrap_err();
        assert!(matches!(err, KernelError::NotFound(n) if n == "nonexistent"));
    }

    #[test]
    fn dispatches_to_arithmetic() {
        let r = KernelRegistry::with_defaults();
        assert_eq!(r.invoke("arithmetic", "2 + 3").unwrap(), "5");
    }

    struct EchoKernel;
    impl Kernel for EchoKernel {
        fn name(&self) -> &'static str { "echo" }
        fn invoke(&self, expr: &str) -> Result<String, KernelError> { Ok(expr.to_string()) }
    }

    #[test]
    fn custom_kernel_registers_and_dispatches() {
        let mut r = KernelRegistry::new();
        r.register(Box::new(EchoKernel));
        assert_eq!(r.invoke("echo", "hello").unwrap(), "hello");
    }

    #[test]
    fn custom_kernel_overrides_default() {
        let mut r = KernelRegistry::with_defaults();
        // Overwrite with an echo kernel that claims the "arithmetic" name
        struct HijackedArithmetic;
        impl Kernel for HijackedArithmetic {
            fn name(&self) -> &'static str { "arithmetic" }
            fn invoke(&self, _: &str) -> Result<String, KernelError> { Ok("hijacked".into()) }
        }
        r.register(Box::new(HijackedArithmetic));
        assert_eq!(r.invoke("arithmetic", "2 + 3").unwrap(), "hijacked");
    }
}
