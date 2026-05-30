//! Portable float-method extension trait for `wasm32v1-none`.
//!
//! `wasm32v1-none` is a `no_std` target: the transcendental and rounding
//! methods normally provided as inherent `f32`/`f64` methods by `std`
//! (`sqrt`, `powf`, `round`, `exp`, …) are unavailable. This crate supplies
//! them via `libm`, behind a `FloatExt` trait.
//!
//! On native targets the inherent `std` methods take priority over trait
//! methods, so importing `FloatExt` there is a harmless no-op. The kernel
//! prelude only imports it under `#[cfg(target_arch = "wasm32")]`, so it is
//! active exactly where the inherent methods are missing — nowhere else.
#![no_std]

/// Float methods backed by `libm`, mirroring the `std` inherent-method API
/// for the operations used across the LARQL kernels.
pub trait FloatExt {
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn round(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn recip(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl FloatExt for f32 {
    #[inline] fn sqrt(self) -> Self { libm::sqrtf(self) }
    #[inline] fn cbrt(self) -> Self { libm::cbrtf(self) }
    #[inline] fn round(self) -> Self { libm::roundf(self) }
    #[inline] fn floor(self) -> Self { libm::floorf(self) }
    #[inline] fn ceil(self) -> Self { libm::ceilf(self) }
    #[inline] fn trunc(self) -> Self { libm::truncf(self) }
    #[inline] fn fract(self) -> Self { self - libm::truncf(self) }
    #[inline] fn abs(self) -> Self { libm::fabsf(self) }
    #[inline] fn signum(self) -> Self {
        if self.is_nan() { Self::NAN } else { libm::copysignf(1.0, self) }
    }
    #[inline] fn copysign(self, sign: Self) -> Self { libm::copysignf(self, sign) }
    #[inline] fn recip(self) -> Self { 1.0 / self }
    #[inline] fn powi(self, n: i32) -> Self { libm::powf(self, n as f32) }
    #[inline] fn powf(self, n: Self) -> Self { libm::powf(self, n) }
    #[inline] fn exp(self) -> Self { libm::expf(self) }
    #[inline] fn exp2(self) -> Self { libm::exp2f(self) }
    #[inline] fn ln(self) -> Self { libm::logf(self) }
    #[inline] fn log(self, base: Self) -> Self { libm::logf(self) / libm::logf(base) }
    #[inline] fn log2(self) -> Self { libm::log2f(self) }
    #[inline] fn log10(self) -> Self { libm::log10f(self) }
    #[inline] fn sin(self) -> Self { libm::sinf(self) }
    #[inline] fn cos(self) -> Self { libm::cosf(self) }
    #[inline] fn tan(self) -> Self { libm::tanf(self) }
    #[inline] fn asin(self) -> Self { libm::asinf(self) }
    #[inline] fn acos(self) -> Self { libm::acosf(self) }
    #[inline] fn atan(self) -> Self { libm::atanf(self) }
    #[inline] fn atan2(self, other: Self) -> Self { libm::atan2f(self, other) }
    #[inline] fn sinh(self) -> Self { libm::sinhf(self) }
    #[inline] fn cosh(self) -> Self { libm::coshf(self) }
    #[inline] fn tanh(self) -> Self { libm::tanhf(self) }
    #[inline] fn hypot(self, other: Self) -> Self { libm::hypotf(self, other) }
    #[inline] fn mul_add(self, a: Self, b: Self) -> Self { libm::fmaf(self, a, b) }
}

impl FloatExt for f64 {
    #[inline] fn sqrt(self) -> Self { libm::sqrt(self) }
    #[inline] fn cbrt(self) -> Self { libm::cbrt(self) }
    #[inline] fn round(self) -> Self { libm::round(self) }
    #[inline] fn floor(self) -> Self { libm::floor(self) }
    #[inline] fn ceil(self) -> Self { libm::ceil(self) }
    #[inline] fn trunc(self) -> Self { libm::trunc(self) }
    #[inline] fn fract(self) -> Self { self - libm::trunc(self) }
    #[inline] fn abs(self) -> Self { libm::fabs(self) }
    #[inline] fn signum(self) -> Self {
        if self.is_nan() { Self::NAN } else { libm::copysign(1.0, self) }
    }
    #[inline] fn copysign(self, sign: Self) -> Self { libm::copysign(self, sign) }
    #[inline] fn recip(self) -> Self { 1.0 / self }
    #[inline] fn powi(self, n: i32) -> Self { libm::pow(self, n as f64) }
    #[inline] fn powf(self, n: Self) -> Self { libm::pow(self, n) }
    #[inline] fn exp(self) -> Self { libm::exp(self) }
    #[inline] fn exp2(self) -> Self { libm::exp2(self) }
    #[inline] fn ln(self) -> Self { libm::log(self) }
    #[inline] fn log(self, base: Self) -> Self { libm::log(self) / libm::log(base) }
    #[inline] fn log2(self) -> Self { libm::log2(self) }
    #[inline] fn log10(self) -> Self { libm::log10(self) }
    #[inline] fn sin(self) -> Self { libm::sin(self) }
    #[inline] fn cos(self) -> Self { libm::cos(self) }
    #[inline] fn tan(self) -> Self { libm::tan(self) }
    #[inline] fn asin(self) -> Self { libm::asin(self) }
    #[inline] fn acos(self) -> Self { libm::acos(self) }
    #[inline] fn atan(self) -> Self { libm::atan(self) }
    #[inline] fn atan2(self, other: Self) -> Self { libm::atan2(self, other) }
    #[inline] fn sinh(self) -> Self { libm::sinh(self) }
    #[inline] fn cosh(self) -> Self { libm::cosh(self) }
    #[inline] fn tanh(self) -> Self { libm::tanh(self) }
    #[inline] fn hypot(self, other: Self) -> Self { libm::hypot(self, other) }
    #[inline] fn mul_add(self, a: Self, b: Self) -> Self { libm::fma(self, a, b) }
}
