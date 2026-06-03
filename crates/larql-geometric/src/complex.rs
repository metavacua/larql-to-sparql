// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

impl C32 {
    #[inline] pub fn new(re: f32, im: f32) -> Self { Self { re, im } }
    #[inline] pub fn conj(self) -> Self { Self { re: self.re, im: -self.im } }
    #[inline] pub fn from_phase(theta: f32) -> Self { Self { re: theta.cos(), im: theta.sin() } }
}

impl std::ops::Mul for C32 {
    type Output = C32;
    #[inline]
    fn mul(self, rhs: C32) -> C32 {
        C32 { re: self.re*rhs.re - self.im*rhs.im, im: self.re*rhs.im + self.im*rhs.re }
    }
}

impl std::ops::Add for C32 {
    type Output = C32;
    #[inline]
    fn add(self, rhs: C32) -> C32 { C32 { re: self.re+rhs.re, im: self.im+rhs.im } }
}

/// φ(x) = exp(ix) = cos(x) + i·sin(x) applied elementwise.
pub fn phi(row: &[f32]) -> Vec<C32> {
    row.iter().map(|&x| C32::from_phase(x)).collect()
}
