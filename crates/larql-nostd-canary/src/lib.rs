//! A minimal canary crate with a deliberately planted, known-in-advance
//! outcome. This file uses `std::vec::Vec`/`std::string::String` directly
//! -- Stage A's clippy lints (`std_instead_of_alloc`) are already proven
//! (this repo's own real CI, Task 16) to rewrite these to their
//! `alloc`-crate equivalents, and Stage B's `insert_no_std_scaffold()` is
//! already proven to complete the result into a crate buildable under
//! `-Z build-std=core,alloc`. This crate has zero dependencies beyond
//! core/alloc, so after Stage A+B mutation it MUST compile cleanly under
//! `-Z build-std=core,alloc` on any target with a real core sysroot --
//! there is no legitimate reason for this specific check to ever fail.

pub fn make_greeting(name: &str) -> std::string::String {
    let mut parts: std::vec::Vec<std::string::String> = std::vec::Vec::new();
    parts.push(std::string::String::from("hello, "));
    parts.push(std::string::String::from(name));
    parts.concat()
}
