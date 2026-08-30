//! Colocated tests for [`super::abi`].

use super::abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};

#[test]
fn the_current_abi_is_supported() {
    assert!(CURRENT_VINDEX3_ABI.is_supported());
}

#[test]
fn a_different_abi_value_is_not_supported() {
    assert!(!Vindex3Abi(CURRENT_VINDEX3_ABI.get() + 1).is_supported());
}

#[test]
fn get_round_trips_the_wrapped_value() {
    assert_eq!(Vindex3Abi(7).get(), 7);
}

#[test]
fn display_renders_the_bare_number() {
    assert_eq!(Vindex3Abi(1).to_string(), "1");
}

#[test]
fn ordering_compares_by_wrapped_value() {
    assert!(Vindex3Abi(1) < Vindex3Abi(2));
}
