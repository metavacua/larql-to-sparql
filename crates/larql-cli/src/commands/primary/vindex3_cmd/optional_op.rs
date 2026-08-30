//! Rendering for operations a component may or may not have.

use std::fmt::Display;

/// How an absent operation prints.
///
/// Deliberately not a number. The plan spends an `Option` to keep "this
/// model declares no such operation" separate from "this model declares a
/// multiply by one"; rendering absence as `1` would discard that
/// distinction at the last step, in the one place a human reads it.
const ABSENT: &str = "absent";

/// Render an optional operation's scalar: its value, or [`ABSENT`].
pub fn scalar<T: Display>(value: Option<T>) -> String {
    value.map_or_else(|| ABSENT.to_string(), |v| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_present_scalar_renders_as_its_value() {
        assert_eq!(scalar(Some(3.87_f64)), "3.87");
        assert_eq!(scalar(Some(0.196_f32)), "0.196");
    }

    #[test]
    fn absence_is_not_rendered_as_an_identity() {
        // The whole point: `None` must not read as `1`, and must not read
        // as the same text a declared 1.0 produces.
        let absent = scalar(None::<f64>);
        assert_eq!(absent, "absent");
        assert_ne!(absent, scalar(Some(1.0_f64)));
    }
}
