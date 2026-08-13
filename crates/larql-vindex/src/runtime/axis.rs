//! Named axes for shape diagnostics.
//!
//! An enum rather than a `&'static str` field, because these strings are
//! compared, matched on in tests, and rendered into operator-facing errors.
//! Spelling one of them differently at one call site produces a message that
//! reads correctly and groups wrongly.

/// Which dimension a shape disagreement is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    Rows,
    Columns,
    /// Total element count of a vector operand.
    Length,
    /// The width a vector-valued operand carries through the operation.
    Width,
    /// The width a projection contracts over.
    InputWidth,
    /// The width a projection or reduction produces.
    OutputWidth,
}

impl Axis {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Columns => "columns",
            Self::Length => "length",
            Self::Width => "width",
            Self::InputWidth => "input width",
            Self::OutputWidth => "output width",
        }
    }
}

impl core::fmt::Display for Axis {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Axis; 6] = [
        Axis::Rows,
        Axis::Columns,
        Axis::Length,
        Axis::Width,
        Axis::InputWidth,
        Axis::OutputWidth,
    ];

    #[test]
    fn every_axis_has_a_distinct_name() {
        let mut names: Vec<&str> = ALL.iter().map(|a| a.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two axes share a name");
    }

    #[test]
    fn an_axis_displays_as_its_name() {
        for axis in ALL {
            assert_eq!(axis.to_string(), axis.name());
        }
    }

    #[test]
    fn axes_are_comparable_so_tests_can_assert_on_them() {
        assert_eq!(Axis::Rows, Axis::Rows);
        assert_ne!(Axis::Rows, Axis::Columns);
    }

    #[test]
    fn names_read_as_prose_in_a_diagnostic() {
        assert_eq!(Axis::InputWidth.name(), "input width");
        assert_eq!(Axis::OutputWidth.name(), "output width");
    }
}
