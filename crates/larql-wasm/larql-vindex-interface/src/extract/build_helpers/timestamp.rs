//! ISO-8601 UTC timestamp without a `chrono` dependency.

/// Simple ISO 8601 timestamp without chrono dependency.
pub(crate) fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Rough UTC timestamp — good enough for provenance
    let days = secs / 86400;
    let years_approx = 1970 + days / 365;
    let remainder_days = days % 365;
    let months = remainder_days / 30 + 1;
    let day = remainder_days % 30 + 1;
    let hour = (secs % 86400) / 3600;
    let min = (secs % 3600) / 60;
    let sec = secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years_approx,
        months.min(12),
        day.min(31),
        hour,
        min,
        sec
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chrono_now_returns_iso8601_z_format() {
        let s = chrono_now();
        assert_eq!(s.len(), 20);
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[19..20], "Z");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
        assert_eq!(&s[13..14], ":");
        assert_eq!(&s[16..17], ":");
    }

    #[test]
    fn chrono_now_year_above_1970() {
        let s = chrono_now();
        let year: u32 = s[..4].parse().expect("year parses");
        assert!(year >= 2020, "year {year} is too old");
    }

    #[test]
    fn chrono_now_clamps_month_and_day() {
        let s = chrono_now();
        let month: u32 = s[5..7].parse().unwrap();
        let day: u32 = s[8..10].parse().unwrap();
        assert!((1..=12).contains(&month), "month {month} out of range");
        assert!((1..=31).contains(&day), "day {day} out of range");
    }
}
