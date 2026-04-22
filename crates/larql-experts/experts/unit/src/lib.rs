//! # Unit expert
//!
//! Unit conversion across length, mass, temperature, volume, speed, energy.
//! Units are language-neutral identifiers (`"km"`, `"mi"`, `"C"`, `"F"`,
//! `"kg"`, `"L"`, `"J"`, etc.) — no English unit-name parsing.
//!
//! ## Ops
//!
//! - `convert {value: num, from: unit_id, to: unit_id} → num | null` (null when
//!   `from` and `to` belong to different groups)
//! - `unit_info {unit: unit_id} → {id, group, to_si, offset}`
//! - `list_units {group?: string} → [unit_id]`
//!
//! ## Groups
//!
//! `length`, `mass`, `temp`, `volume`, `speed`, `energy`.

use expert_interface::{arg_f64, arg_str, expert_exports, json, Value};

expert_exports!(
    id = "unit",
    tier = 1,
    description = "Unit conversion: length, mass, temperature, volume, speed, energy",
    version = "0.2.0",
    ops = ["convert", "unit_info", "list_units"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "convert" => {
            let v = arg_f64(args, "value")?;
            let from = find(arg_str(args, "from")?)?;
            let to = find(arg_str(args, "to")?)?;
            if from.group != to.group { return Some(Value::Null); }
            let si = (v + from.offset) * from.to_si;
            Some(json!(si / to.to_si - to.offset))
        }
        "unit_info" => {
            let u = find(arg_str(args, "unit")?)?;
            Some(json!({
                "id": u.id,
                "group": u.group,
                "to_si": u.to_si,
                "offset": u.offset,
            }))
        }
        "list_units" => {
            let group = args.get("group").and_then(|v| v.as_str());
            let ids: Vec<&str> = UNITS.iter()
                .filter(|u| group.is_none_or(|g| u.group == g))
                .map(|u| u.id)
                .collect();
            Some(json!(ids))
        }
        _ => None,
    }
}

/// A canonical unit. `id` is the language-neutral identifier; conversions use
/// SI = (value + offset) * to_si (offset is zero for everything except temperature).
struct Unit {
    id: &'static str,
    to_si: f64,
    offset: f64,
    group: &'static str,
}

const UNITS: &[Unit] = &[
    // Length (SI = metre)
    Unit { id: "m",   to_si: 1.0,      offset: 0.0, group: "length" },
    Unit { id: "km",  to_si: 1000.0,   offset: 0.0, group: "length" },
    Unit { id: "cm",  to_si: 0.01,     offset: 0.0, group: "length" },
    Unit { id: "mm",  to_si: 0.001,    offset: 0.0, group: "length" },
    Unit { id: "mi",  to_si: 1609.344, offset: 0.0, group: "length" },
    Unit { id: "yd",  to_si: 0.9144,   offset: 0.0, group: "length" },
    Unit { id: "ft",  to_si: 0.3048,   offset: 0.0, group: "length" },
    Unit { id: "in",  to_si: 0.0254,   offset: 0.0, group: "length" },
    Unit { id: "nmi", to_si: 1852.0,   offset: 0.0, group: "length" },
    // Mass (SI = kilogram)
    Unit { id: "kg", to_si: 1.0,           offset: 0.0, group: "mass" },
    Unit { id: "g",  to_si: 0.001,         offset: 0.0, group: "mass" },
    Unit { id: "mg", to_si: 1e-6,          offset: 0.0, group: "mass" },
    Unit { id: "t",  to_si: 1000.0,        offset: 0.0, group: "mass" },
    Unit { id: "lb", to_si: 0.45359237,    offset: 0.0, group: "mass" },
    Unit { id: "oz", to_si: 0.028349523125, offset: 0.0, group: "mass" },
    Unit { id: "st", to_si: 6.35029318,    offset: 0.0, group: "mass" },
    // Temperature (SI = celsius; linear: celsius = (value + offset) * to_si)
    Unit { id: "C", to_si: 1.0,     offset: 0.0,    group: "temp" },
    Unit { id: "K", to_si: 1.0,     offset: -273.15, group: "temp" },
    Unit { id: "F", to_si: 5.0/9.0, offset: -32.0,  group: "temp" },
    // Volume (SI = litre)
    Unit { id: "L",    to_si: 1.0,           offset: 0.0, group: "volume" },
    Unit { id: "mL",   to_si: 0.001,         offset: 0.0, group: "volume" },
    Unit { id: "cL",   to_si: 0.01,          offset: 0.0, group: "volume" },
    Unit { id: "m3",   to_si: 1000.0,        offset: 0.0, group: "volume" },
    Unit { id: "gal",  to_si: 3.785411784,   offset: 0.0, group: "volume" },
    Unit { id: "ukgal", to_si: 4.54609,      offset: 0.0, group: "volume" },
    Unit { id: "pt",   to_si: 0.473176473,   offset: 0.0, group: "volume" },
    Unit { id: "floz", to_si: 0.0295735296,  offset: 0.0, group: "volume" },
    Unit { id: "cup",  to_si: 0.2365882365,  offset: 0.0, group: "volume" },
    Unit { id: "tbsp", to_si: 0.014786765,   offset: 0.0, group: "volume" },
    Unit { id: "tsp",  to_si: 0.004928922,   offset: 0.0, group: "volume" },
    // Speed (SI = m/s)
    Unit { id: "mps",  to_si: 1.0,      offset: 0.0, group: "speed" },
    Unit { id: "kmh",  to_si: 1.0/3.6,  offset: 0.0, group: "speed" },
    Unit { id: "mph",  to_si: 0.44704,  offset: 0.0, group: "speed" },
    Unit { id: "kn",   to_si: 0.514444, offset: 0.0, group: "speed" },
    // Energy (SI = joule)
    Unit { id: "J",    to_si: 1.0,       offset: 0.0, group: "energy" },
    Unit { id: "kJ",   to_si: 1000.0,    offset: 0.0, group: "energy" },
    Unit { id: "cal",  to_si: 4.184,     offset: 0.0, group: "energy" },
    Unit { id: "kcal", to_si: 4184.0,    offset: 0.0, group: "energy" },
    Unit { id: "Wh",   to_si: 3600.0,    offset: 0.0, group: "energy" },
    Unit { id: "kWh",  to_si: 3600000.0, offset: 0.0, group: "energy" },
];

fn find(id: &str) -> Option<&'static Unit> {
    UNITS.iter().find(|u| u.id == id)
}
