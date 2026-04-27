//! # Geometry expert
//!
//! Areas, perimeters, volumes, Pythagorean theorem.
//!
//! ## Ops
//!
//! - `circle_area {r} → num`
//! - `circle_circumference {r} → num`
//! - `circle_diameter {r} → num`
//! - `sphere_volume {r} → num`
//! - `sphere_surface_area {r} → num`
//! - `cylinder_volume {r, h} → num`
//! - `cone_volume {r, h} → num`
//! - `cube_volume {s} → num`
//! - `box_volume {l, w, h} → num`
//! - `square_area {s} → num`
//! - `square_perimeter {s} → num`
//! - `rectangle_area {l, w} → num`
//! - `rectangle_perimeter {l, w} → num`
//! - `triangle_area_bh {base, height} → num`
//! - `triangle_area_heron {a, b, c} → num`
//! - `trapezoid_area {a, b, h} → num`
//! - `ellipse_area {a, b} → num`
//! - `hypotenuse {a, b} → num`

use expert_interface::{arg_f64, expert_exports, json, Value};

expert_exports!(
    id = "geometry",
    tier = 1,
    description = "Geometry: areas, perimeters, volumes, Pythagorean theorem",
    version = "0.2.0",
    ops = [
        ("circle_area",          ["r"]),
        ("circle_circumference", ["r"]),
        ("circle_diameter",      ["r"]),
        ("sphere_volume",        ["r"]),
        ("sphere_surface_area",  ["r"]),
        ("cylinder_volume",      ["r", "h"]),
        ("cone_volume",          ["r", "h"]),
        ("cube_volume",          ["s"]),
        ("box_volume",           ["l", "w", "h"]),
        ("square_area",          ["s"]),
        ("square_perimeter",     ["s"]),
        ("rectangle_area",       ["l", "w"]),
        ("rectangle_perimeter",  ["l", "w"]),
        ("triangle_area_bh",     ["base", "height"]),
        ("triangle_area_heron",  ["a", "b", "c"]),
        ("trapezoid_area",       ["a", "b", "h"]),
        ("ellipse_area",         ["a", "b"]),
        ("hypotenuse",           ["a", "b"]),
    ],
    dispatch = dispatch
);

const PI: f64 = std::f64::consts::PI;

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "circle_area" => Some(json!(PI * sq(arg_f64(args, "r")?))),
        "circle_circumference" => Some(json!(2.0 * PI * arg_f64(args, "r")?)),
        "circle_diameter" => Some(json!(2.0 * arg_f64(args, "r")?)),
        "sphere_volume" => {
            let r = arg_f64(args, "r")?;
            Some(json!(4.0 / 3.0 * PI * r * r * r))
        }
        "sphere_surface_area" => Some(json!(4.0 * PI * sq(arg_f64(args, "r")?))),
        "cylinder_volume" => Some(json!(PI * sq(arg_f64(args, "r")?) * arg_f64(args, "h")?)),
        "cone_volume" => Some(json!(PI * sq(arg_f64(args, "r")?) * arg_f64(args, "h")? / 3.0)),
        "cube_volume" => {
            let s = arg_f64(args, "s")?;
            Some(json!(s * s * s))
        }
        "box_volume" => Some(json!(arg_f64(args, "l")? * arg_f64(args, "w")? * arg_f64(args, "h")?)),
        "square_area" => Some(json!(sq(arg_f64(args, "s")?))),
        "square_perimeter" => Some(json!(4.0 * arg_f64(args, "s")?)),
        "rectangle_area" => Some(json!(arg_f64(args, "l")? * arg_f64(args, "w")?)),
        "rectangle_perimeter" => Some(json!(2.0 * (arg_f64(args, "l")? + arg_f64(args, "w")?))),
        "triangle_area_bh" => Some(json!(0.5 * arg_f64(args, "base")? * arg_f64(args, "height")?)),
        "triangle_area_heron" => {
            let a = arg_f64(args, "a")?;
            let b = arg_f64(args, "b")?;
            let c = arg_f64(args, "c")?;
            let s = (a + b + c) / 2.0;
            let inside = s * (s - a) * (s - b) * (s - c);
            if inside < 0.0 { return None; }
            Some(json!(inside.sqrt()))
        }
        "trapezoid_area" => Some(json!(0.5 * (arg_f64(args, "a")? + arg_f64(args, "b")?) * arg_f64(args, "h")?)),
        "ellipse_area" => Some(json!(PI * arg_f64(args, "a")? * arg_f64(args, "b")?)),
        "hypotenuse" => {
            let a = arg_f64(args, "a")?;
            let b = arg_f64(args, "b")?;
            Some(json!((a * a + b * b).sqrt()))
        }
        _ => None,
    }
}

fn sq(x: f64) -> f64 { x * x }
