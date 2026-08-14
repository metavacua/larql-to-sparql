//! Emit conformance fixture A as a real VINDEX3 container on disk.
//!
//! The smallest way to get a genuine `index.json.version: 3` directory to
//! point tooling at — `larql show`, `larql verify`, or anything else that
//! needs to prove it handles both generations rather than asserting it.
//!
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_make_fixture_a -- /tmp/fixture-a.vindex
//! larql show   /tmp/fixture-a.vindex
//! larql verify /tmp/fixture-a.vindex
//! ```

use larql_vindex::format::vindex3::{test_support::fixture_a_spec, write_container};

fn main() -> Result<(), String> {
    let out = std::env::args()
        .nth(1)
        .ok_or("usage: vindex3_make_fixture_a <out dir>")?;
    let dir = std::path::PathBuf::from(&out);
    write_container(&dir, &fixture_a_spec()).map_err(|e| format!("write container: {e}"))?;
    println!("wrote VINDEX3 container: {}", dir.display());
    println!("  larql show   {out}");
    println!("  larql verify {out}");
    Ok(())
}
