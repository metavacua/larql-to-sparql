//! `larql capabilities` — print this release's capability manifest
//! (docs/vindex-factory.md §15.2) as JSON.

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = larql_factory::capabilities_manifest();
    let json = serde_json::to_string_pretty(&manifest)?;
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_succeeds() {
        assert!(run().is_ok());
    }
}
