use std::path::Path;

use wasmi::{Engine, Instance, Linker, Module, Store};

use super::wasi_shim;

/// Per-instance store data. Empty for now; WASI state is handled statelessly
/// by the shim closures (they write directly to host stdout/stderr).
pub struct ExpertStore {}

/// Compile a WASM expert's `Module` from the given `.wasm` file.
pub fn load_module(engine: &Engine, path: &Path) -> anyhow::Result<Module> {
    let bytes = std::fs::read(path)?;
    Module::new(engine, &bytes).map_err(Into::into)
}

/// Instantiate a previously compiled `Module` with a fresh WASI context.
pub fn instantiate(
    engine: &Engine,
    module: &Module,
) -> anyhow::Result<(Store<ExpertStore>, Instance)> {
    let mut store = Store::new(engine, ExpertStore {});
    let mut linker: Linker<ExpertStore> = Linker::new(engine);
    wasi_shim::add_to_linker(&mut linker)?;
    let instance = linker.instantiate_and_start(&mut store, module)?;
    Ok((store, instance))
}

/// Compile and instantiate a WASM expert in one step.
pub fn load_expert(
    engine: &Engine,
    path: &Path,
) -> anyhow::Result<(Store<ExpertStore>, Instance)> {
    let module = load_module(engine, path)?;
    instantiate(engine, &module)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_loader_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    // The .cwasm cache is gone; keep a trivial smoke test so the module
    // compiles and the loader functions are reachable.
    #[test]
    fn fresh_path_is_unique() {
        let a = fresh_path("a");
        let b = fresh_path("b");
        assert_ne!(a, b);
    }

    /// Minimal valid WASM module — 8-byte magic + version header.
    /// `wasmtime::Module::new` accepts this as an empty (no-export) module,
    /// which is enough to drive every code path in `load_module` /
    /// `instantiate` / `load_expert` without bundling a real expert binary.
    const MINIMAL_WASM: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

    fn write_minimal_wasm(name: &str) -> std::path::PathBuf {
        let path = fresh_path(name).with_extension("wasm");
        std::fs::write(&path, MINIMAL_WASM).expect("write wasm");
        path
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("cwasm"));
    }

    #[test]
    fn load_module_compiles_and_writes_cache() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_module_compile");
        let cache_path = path.with_extension("cwasm");
        assert!(
            !cache_path.exists(),
            "cache must not exist before first load"
        );

        // First call: cache miss → compile + best-effort cache write.
        let _ = load_module(&engine, &path).expect("load_module compile path");
        // The serialize step is best-effort; check the cache appeared on
        // any filesystem that allows write next to source.
        assert!(cache_path.exists(), "cache write must succeed in tempdir");

        cleanup(&path);
    }

    #[test]
    fn load_module_uses_cache_on_second_call() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_module_cached");
        // Prime the cache.
        let _ = load_module(&engine, &path).expect("first load");
        let cache_path = path.with_extension("cwasm");
        assert!(cache_path.exists(), "cache must be primed");

        // Second call: cache_is_fresh=true → deserialize branch.
        let _ = load_module(&engine, &path).expect("cached load");

        cleanup(&path);
    }

    #[test]
    fn load_module_falls_through_when_cache_corrupt() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_module_corrupt_cache");
        let cache_path = path.with_extension("cwasm");
        // Create a fresh-looking but invalid cache: deserialize will fail,
        // exercising the fall-through to canonical compile.
        std::fs::write(&cache_path, b"definitely not a wasmtime artifact").unwrap();
        // Ensure cache mtime >= source mtime so cache_is_fresh returns true.
        std::thread::sleep(std::time::Duration::from_millis(20));
        // Touch cache to be newer.
        std::fs::write(&cache_path, b"still not a wasmtime artifact").unwrap();

        let _ = load_module(&engine, &path).expect("fallback compile path");

        cleanup(&path);
    }

    #[test]
    fn instantiate_returns_store_and_instance() {
        let engine = Engine::default();
        let path = write_minimal_wasm("instantiate_test");
        let module = load_module(&engine, &path).expect("module");

        let (_store, _instance) = instantiate(&engine, &module).expect("instantiate empty module");

        cleanup(&path);
    }

    #[test]
    fn load_expert_compiles_and_instantiates_in_one_step() {
        let engine = Engine::default();
        let path = write_minimal_wasm("load_expert_test");
        let (_store, _instance) = load_expert(&engine, &path).expect("load_expert");
        cleanup(&path);
    }
}
