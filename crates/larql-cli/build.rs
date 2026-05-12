// Windows: bump the `larql` binary's reserved stack to 8 MiB.
//
// The default MSVC PE reserve is 1 MiB. Some clap command enums in this crate
// have intentionally large variants (see the `large_enum_variant` allow in
// `main.rs`), and clap materialises the full enum on the parser's stack frame.
// That tips over the 1 MiB limit on Windows long before `Cli::parse_from` can
// even emit `--help`, so `larql run --help` aborts with
// `thread 'main' has overflowed its stack`.
//
// Linux/macOS already reserve 8 MiB by default and aren't affected. The change
// is scoped to the `larql` bin so library/test binaries keep the default.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "windows" {
        if target_env == "msvc" {
            println!("cargo:rustc-link-arg-bin=larql=/STACK:8388608");
        } else {
            // GNU toolchain uses GNU ld syntax.
            println!("cargo:rustc-link-arg-bin=larql=-Wl,--stack,8388608");
        }
    }
}
