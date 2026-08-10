fn main() {
    // build.rs itself always compiles for and runs on the HOST -- a
    // #[cfg(target_arch = "wasm32")] here would test the host's arch,
    // not the arch actually being built for, so the only reliable way
    // to detect the real target inside a build script is the TARGET
    // env var Cargo sets. wasm32v1-none has no C toolchain/libc sysroot
    // at all (confirmed via CI: `cc` fails to find string.h), so there
    // is nothing to compile there -- q4_dot.c's Rust FFI callers need
    // their own #[cfg(not(target_arch = "wasm32"))] gating in src/,
    // deferred to a real CI round rather than guessed here.
    if std::env::var("TARGET")
        .map(|t| t.starts_with("wasm32"))
        .unwrap_or(false)
    {
        return;
    }

    // Rebuild if anything under csrc/ changes (new .c, new .h, modified source).
    // The cc crate only auto-tracks files passed to .file(); this widens the net so
    // a new or modified C source always triggers recompilation of q4_dot.
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");

    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    #[cfg(target_arch = "aarch64")]
    build.flag_if_supported("-march=armv8.2-a+dotprod");

    #[cfg(target_arch = "x86_64")]
    build.flag_if_supported("-mavx2");

    build.compile("q4_dot");
}
