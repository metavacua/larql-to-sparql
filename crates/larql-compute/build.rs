fn main() {
    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    #[cfg(target_arch = "aarch64")]
    build.flag("-march=armv8.2-a+dotprod");

    #[cfg(target_arch = "x86_64")]
    build.flag("-mavx2");

    build.compile("q4_dot");
}
