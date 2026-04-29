// SPDX-License-Identifier: Apache-2.0

use larql_build::compiler;

fn main() {
    // Rebuild if anything under csrc/ changes (new .c, new .h, modified source).
    // The cc crate only auto-tracks files passed to .file(); this widens the net so
    // a new or modified C source always triggers recompilation of q4_dot.
    compiler::set_rerun_triggers(&["csrc", "build.rs"]);

    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");

    // Apply platform-specific compiler configuration
    // (sets optimization level and CPU architecture flags)
    compiler::configure_c_compiler(&mut build);

    build.compile("q4_dot");
}
