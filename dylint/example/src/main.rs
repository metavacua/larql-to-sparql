// Mirrors the real motivating shape (a portable helper whose only caller
// is itself wasm32-gated) closely enough to demonstrate the mechanism,
// without needing an actual wasm32 cross-compile: `print_summary` is only
// compiled on non-wasm32 targets, and it is `render_overview`'s only
// caller. The lint itself fires on the `render_` naming convention alone
// (see wasm32_orphaned_portable's "Known problems" doc) -- it does not
// depend on this shape actually being present.

#[cfg(not(target_arch = "wasm32"))]
fn print_summary() {
    render_overview();
}

fn render_overview() {
    println!("overview");
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    print_summary();
}
