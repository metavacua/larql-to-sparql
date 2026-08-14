//! Is a low eta a property of the KERNEL or of the SHAPE?
//!
//! The composed ledger assigns `q4k_matvec`'s measured 0.59 to K3's attention
//! class — but that was measured on a Gemma shape in Q4K, while a transcoded
//! image runs Q6K on a K3 shape. Both the kernel and the geometry differ, so the
//! attribution is unproven and the two possibilities imply opposite plans.
//!
//! This crosses both kernels over both shape families under the batched,
//! cold-rotating protocol. The Gemma rows are anchors: they must reproduce the
//! banked 0.59 / 0.85 or the harness is not comparable.
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example diag_shape_census

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    let cells = larql_compute_metal::diag::kernel_profile::profile_shape_census(34, 5, 30);

    println!();
    println!("--- attribution ---");
    let find = |kernel: &str, shape: &str| {
        cells
            .iter()
            .find(|c| c.kernel == kernel && c.shape.contains(shape))
            .map(|c| c.eta)
    };
    let (gw4, gd6) = (find("q4k", "gemma Wo"), find("q6k", "gemma down"));
    let (ka4, ka6) = (find("q4k", "K3 KDA"), find("q6k", "K3 KDA"));
    if let (Some(gw4), Some(gd6)) = (gw4, gd6) {
        println!(
            "  anchors: gemma Wo q4k {gw4:.2} (banked 0.59), gemma down q6k {gd6:.2} (banked 0.85)"
        );
    }
    if let (Some(a4), Some(a6)) = (ka4, ka6) {
        println!("  K3 KDA attention: q4k {a4:.2}, q6k {a6:.2}");
        if a6 >= 0.75 {
            println!("  => KERNEL-BORNE. Q6K transcode sidesteps the Wo problem;");
            println!("     rebuild the composed ceiling with attention at {a6:.2}.");
        } else if let Some(d6) = gd6 {
            if a6 < d6 * 0.85 {
                println!(
                    "  => SHAPE-BORNE. The same q6k kernel loses {:.0}% moving from",
                    100.0 * (1.0 - a6 / d6)
                );
                println!("     the down shape to the attention shape. The transcode ceiling");
                println!("     stands, AND every future attention kernel inherits this —");
                println!("     it is a design constraint on the family, not one kernel to fix.");
            }
        }
    }
}
