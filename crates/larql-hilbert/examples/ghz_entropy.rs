//! Verification surface for the n-qubit generalization: build GHZ_n two ways
//! (constructor and a H+CNOT-ladder gate circuit), confirm they agree, and
//! report entanglement entropy across each single-qubit cut plus the classical
//! outcome entropy via the NRegister trait.

use larql_hilbert::nqubit::NQubit;
use larql_hilbert::ngate::{apply_1q, apply_cnot};
use larql_hilbert::entropy::entanglement_entropy_bipartition;
use larql_hilbert::register::NRegister;
use larql_hilbert::unitary::hadamard;

fn main() {
    let n = 4;

    // Build GHZ_n by a circuit: H on qubit 0, then a CNOT ladder.
    let mut circuit = apply_1q(&NQubit::basis(n, 0), &hadamard(), 0);
    for k in 0..n - 1 {
        circuit = apply_cnot(&circuit, k, k + 1);
    }
    let constructed = NQubit::ghz(n);
    let agree = circuit
        .amp
        .iter()
        .zip(constructed.amp.iter())
        .all(|(a, b)| (a - b).norm() < 1e-12);
    println!("GHZ_{n}: circuit matches constructor = {agree}");

    println!("entanglement entropy across each single-qubit cut (expect 1 ebit):");
    for q in 0..n {
        let s = entanglement_entropy_bipartition(&constructed, &[q]);
        println!("  cut {{{q}}} -> {s:.6} ebits");
    }

    println!(
        "classical outcome entropy (NRegister) = {:.6} bits (expect 1.0)",
        constructed.entropy_bits()
    );
}
