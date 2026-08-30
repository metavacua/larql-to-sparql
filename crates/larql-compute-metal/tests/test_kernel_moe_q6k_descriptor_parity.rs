#![cfg(target_os = "macos")]

//! Rung D of the GPU-dataflow routing ladder: descriptor-driven
//! addressing feeds the EXISTING Q6_K grouped-expert kernel.
//!
//! Both arms run the production `q6k_grouped_experts` MSL — reduction
//! body, binding signature, dispatch geometry all unchanged:
//!
//! - CONTROL: per-slot offset tables computed on the CPU and injected
//!   with `set_bytes`, exactly today's `moe_zero_copy` shape (the
//!   encode-time host decision).
//! - CANDIDATE: `selected_ids` + descriptor table → `moe_descriptor_gather`
//!   on the GPU → the expanded offset buffers bound with `set_buffer`.
//!
//! Outputs must agree BITWISE: same kernel, same bytes, same reduction
//! order — the only degree of freedom is where the offsets came from. A
//! mismatch therefore convicts descriptor lookup/binding, never Q6_K
//! arithmetic.
//!
//! Negative control: mutate ONE gathered-side descriptor offset by one
//! Q6_K row, holding everything else fixed — the candidate must diverge;
//! restore it — bitwise equality must return. The gate proves its own
//! ability to fail rather than inheriting confidence from rung C.

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
use larql_compute_metal::moe_descriptor::GpuExpertDescriptor;
use larql_compute_metal::shaders;
use metal::{
    CommandBufferRef, CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

const NUM_EXPERTS: usize = 32;
/// Rows per fused half (the expert's `inter`).
const N_ROWS: usize = 8;
/// Columns — one Q6_K superblock per row.
const K_COLS: usize = 256;
/// Q6_K bytes per 256-element row.
const ROW_BYTES: usize = 210;
const GATE_HALF_BYTES: usize = N_ROWS * ROW_BYTES;
const EXPERT_BYTES: usize = 2 * GATE_HALF_BYTES;
/// Non-monotonic route: slot ≠ expert_id everywhere.
const ROUTE: [u32; 4] = [17, 2, 29, 5];
/// Anti-symmetric expert placement inside the bank (quadratic stagger).
fn placement(e: usize) -> usize {
    64 + e * EXPERT_BYTES + e * e * 16
}

struct Rig {
    device: Device,
    queue: metal::CommandQueue,
    grouped: ComputePipelineState,
    gather: ComputePipelineState,
    bank: metal::Buffer,
    descs: metal::Buffer,
    ids: metal::Buffer,
    x: metal::Buffer,
}

fn build_rig() -> Rig {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let src = format!(
        "{}{}{}",
        shaders::common::HEADER,
        shaders::q6k_grouped_experts::SHADER,
        shaders::moe_descriptor::SHADER,
    );
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile grouped+descriptor shaders");
    let pipe = |name: &str| {
        let f = lib.get_function(name, None).expect("fn");
        let d = ComputePipelineDescriptor::new();
        d.set_compute_function(Some(&f));
        device.new_compute_pipeline_state(&d).expect("pipeline")
    };

    // Bank: every expert quantized from distinctive f32 rows so any
    // addressing slip produces a numerically different (finite) output.
    let bank_size = placement(NUM_EXPERTS);
    let mut bank_bytes = vec![0u8; bank_size];
    for e in 0..NUM_EXPERTS {
        let vals: Vec<f32> = (0..2 * N_ROWS * K_COLS)
            .map(|i| ((e * 131 + i) as f32 * 0.013).sin() * 0.4)
            .collect();
        let q = quantize_q6_k(&vals);
        assert_eq!(q.len(), EXPERT_BYTES, "quantizer output size");
        bank_bytes[placement(e)..placement(e) + EXPERT_BYTES].copy_from_slice(&q);
    }

    let descs_host: Vec<GpuExpertDescriptor> = (0..NUM_EXPERTS)
        .map(|e| GpuExpertDescriptor {
            gate_up_payload_off: placement(e) as u32,
            down_payload_off: 0,
            gate_up_scale_off: 0,
            down_scale_off: 0,
            gate_up_bias_off: 0,
            down_bias_off: 0,
        })
        .collect();

    let buf = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let x_vals: Vec<f32> = (0..K_COLS).map(|i| ((i as f32) * 0.031).cos()).collect();

    Rig {
        grouped: pipe("q6k_grouped_experts"),
        gather: pipe("moe_descriptor_gather"),
        bank: buf(&bank_bytes),
        descs: buf(unsafe {
            std::slice::from_raw_parts(
                descs_host.as_ptr() as *const u8,
                std::mem::size_of_val(descs_host.as_slice()),
            )
        }),
        ids: buf(unsafe { std::slice::from_raw_parts(ROUTE.as_ptr() as *const u8, 16) }),
        x: buf(unsafe { std::slice::from_raw_parts(x_vals.as_ptr() as *const u8, K_COLS * 4) }),
        device,
        queue,
    }
}

enum Offsets<'a> {
    /// CONTROL — CPU-computed, `set_bytes`: today's encode-time host
    /// decision, verbatim.
    Inline(&'a [u32]),
    /// CANDIDATE — offsets come from the GPU buffer `moe_descriptor_gather`
    /// writes from the rig's descriptor table.
    Gathered,
}

impl Rig {
    /// Run the production grouped kernel over the route's gate half and
    /// return the `[n_slots × N_ROWS]` outputs as raw bits (bitwise
    /// comparison — NaN-proof, tolerance-free).
    fn run_grouped(&self, offsets: Offsets<'_>) -> Vec<u32> {
        let n_slots = ROUTE.len();
        let out = self.device.new_buffer(
            (n_slots * N_ROWS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd: &CommandBufferRef = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // Candidate: the gather dispatch precedes the grouped kernel in
        // the SAME command buffer — the route is consumed on the GPU.
        let gathered;
        if let Offsets::Gathered = offsets {
            let slot_descs = self.device.new_buffer(
                (n_slots * std::mem::size_of::<GpuExpertDescriptor>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let gate0 = self
                .device
                .new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared);
            let gate1 = self
                .device
                .new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared);
            let down = self
                .device
                .new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared);
            let n = n_slots as u32;
            let ghb = GATE_HALF_BYTES as u32;
            let gu_sc = self
                .device
                .new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared);
            let dn_sc = self
                .device
                .new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared);
            enc.set_compute_pipeline_state(&self.gather);
            enc.set_buffer(0, Some(&self.descs), 0);
            enc.set_buffer(1, Some(&self.ids), 0);
            enc.set_buffer(2, Some(&slot_descs), 0);
            enc.set_buffer(3, Some(&gate0), 0);
            enc.set_buffer(4, Some(&gate1), 0);
            enc.set_buffer(5, Some(&down), 0);
            enc.set_buffer(6, Some(&gu_sc), 0);
            enc.set_buffer(7, Some(&dn_sc), 0);
            enc.set_bytes(8, 4, &n as *const u32 as *const _);
            enc.set_bytes(9, 4, &ghb as *const u32 as *const _);
            // #229 bounds guard: the table's length and a refusal counter.
            let num_experts = NUM_EXPERTS as u32;
            enc.set_bytes(10, 4, &num_experts as *const u32 as *const _);
            let bad_ids = self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
            enc.set_buffer(11, Some(&bad_ids), 0);
            enc.dispatch_threads(
                MTLSize::new(n_slots as u64, 1, 1),
                MTLSize::new(n_slots as u64, 1, 1),
            );
            gathered = gate0;
        } else {
            gathered = self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
        }

        let n_u32 = N_ROWS as u32;
        let k_u32 = K_COLS as u32;
        let xstride: u32 = 0;
        enc.set_compute_pipeline_state(&self.grouped);
        enc.set_buffer(0, Some(&self.bank), 0);
        match offsets {
            Offsets::Inline(offs) => {
                enc.set_bytes(1, (offs.len() * 4) as u64, offs.as_ptr() as *const _);
            }
            Offsets::Gathered => {
                enc.set_buffer(1, Some(&gathered), 0);
            }
        }
        enc.set_buffer(2, Some(&self.x), 0);
        enc.set_buffer(3, Some(&out), 0);
        enc.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
        enc.set_bytes(5, 4, &k_u32 as *const u32 as *const _);
        enc.set_bytes(6, 4, &xstride as *const u32 as *const _);
        let row_tiles = (N_ROWS as u64).div_ceil(shaders::q6k_grouped_experts::ROWS_PER_TG);
        enc.dispatch_thread_groups(
            MTLSize::new(row_tiles, n_slots as u64, 1),
            MTLSize::new(shaders::q6k_grouped_experts::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, n_slots * N_ROWS) }.to_vec()
    }

    /// Mutate one descriptor's gate_up payload offset in place (shared
    /// storage) — the negative control's single-fact perturbation.
    fn nudge_descriptor(&self, expert: usize, delta_bytes: i64) {
        unsafe {
            let p = (self.descs.contents() as *mut GpuExpertDescriptor).add(expert);
            (*p).gate_up_payload_off = ((*p).gate_up_payload_off as i64 + delta_bytes) as u32;
        }
    }
}

/// D2 — the payoff assertion: expert identity moved from CPU control
/// flow into GPU data, with the expert mathematics untouched. Bitwise.
#[test]
fn descriptor_driven_addressing_matches_set_bytes_bitwise() {
    let rig = build_rig();
    let cpu_offsets: Vec<u32> = ROUTE
        .iter()
        .map(|&e| placement(e as usize) as u32)
        .collect();

    let control = rig.run_grouped(Offsets::Inline(&cpu_offsets));
    let candidate = rig.run_grouped(Offsets::Gathered);

    // Sanity: the fixture computes real numbers, not zeros.
    let finite_nonzero = control
        .iter()
        .filter(|&&b| f32::from_bits(b) != 0.0 && f32::from_bits(b).is_finite())
        .count();
    assert!(
        finite_nonzero > control.len() / 2,
        "fixture degenerated: control output mostly zero/non-finite"
    );
    assert_eq!(
        control, candidate,
        "descriptor-driven addressing diverges from set_bytes addressing \
         on the byte-identical kernel — suspect set: gather/binding, not \
         Q6_K arithmetic"
    );
}

/// Negative control — one descriptor offset moved by ONE Q6_K row while
/// everything else stays fixed. The candidate must diverge; restoring
/// the offset must restore bitwise equality.
#[test]
fn negative_control_single_descriptor_mutation_diverges_then_restores() {
    let rig = build_rig();
    let cpu_offsets: Vec<u32> = ROUTE
        .iter()
        .map(|&e| placement(e as usize) as u32)
        .collect();
    let control = rig.run_grouped(Offsets::Inline(&cpu_offsets));

    let victim = ROUTE[0] as usize;
    rig.nudge_descriptor(victim, ROW_BYTES as i64);
    let perturbed = rig.run_grouped(Offsets::Gathered);
    assert_ne!(
        control, perturbed,
        "instrument BLIND: a one-row descriptor offset error produced \
         bitwise-identical output — this gate cannot catch addressing bugs"
    );
    // Only the victim's slot may move; the other slots' rows must be
    // untouched (the mutation is surgical, so the divergence must be too).
    for (slot, &id) in ROUTE.iter().enumerate() {
        let range = slot * N_ROWS..(slot + 1) * N_ROWS;
        if id as usize == victim {
            assert_ne!(
                &control[range.clone()],
                &perturbed[range],
                "victim slot moved"
            );
        } else {
            assert_eq!(
                &control[range.clone()],
                &perturbed[range],
                "slot {slot} (expert {id}) changed without its descriptor \
                 being touched — mutation leaked across slots"
            );
        }
    }

    rig.nudge_descriptor(victim, -(ROW_BYTES as i64));
    let restored = rig.run_grouped(Offsets::Gathered);
    assert_eq!(
        control, restored,
        "restored descriptor fails to restore bitwise equality"
    );
}

/// Both fused halves address correctly: the up half's table is the
/// gate table shifted by the ContiguousHalves layer fact, computed on
/// the GPU — and must match a CPU-shifted control bitwise.
#[test]
fn up_half_table_matches_shifted_control_bitwise() {
    let rig = build_rig();
    let up_offsets: Vec<u32> = ROUTE
        .iter()
        .map(|&e| (placement(e as usize) + GATE_HALF_BYTES) as u32)
        .collect();
    let control = rig.run_grouped(Offsets::Inline(&up_offsets));

    // Candidate: gather, then bind gate1 (the GPU-computed up half).
    let n_slots = ROUTE.len();
    let slot_descs = rig.device.new_buffer(
        (n_slots * std::mem::size_of::<GpuExpertDescriptor>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mk = |bytes: u64| {
        rig.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    };
    let (gate0, gate1, down) = (mk(16), mk(16), mk(16));
    let (gu_sc, dn_sc) = (mk(16), mk(16));
    let out = mk((n_slots * N_ROWS * 4) as u64);

    let cmd = rig.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let n = n_slots as u32;
    let ghb = GATE_HALF_BYTES as u32;
    enc.set_compute_pipeline_state(&rig.gather);
    enc.set_buffer(0, Some(&rig.descs), 0);
    enc.set_buffer(1, Some(&rig.ids), 0);
    enc.set_buffer(2, Some(&slot_descs), 0);
    enc.set_buffer(3, Some(&gate0), 0);
    enc.set_buffer(4, Some(&gate1), 0);
    enc.set_buffer(5, Some(&down), 0);
    enc.set_buffer(6, Some(&gu_sc), 0);
    enc.set_buffer(7, Some(&dn_sc), 0);
    enc.set_bytes(8, 4, &n as *const u32 as *const _);
    enc.set_bytes(9, 4, &ghb as *const u32 as *const _);
    // #229 bounds guard: the table's length and a refusal counter.
    let num_experts = NUM_EXPERTS as u32;
    enc.set_bytes(10, 4, &num_experts as *const u32 as *const _);
    let bad_ids = rig
        .device
        .new_buffer(4, MTLResourceOptions::StorageModeShared);
    enc.set_buffer(11, Some(&bad_ids), 0);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(n as u64, 1, 1));

    let n_u32 = N_ROWS as u32;
    let k_u32 = K_COLS as u32;
    let xstride: u32 = 0;
    enc.set_compute_pipeline_state(&rig.grouped);
    enc.set_buffer(0, Some(&rig.bank), 0);
    enc.set_buffer(1, Some(&gate1), 0);
    enc.set_buffer(2, Some(&rig.x), 0);
    enc.set_buffer(3, Some(&out), 0);
    enc.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
    enc.set_bytes(5, 4, &k_u32 as *const u32 as *const _);
    enc.set_bytes(6, 4, &xstride as *const u32 as *const _);
    let row_tiles = (N_ROWS as u64).div_ceil(shaders::q6k_grouped_experts::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        MTLSize::new(row_tiles, n_slots as u64, 1),
        MTLSize::new(shaders::q6k_grouped_experts::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let candidate =
        unsafe { std::slice::from_raw_parts(out.contents() as *const u32, n_slots * N_ROWS) }
            .to_vec();
    assert_eq!(control, candidate, "up-half addressing diverges");
}
