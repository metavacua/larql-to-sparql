// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use wgpu::util::DeviceExt;
use crate::attention::{AttnBackend, AttnInput, AttnOutput};
use crate::gpu::context::GpuContext;

pub struct ComplexLinearAttnGpu {
    ctx: GpuContext,
}

impl ComplexLinearAttnGpu {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Option<Self> {
        GpuContext::new_blocking().map(|ctx| Self { ctx })
    }
}

impl AttnBackend for ComplexLinearAttnGpu {
    fn name(&self) -> &str { "complex-linear-gpu" }

    fn forward(&self, inp: &AttnInput) -> AttnOutput {
        let n = inp.seq_len as u32;
        let d = inp.head_dim as u32;
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mk_storage = |data: &[f32]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let mk_rw = |size: u64| dev.create_buffer(&wgpu::BufferDescriptor {
            label: None, size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_q = mk_storage(inp.q);
        let b_k = mk_storage(inp.k);
        let b_v = mk_storage(inp.v);
        let b_ktv_re = mk_rw((d * d * 4) as u64);
        let b_ktv_im = mk_rw((d * d * 4) as u64);
        let b_out    = mk_rw((n * d * 4) as u64);
        let b_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[n, d]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/linear_complex.wgsl").into()
            ),
        });

        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_ktv_re.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_ktv_im.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b_out.as_entire_binding() },
            ],
        });

        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let mk_pipeline = |entry: &str| dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&pl), module: &shader,
            entry_point: entry,
            compilation_options: Default::default(),
        });

        let p1 = mk_pipeline("pass1_ktv");
        let p2 = mk_pipeline("pass2_out");

        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (n * d * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = dev.create_command_encoder(&Default::default());
        { let mut cp = enc.begin_compute_pass(&Default::default());
          cp.set_pipeline(&p1); cp.set_bind_group(0, &bg, &[]);
          let wg = (d + 7) / 8;
          cp.dispatch_workgroups(wg, wg, 1); }
        { let mut cp = enc.begin_compute_pass(&Default::default());
          cp.set_pipeline(&p2); cp.set_bind_group(0, &bg, &[]);
          cp.dispatch_workgroups((n * d + 63) / 64, 1, 1); }
        enc.copy_buffer_to_buffer(&b_out, 0, &readback, 0, (n * d * 4) as u64);
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        AttnOutput { out, seq_len: inp.seq_len, head_dim: inp.head_dim }
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
