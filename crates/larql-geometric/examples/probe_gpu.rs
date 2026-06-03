// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).into_iter().collect();
    if adapters.is_empty() {
        println!("No GPU adapters found.");
        return;
    }
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("Adapter {i}: {}", info.name);
        println!("  Backend:    {:?}", info.backend);
        println!("  DeviceType: {:?}", info.device_type);
        println!("  Driver:     {}", info.driver);
        println!("  DriverInfo: {}", info.driver_info);
        println!("  Vendor:     0x{:04x}", info.vendor);
        println!("  Device:     0x{:04x}", info.device);
    }
}
