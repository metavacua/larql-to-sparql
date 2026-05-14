import { test, expect } from '@playwright/test';

// Probes WebGPU availability via SwiftShader in headless Chromium.
// These tests validate the browser infrastructure before WASM build
// artifacts exist. Once wgpu + wasm-bindgen land, extend this file
// with WASM module load and API surface tests.

test('navigator.gpu is present', async ({ page }) => {
  const hasGpu = await page.evaluate(() => 'gpu' in navigator);
  expect(hasGpu).toBe(true);
});

test('WebGPU adapter is available via SwiftShader', async ({ page }) => {
  const result = await page.evaluate(async () => {
    if (!('gpu' in navigator)) {
      return { available: false, reason: 'navigator.gpu not present' };
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { available: false, reason: 'requestAdapter returned null' };
    }
    const info = await adapter.requestAdapterInfo();
    return {
      available: true,
      vendor: info.vendor,
      device: info.device,
      description: info.description,
    };
  });

  console.log('WebGPU adapter:', JSON.stringify(result, null, 2));
  expect(result.available).toBe(true);
});

test('WebGPU device can be created and destroyed', async ({ page }) => {
  const result = await page.evaluate(async () => {
    if (!('gpu' in navigator)) return { success: false, reason: 'no WebGPU' };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { success: false, reason: 'no adapter' };
    try {
      const device = await adapter.requestDevice();
      device.destroy();
      return { success: true };
    } catch (e) {
      return { success: false, reason: String(e) };
    }
  });

  console.log('WebGPU device:', JSON.stringify(result, null, 2));
  expect(result.success).toBe(true);
});
