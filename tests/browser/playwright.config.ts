import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  timeout: 30_000,
  retries: 1,
  workers: 1,

  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          // SwiftShader provides software WebGPU in headless CI without a GPU.
          // The same test binary validates ChromeOS, Android Chrome, and all
          // desktop platforms — hardware and OS agnostic by design.
          args: [
            '--enable-unsafe-webgpu',
            '--use-angle=swiftshader',
            '--disable-vulkan-surface',
            '--no-sandbox',
            '--disable-setuid-sandbox',
          ],
        },
      },
    },
  ],
});
