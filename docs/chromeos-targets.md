# ChromeOS Target Accounting

This document records the authoritative mapping between ChromeOS-adjacent
execution environments and their Rust target triples, and documents CI
decisions for each.

## Environment Taxonomy

| Environment | What it is | Rust target triple | CI decision |
|---|---|---|---|
| **ArcVM** | Android Runtime for Chrome, running in a KVM-backed VM managed by crosvm | `aarch64-linux-android`, `armv7-linux-androideabi` | Covered by Phase 2b Android CI (separate) |
| **CrosVM** | ChromeOS Virtual Machine Monitor (a Rust VMM); runs as a host-side Linux process | Host-arch Linux (ChromeOS toolchain) | Functionally covered by existing Linux CI |
| **crosh** | ChromeOS shell (`Ctrl+Alt+T`); entry point for native ChromeOS host binaries built against the CrOS sysroot | `x86_64-cros-linux-gnu`, `aarch64-cros-linux-gnu`, `armv7a-cros-linux-gnueabihf`, `x86_64-pc-linux-gnu` | **Covered by `.github/workflows/chromeos.yml`** (cros_sdk chroot, weekly + manual) |
| **Crostini** | Debian-based Linux container running inside a Termina VM via CrosVM; surfaces as "Linux apps" | `x86_64-unknown-linux-gnu` | Redundant with Ubuntu CI (same runner, same target) — no dedicated CI job |
| **seL4** | Formally verified L4 microkernel; requires `no_std` Rust and bespoke build integration | `*-sel4-*` | Out of scope — future work |
| **Browser (Chrome/Firefox)** | WebAssembly in a JS runtime | `wasm32-unknown-unknown` | Out of scope for this PR — handled separately |

## Why Crostini ≠ ChromeOS

Crostini is a Debian container (`x86_64-unknown-linux-gnu`) running inside
ChromeOS via a VM. It is functionally a standard Linux environment; a build
that passes on `ubuntu-24.04` already approximates Crostini with the same
target triple. There is no value in a separate "chromeos-24.04" matrix entry
that runs the same job on the same runner.

**crosh** is the native ChromeOS host shell. Binaries accessible from crosh
are built with ChromeOS's own toolchain and sysroot (Portage `dev-lang/rust`,
cross-toolchains for CrOS-specific triples) via the Chromium OS SDK (`cros_sdk`
chroot). The SDK is self-contained (~5 GB tarball from
`storage.googleapis.com/chromiumos-sdk/`); no `repo sync` is needed for a
host-native `cargo build`.

## ChromeOS CI: `chromeos.yml`

The `.github/workflows/chromeos.yml` workflow builds each crate in the workspace
against the four CrOS target triples listed above, inside a `cros_sdk` chroot on
an `ubuntu-latest` runner.

- **Triggers:** `workflow_dispatch` (manual) and `schedule: '0 4 * * 1'` (weekly)
- **Build-only:** CI passes on successful `cargo build`; no test execution
- **Caching:** the chroot is cached with a weekly key per target triple

See `scripts/ci/build-crosh.sh` for the per-crate build script.

## GPU Backend Roadmap (Follow-on)

The following GPU acceleration paths are explicitly out of scope for this PR
and are documented here for future tracking:

| Backend | Path | Status |
|---|---|---|
| WebGPU | Browser (Chrome 113+, via Dawn); `wasm32-unknown-unknown` + web APIs | Future PR |
| Vulkan | Native Linux/Android; ChromeOS devices with Vulkan support | Future PR |
| CUDA | NVIDIA hardware; Linux only | Future PR |
| Metal | Apple hardware; macOS/iOS only | Future PR |
| OpenCL | Cross-platform; decreasing vendor support | Evaluate if needed |

For CI matrix details see `docs/ci-cd-local.md`.
