# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# larql-to-sparql Linux CI environment.
#
# This Dockerfile is the *manifest* for what `runs-on: ubuntu-24.04` jobs
# in .github/workflows/ci.yml execute against. Reading this file answers
# "what apt packages, what Rust toolchain, what Python interpreter, what
# uv version does Linux CI assume?" — there is no implicit reliance on
# whatever GitHub's runner image happens to ship that week.
#
# Bumping this image:
#   1. edit this file
#   2. run .github/workflows/publish-ci-image.yml (workflow_dispatch)
#   3. copy the resulting digest into env.LINUX_CI_IMAGE in
#      .github/workflows/ci.yml
#
# See AGENTS.md "Cross-OS environment leaks" for the broader rationale
# (the container *is* the manifest; missing system libraries surface as
# image build-time failures, not as CI runtime failures).

FROM rust:1.88.0-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# System packages required by *-sys crates and the Python bindings build:
#   - libopenblas-dev, pkg-config: openblas-src "system" feature, declared
#     by crates/larql-compute and crates/larql-inference. macOS gets the
#     equivalent (Accelerate) bundled with the OS; Linux must install it.
#   - build-essential, ca-certificates, curl, git: standard Rust + uv
#     toolchain prerequisites.
#   - python3, python3-pip, python3-venv: required by crates/larql-python
#     (maturin + pytest harness, run inside build-test-matrix).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        pkg-config \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# clippy + rustfmt are not in rust:slim by default; the static-analysis
# stage in ci.yml requires both.
RUN rustup component add clippy rustfmt

# uv (Astral) for the Python bindings job. Pinned-version installs are
# the responsibility of the larql-python crate's uv.lock; this image just
# provides a uv binary on PATH. Sourced from the upstream uv image rather
# than a `curl | sh` shell pipeline so the binary's provenance is the
# Astral container registry, not an unverified script execution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build-time smoke check. If any of these fail, the image build fails —
# which is the correct moment to discover a broken environment, not when
# a CI job tries to compile a downstream crate. Treat each as a
# manifest-level invariant.
RUN cargo --version \
    && rustc --version \
    && cargo clippy --version \
    && cargo fmt --version \
    && pkg-config --modversion openblas \
    && python3 --version \
    && uv --version
