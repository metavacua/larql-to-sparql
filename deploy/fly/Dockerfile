# syntax=docker/dockerfile:1
FROM rust:1-slim AS builder
WORKDIR /build

# Copy workspace files — all members needed for dep resolution even if not built
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build larql-server only (larql-cli has unguarded Metal references; not needed on the server)
RUN apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler cmake g++ libopenblas-dev && \
    RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release -p larql-server && \
    strip target/release/larql-server

FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y ca-certificates libssl3 curl python3 python3-pip libopenblas0 && \
    pip3 install --no-cache-dir --break-system-packages huggingface_hub[hf_transfer] hf_transfer && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/larql-server /usr/local/bin/
COPY deploy/fly/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8080
CMD ["/start.sh"]
