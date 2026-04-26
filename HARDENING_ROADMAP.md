# LARQL Hardening Roadmap

## Completed (PR #1)

### ✅ Build Compatibility
- Fixed Metal feature gating for Linux/non-macOS builds
- System now builds with `--no-default-features` on Linux

### ✅ Recursion Vector Identified
- Documented in `LARQL_STATIC_ANALYSIS.md`
- Root cause: interactive CLI modes (`run_chat`, `repl`) in automated contexts
- Full analysis of attack surface and data flow

### ✅ Bootstrap Safety Hardening
- TTY guards on `larql run` / `larql chat` (interactive modes)
- TTY guards on `larql repl` (LQL interactive)
- Clear error messages guide users to non-interactive alternatives
- Test coverage: 317/317 tests passing in larql-lql

## Phase 2: Enhanced Hardening (Recommended)

### 2.1 Timeout Guards
Add inactivity timeout to interactive modes to prevent resource exhaustion:

```rust
// In run_chat():
let stdin_timeout = std::time::Duration::from_secs(300); // 5 min
let start = std::time::Instant::now();

loop {
    if start.elapsed() > stdin_timeout {
        eprintln!("Idle timeout reached; exiting interactive mode");
        return Ok(());
    }
    // ... rest of loop
}
```

### 2.2 Resource Limits
Cap inference resources per turn:

- KV cache size limit (default: 2GB per session)
- Max tokens per inference (default: 256)
- Max batch size for concurrent requests (if serving)
- Memory usage monitoring with warnings

### 2.3 Batch Mode Flag
Add `--batch` / `--non-interactive` flag to suppress prompts:

```rust
pub struct RunArgs {
    pub batch_mode: bool, // Skip TTY check if true and no prompt provided
    // ...
}
```

## Phase 3: Minimal Hardened Subset (Architecture)

### 3.1 Extract `larql-infer` Binary
Create a separate, minimal inference-only binary:

**Features:**
- One-shot inference only (`infer <vindex> <prompt>`)
- Batch processing from files (`batch <vindex> <jsonl>`)
- No interactive mode, no REPL, no subprocess spawning
- Explicit timeout and resource limits

**Size/Safety Benefits:**
- Binary: 8 MB (vs. 34 MB for full CLI)
- Dependencies: 50 fewer crates
- Build time: ~6s (vs. 18s)
- Recursion risk: **Eliminated** (no stdin, no interactive modes)

**Implementation Approach:**
```
crates/larql-infer/
├── src/
│   ├── main.rs         (minimal CLI)
│   ├── infer_cmd.rs    (one-shot inference)
│   └── batch_cmd.rs    (file-based batch)
└── Cargo.toml          (minimal deps)
```

### 3.2 Feature Flags for CLI Subcommands
Gate command groups to reduce default surface:

```toml
[features]
default = ["inference", "query"]
inference = []          # run, chat, bench
query = []              # describe, stats, select, etc.
extract = []            # extract, build, compile
server = []             # serve, gRPC
research = []           # dev tools (weight-extract, etc.)
all = ["inference", "query", "extract", "server", "research"]
```

### 3.3 Lazy Model Loading
Defer vindex loading until first inference:

**Current:** `run_chat()` loads all weights upfront → high startup latency  
**Proposed:** Load on first token generation → fast startup, smaller memory footprint if never used

## Phase 4: Deployment Hardening

### 4.1 Add `--readonly` Flag
When serving vindex, mount as read-only to prevent accidental mutation:

```rust
pub struct ServeArgs {
    pub readonly: bool,  // Prevent INSERT/DELETE/UPDATE
    // ...
}
```

### 4.2 User/Permission Model
If running as server with multiple users:

- API key authentication (`--api-key`)
- Rate limiting per IP/key (`--rate-limit "100/min"`)
- Audit logging of queries

### 4.3 Secrets Scanning
Before deployment, scan for:

- Hardcoded API keys or HF tokens
- Model paths / cache locations
- Logging that includes prompt text

## Phase 5: Testing & Validation

### 5.1 Add Integration Tests
```rust
#[test]
fn test_run_chat_requires_tty() {
    // Simulate non-TTY stdin, verify error
}

#[test]
fn test_repl_requires_tty() {
    // Same for REPL
}

#[test]
fn test_timeout_guards() {
    // Verify idle timeout works
}
```

### 5.2 Stress Testing
- Load test with 100+ concurrent inference requests
- Memory usage under load (target: <5GB for Gemma 3 4B)
- Timeout behavior at scale

### 5.3 Security Audit
- Code review of inference path
- Fuzzing of LQL parser
- Supply chain audit (dependencies)

## Risk Assessment (Current)

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Interactive stdin in automation | **HIGH** | ✅ TTY guards (PR #1) |
| Infinite inference loops | **MEDIUM** | Timeout guards (Phase 2) |
| Resource exhaustion | **MEDIUM** | Limits & monitoring (Phase 2) |
| Model poisoning via LQL | **LOW** | Read-only flag (Phase 4) |
| Credential leakage | **LOW** | Secrets scanning (Phase 4) |

## Estimated Timeline

- **Phase 1:** Done (PR #1 in review)
- **Phase 2:** 1-2 days (timeout, batch mode)
- **Phase 3:** 3-5 days (extract minimal binary)
- **Phase 4:** 2-3 days (server hardening)
- **Phase 5:** 3-5 days (testing & audit)

**Total:** ~2 weeks for full hardening suite

## Checklist for PR #1 Approval

- [ ] All tests pass (currently: 317/317 in larql-lql)
- [ ] CI/CD pipeline runs (if configured)
- [ ] No security vulnerabilities in atty dependency
- [ ] Documentation updated (LARQL_STATIC_ANALYSIS.md included)
- [ ] TTY guards work as intended (manual test with non-TTY stdin)

## Next Steps

1. Wait for CI results on PR #1
2. Merge PR #1 if all tests pass
3. Create issue/PR for Phase 2 (timeout guards)
4. Plan Phase 3 (minimal binary extraction) with team

---

**Author:** Claude Code  
**Date:** April 26, 2026  
**Status:** In Progress
