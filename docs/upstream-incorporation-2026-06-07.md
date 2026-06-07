# Upstream Incorporation Audit — 2026-06-07

Exhaustive classification of all commits in `chrishayuk/larql` and `ianblenke/larql`
that are not present in `metavacua/larql-to-sparql` as of 2026-06-07.

**Common ancestor (chrishayuk):** `c61e9644` — Merge PR #92 fix/evalexpr-license-agpl  
**Common ancestor (ianblenke):** `edd395ab` — Merge PR #90 deem0n/windows-fix

**Status legend:**
- `✓ INCORPORATED` — cherry-picked into this PR (`upstream/incorporation`)
- `⚡ ALREADY IN FORK` — already present in `metavacua/larql-to-sparql`
- `⏳ DEFERRED` — not cherry-picked; reason given

---

## Part 1: chrishayuk/larql (upstream/main) — 155 commits

### Incorporated (this PR)

| Commit | Description | Priority | Notes |
|--------|-------------|----------|-------|
| `32f78fe2` | fix(probe): signed thresholding, full-depth scan, combined-index matching | P0 | Fork deleted 3 pilot scripts; restored upstream versions |
| `8c816dca` | fix(gguf): fall back to expert_feed_forward_length for MoE-only configs | P0 | Prevents crash on pure-MoE DeepSeek V4 configs |
| `834d0659` | fix(larql-vindex): document u64 overflow guard for 32-bit hosts | P0 | Code already in fork; added upstream explanatory comment |
| `d0b915b9` | fix(gguf): map deepseek_v4/deepseekv4 arch string to DeepSeekV4Arch | P1 | Clean cherry-pick |
| `a4ea55f1` | fix(cli): make HF cache scan recognise model-repo pulls | P2 | Clean cherry-pick |
| `4e4f7b29` | style(cli): rustfmt cache.rs scan_finds_both_models prefixes | P2 | Goes with a4ea55f1 |
| `9b56cd2e` | feat(mla): add qk_nope/rope/v_head_dim fields for DS-V3 MLA absorption | P1 | Clean cherry-pick |
| `2a1fc079` | feat(vindex): MLA absorption — fuse DS-V3 low-rank Q/K/V into dense weight matrices | P1 | Minor conflict in weights/mod.rs (added module declarations) |
| `d93797fe` | feat(gqa): add gqa_attention_asym for MLA-absorbed asymmetric head dims | P1 | gqa.rs deleted in fork but exists in upstream; restored |
| `f2a4c348` | feat(ggml): add Q3_K and Q5_K dequantization (types 11 and 13) | P1 | From mvkorobkov PR #103; clean cherry-pick |
| `ebc50c34` | feat(ggml): TQ1_0/TQ2_0 ternary quantisation for BitNet 1.58 | P1 | From gburd PR #148; clean cherry-pick |
| `9467ae11` | feat(ggml): add I2_S decoder for Microsoft bitnet.cpp GGUFs | P1 | Goes with ebc50c34 |
| `8f1c8f3f` | feat(gguf): surface MLA metadata for DeepSeek-V2/V3 + Kimi K2 | P1 | Merge commit; cherry-picked with -m 1 |
| `2d10daa8` | feat(vindex): wire MLA absorption into f32 weight writer | P1 | Minor conflict in gemma3.rs (added method) |
| `58c849fa` | fix(extract): accept GGUF input (file or directory) | P2 | From mvkorobkov PR #133; clean cherry-pick |

### Not incorporated — already in fork

| Commit | Description | Notes |
|--------|-------------|-------|
| `f3831ce2` | feat(android): enable aarch64-linux-android cross-compilation | Already in fork (no-op cherry-pick) |

### Deferred

| Commit | Description | Reason |
|--------|-------------|--------|
| `89cadf70` | fix(ci): serialise larql-server coverage run | CI policy conflict; fork has diverged CI config |
| `a3efe70e` | test(models): restore tq.rs coverage, retire baseline | Requires synchronized coverage baselines |
| `7cc91a17` | test(kv): restore standard.rs + generation.rs coverage | Touches deleted files in fork (kv engines refactored) |
| `60c5306a` | test(compute): restore decode.rs + kv_dispatch/cpu.rs coverage | Touches deleted files in fork |
| `c95331c2` | fix(ci): baseline inherited compute per-file coverage debt | Coverage baseline divergence |
| `5bf8bb4f` | fix(ci): cross-platform compile + coverage-policy fixes for green main | Multi-file conflict; kv_generate.rs diverged |
| `b61836bc` | Merge fix/moe-setup-pure-moe: MoE KV engines, V1 probe, lql coverage | Large merge with many conflicts; see individual commits below |
| `3b09cc87` | test(lql): raise line coverage 89.2%->93.9% (20 files cleared 90%) | Requires synchronized test state |
| `d94e15ee` | style(inference): fix needless_range_loop in walk_ffn examples | walk_ffn examples deleted/moved in fork |
| `d40a1690` | fix(ci): serialise larql-compute-metal tests | Metal extraction not yet in fork |
| `ada668ef` | feat(moe): within-expert V1 aim-validation probe — FALSIFIED, closes KU4 | Conflicts in ROADMAP_STATUS.md (deleted in fork), kquant_forward/hidden.rs (deleted) |
| `623ea02c` | improving cpu speed | WIP commit; part of unmerged compute branch |
| `2ddd0bad` | improving cpu speed | WIP commit |
| `0c586b8b` | working on update | WIP commit |
| `d6c60884` | working on moe fix | WIP commit |
| `b6d5e8d5` | Merge pull request #148 from gburd/feat/bitnet-vindex | Merge commit; individual content already cherry-picked via ebc50c34/9467ae11 |
| `270269ca` | chore: fix clippy + fmt in tq.rs, remove UPSTREAM_PRS.md | Merge of #152; conflicts in pipeline_layer.rs (deleted in fork) |
| `8653546f` | Merge pull request #152 from deem0n/fix/moe-shards-pure-moe-and-metal | Large merge; pipeline_layer.rs deleted in fork |
| `49403543` | fix(moe-shards): guard empty q4 dense FFN + wire --metal | Requires pipeline_layer.rs context (deleted in fork) |
| `a7ba8b48` | docs: methods note, changeset specs, working model updates | Upstream roadmap docs; would conflict with fork's roadmap |
| `d248a59d` | Merge pull request #145 from chrishayuk/feat/streaming-gguf | Large merge commit; gguf streaming is a significant restructure |
| `c54875db` | feat(gguf): consolidate PRs #135-139, split gguf.rs into modular gguf/ directory | Large structural refactor; conflicts in gguf.rs which fork has modified |
| `be5d7dcd` | fix(coverage): update policy for gguf/ module split | Depends on c54875db landing |
| `281084a3` | fix(coverage): lower vindex streaming baselines | Depends on c54875db |
| `9598975e` | fix: add has_vision_config to test configs after rebase | Depends on multi-modal merge |
| `5294a3b2` | feat(streaming/down_meta): incremental per-layer flush | Part of GGUF streaming refactor (c54875db) |
| `fd6f0b43` | feat(streaming): GGUF support in extract pipeline | Part of GGUF streaming refactor |
| `80bd96b6` | feat(streaming): GGUF support in extract pipeline (browse-level) | Part of GGUF streaming refactor |
| `575963d5` | feat(gguf): expose GgufTensorInfo accessors for streaming consumers | Part of GGUF streaming refactor |
| `2de3fd27` | feat(gguf): multi-shard reader for *-NNNNN-of-NNNNN.gguf splits | Part of GGUF streaming refactor; requires structural changes |
| `8f1c8f3f` | Merge pull request #144 from chrishayuk/feat/multi-modal-phase2 | Already cherry-picked above as MLA metadata |
| `143c048a` | fix(coverage): lower projector.rs baseline | Depends on multi-modal merge |
| `8432d1c7` | docs: update roadmaps and multi-modal doc for Phase 2 shipped status | Upstream roadmap; conflicts with fork's roadmap |
| `11e012d0` | feat(multi-modal): Phase 2 — Granite Vision protocol, MLP connector, AnyRes tiler | Very large; conflicts in 15+ files including deleted kv_engine files |
| `4654ef76` | fix(coverage): update policy entry for renamed connectors/projector.rs | Depends on multi-modal Phase 2 |
| `1173f18b` | feat(multi-modal): Phase 0 + Phase 1 — cross-architecture vision captioning | Very large; conflicts in 15+ files (kv_engine.rs, generation.rs, etc.) |
| `da51ac37` | Merge pull request #143 from chrishayuk/feat/multi-modal-phase1 | Merge of 1173f18b |
| `8964ece2` | Merge pull request #142 from chrishayuk/refactor/kv-engine-retrieval-trait-split | KV engine refactor; kv_engine.rs deleted in fork |
| `66c825a7` | reworked kv engine to support mode 5 upcoming | KV engine refactor; file deleted in fork |
| `2105f792` | Merge pull request #141 from chrishayuk/chore/accuracy-score-outcome | Depends on accuracy files (deleted in fork) |
| `3202067b` | improved coverage tests | WIP; depends on accuracy files |
| `a6166c3c` | Merge pull request #140 from chrishayuk/chore/accuracy-score-outcome | Depends on accuracy files |
| `52e4e0c3` | kv engine coverage | WIP; KV files deleted in fork |
| `38753588` | accuracy | WIP commit |
| `8e1e9f1d` | working on accuracy scoring for kv | WIP commit |
| `07684457` | accuracy: surface skipped prompts as ScoreOutcome variants | accuracy_cmd.rs deleted in fork; bench/run.rs deleted |
| `460ebf2b` | working on kv improvements | WIP commit |
| `cce73861` | Merge pull request #96 from mvkorobkov/main | Merge; contained in Q3K/Q5K cherry-pick and GGUF extract cherry-pick |
| `45f473a4` | feat(gguf): surface MLA metadata (earlier attempt) | Superseded by 8f1c8f3f |
| `b97fa188` | Merge pull request #133 from mvkorobkov/fix/extract-gguf-input-regression | Merge; content cherry-picked via 58c849fa |
| `72e8f3a0` | Merge pull request #132 from chrishayuk/fix/bench-regress-crate-routing | Depends on larql-compute-metal extraction; would need adapt |
| `224bbc0c` | fix(ci): route bench-regress to the new crate homes after Metal extraction | Depends on Metal extraction |
| `2ab8277c` | Merge pull request #130 from chrishayuk/chore/example-clippy-and-doctest-cleanup | Examples/doctests differ in fork |
| `423803d2` | chore: fix 3 example clippy lints + 1 broken doc-test | Depends on example state |
| `810f1639` | Merge pull request #129 from chrishayuk/fix/ci-bench-regress-and-protoc | CI divergence |
| `996cf04f` | ci: restore bench-regress.sh, swap protoc action | CI divergence |
| `0bcd14cf` | Merge pull request #127 from chrishayuk/ci/windows-disable-openblas | kv_generate.rs conflict |
| `022203b5` | ci(windows): drop OpenBLAS, fall back to pure-Rust ndarray matmul | kv_generate.rs conflict (deleted in fork) |
| `5b42e70` | Merge pull request #126 from chrishayuk/fix/remote-ffn-decode-norms | |
| `4b8ac8e1` | fix(remote-ffn): apply pre/post FFN norms across every decode-loop dispatch branch | kv_generate.rs deleted in fork |
| `83345ad5` | fix(kv): apply PLE + layer_scalar in cached prefill / decode | kv_generate.rs conflict |
| `025f7a4a` | Merge pull request #125 from deem0n/fix/gemma4-ple-kv-cached-decode | merge of 83345ad5 |
| `1f268aca` | Merge pull request #124 from chrishayuk/ci/windows-blas-single-threaded | kv_generate.rs conflict |
| `351b1ac4` | ci(windows): single-threaded BLAS + cargo runner | kv_generate.rs conflict |
| `13b380a3` | test(remote-ffn): cover apply_post_ffn_norm + apply_norm_for_ffn | Depends on kv files |
| `5ab2d078` | fix: wire --metal flag into remote FFN path, add post-FFN norms | Depends on kv files |
| `4bbc04c6` | Merge pull request #122 from chrishayuk/fix/remote-ffn-norms-and-metal-flag | merge of 5ab2d078 |
| `ae35058b` | test(vindex): cover the PLE-arch missing-sidecar load rejection | Depends on PLE context |
| `ae35058b` | fix(vindex): write + validate Gemma-4 PLE sidecars on --quant none | Depends on PLE sidecar |
| `8054dccc` | Merge pull request #121 from chrishayuk/fix/gemma4-ple-sidecars-cherrypick | merge of ae35058b |
| `716355fb` | fix(vindex): write + validate Gemma-4 PLE sidecars | fork has different write path |
| `fd42b88d` | Merge pull request #111 from chrishayuk/dependabot | CI deps bump |
| `fd42b88d` | Merge pull request #120 from chrishayuk/compute-refactor | Mega Metal extraction; too large |
| `e8817644` | docs update | WIP |
| `93c4ec58` | clippy | WIP |
| `bc223457` | Merge pull request #109 from chrishayuk/compute-refactor | Mega Metal extraction |
| `26a09c8f` | improved kv engines | WIP |
| `d2545e1c` | modularized kv engines | WIP; kv_engine.rs deleted in fork |
| `517e7c7b` | working on ci fixes | WIP |
| `a4908d9a` | fix(server): drain completion response bodies | Server refactoring conflict |
| `c7c57e5a` | test(metal+server): cover W10 state-dump masks | Metal-specific test |
| `b56b5f38` | fix ci build issues | WIP |
| `cce4d634` | Merge branch 'main' into compute-refactor | merge commit |
| `5aaf2490` | upped coverage | WIP |
| `fc28a595` | Merge pull request #113 from deem0n/windows-fix-attention-step-async | Windows fix |
| `96154b1a` | fix(windows): gate attention_step_async_matches_sync on non-Windows | Attention step async deleted in fork |
| `ac646a62` | metal decouple and kv engine | WIP Metal extraction |
| `f0bf043f` | ci(deps): bump the github-actions group | Deps bump CI |
| `82a333a7` | compute rework | WIP Metal extraction |
| `ed5e9859` | Merge pull request #104 from chrishayuk/grid-server | Grid server; large change |
| `23a57f9f` | fix server ci: guard q8k batch decode against bogus num_entries | Conflicts with server state |
| `fc1dabc2` | fix windows ci: gate prefill K/V parity | Windows CI |
| `53ea9598` | fix blas issue | WIP |
| `90a23cdc` | fixing issues with ci | WIP |
| `c7413013` | working on engines | WIP |
| `289ca738` | updated test coverage for kv etc | WIP |
| `adda9546` | working on quant | WIP |
| `eea64f20` | working on quant | WIP |
| `9a7f3d7a` | rename internal q4k API surface to kquant | Large rename; conflicts everywhere |
| `7fdc3f7a` | rename q4k-only Rust identifiers to kquant-generic | Large rename |
| `f9e9ddab` | docs update | WIP |
| `0736a38b` | working on coverage still | WIP |
| `bdc6b6e3` | working on coverage and kv engines | WIP |
| `b4062d6f` | improving coverage | WIP |
| `14ca3bf0` | clearing ci issues | WIP |
| `14bba27a` | granite and coverage | WIP |
| `0758890d` | working on the pr failures | WIP |
| `20f2332a` | improving coverage | WIP |
| `62c55fda` | improving test coverage | WIP |
| `cfc44cdd` | working on quality fixes | WIP |
| `bd4ac025` | fixed issues in ci build | WIP |
| `6ece3f38` | tests | WIP |
| `0bbd813f` | Merge remote-tracking branch 'origin/main' into grid-server | merge commit |
| `50f9866f` | clean up of larql-compute samples | WIP |
| `a0f6ff6c` | cleaned up larql-kv | WIP; kv refactored |
| `b1d22bd6` | models coverage | WIP |
| `151360f1` | coverage of metal now over 90% | Metal coverage; requires Metal extraction |
| `eaec33f9` | working on kv cache | WIP |
| `f638894c` | working on kv | WIP |
| `c467213f` | mega split for larql-compute-metal | Core Metal extraction; foundational but large |
| `f23b08ea` | mega split for larql-compute-metal | Duplicate/earlier version |
| `61f0f279` | working on kv unification | WIP |
| `fc2906a2` | kv unification | WIP |
| `e36c00d4` | working kv refactor | WIP |
| `90f61bce` | PERFORMANCE | WIP |
| `9d3e4ca5` | working on performance | WIP |
| `b5d04b9e` | working on test coverage | WIP |
| `c14886a4` | working on grid | WIP |
| `4e69a475` | Merge pull request #100 from chrishayuk/fix/lm-head-vocab-overflow-32bit | Merge; content already in fork |
| `b7a8627f` | Merge pull request #102 from chrishayuk/dependabot | CI deps |
| `5134e9d0` | Merge pull request #99 from chrishayuk/feat/android-aarch64-blas-gate | Already in fork |
| `bf113b4a` | ci(deps): bump the github-actions group | CI deps |
| `e2bfe4e0` | Merge pull request #101 from chrishayuk/chore/dependabot-github-actions | CI |
| `18b8fb86` | chore(ci): add Dependabot config for GitHub Actions | CI |
| `1611bb01` | working on server grid | WIP |
| `c3876e17` | Merge pull request #93 from chrishayuk/fix/hf-cache-scan-model-repos | Merge; content cherry-picked |

---

## Part 2: ianblenke/larql (ianblenke/main) — 597 commits

**Decision: ALL 597 commits DEFERRED**

### Rationale for blanket deferral

ianblenke's fork represents a major independent research track focused on:

1. **CUDA GPU backend** (100+ commits): NVIDIA cuBLAS, cudarc, PTX kernels, cuda-oxide
   migration — requires NVIDIA GPU hardware. Not available on metavacua's Chromebook.
   
2. **DeepSeek V4 (DSv4) inference** (150+ commits, stages 1–8h + variants):
   - Full DSv4 GGUF extraction pipeline, resident/streaming forward
   - MLA attention variants (NoCompress, Compress, Indexer)
   - Hash routing, sliding window attention, FP8 KV cache
   - Requires DSv4 Flash GGUF weights + CUDA hardware

3. **Speculative decoding** (40+ commits):
   - CPU oracle, GPU verify kernel, tree-mask attention
   - CUDA Graphs for spec batched forward
   - Requires CUDA hardware

4. **RotorQuant KV cache** (20+ commits):
   - Quantized KV format for compressed KV cache
   - CUDA-specific implementation (CUDA FFI wrappers)
   - Vendored RotorQuant CUDA provenance

5. **Qwen3.5 MoE** (80+ commits):
   - Gated DeltaNet recurrence, Q4/Q5/Q6 SIMD paths
   - CUDA GPU dispatch for Qwen3.6-35B-A3B
   - Requires CUDA hardware for GPU path

6. **Attention service routes** (15+ commits):
   - gRPC AttentionService + HTTP REST routes
   - Significant server infrastructure changes

7. **Performance optimization docs** (20+ commits):
   - Bench results, profiling notes, RESUME prompts
   - Documentation of empirical CUDA perf findings

### Selective note: potentially extractable (future PRs)

The following commits from ianblenke may be portable without CUDA and worth
revisiting in a future PR:

| Commit | Description | Condition |
|--------|-------------|-----------|
| `2f131862` | fix(ci): three pre-existing test failures | CI fix; may be independent |
| `6e9bd777` | ci(cli): re-enable clippy on larql-cli | CI fix; backlog of 82-112 errors gone |
| `751f6a91` | chore(fmt): cargo fmt --all | Formatting; may apply cleanly |
| `8bbefb23` | test(workspace): close the last build-target drift | Test infrastructure |
| `f86cc45d` | test(compute,vindex): fix trait drift + struct-field drift in test fixtures | Test infrastructure |
| `17da8e48` | fix(server): /v1/chat/completions deadlock | Fix for server deadlock (pick_template held read lock) |
| `b8192d45` | fix(compute): f16_to_f32 subnormal off-by-one — every subnormal decoded 2× too large | Correctness fix: **important** — but depends on ianblenke's compute structure |
| Various | `fix(qwen35):` prefix commits | Qwen3.5-specific fixes; not relevant until Qwen3.5 supported |
| `3c0c97aa` | Merge fix/encode-cached-ids-sync — fix main build + bring CI green | CI fix but requires ianblenke's server changes |

**Note on `b8192d45` (f16 subnormal bug):** This is a correctness fix (every f16
subnormal was decoded 2× too large). It may be independently applicable to the
`f16_to_f32` function in `larql-compute`. Should be evaluated as a separate PR.

### ianblenke commit categories (full enumeration)

**CUDA backend (commits 1-50, newest first):**
- `e5e2a905` feat(dsv4): CLI extraction command + serve-smoke script (#409)
- `39ab9702` feat(dsv4): serve DeepSeek-V4 through chat route (#408)
- `1ee30cc9` feat(dsv4-vindex): DsV4VindexMeta → DsV4Hyperparams reconstruction (#407)
- `ff8aae31` feat(dsv4-vindex): emit tokenizer.json + de-risk bootstrap (#406)
- `96a97c35` feat(dsv4-vindex): emit server-conforming index.json + embeddings.bin (#405)
- `dfeb65bc` feat(dsv4-vindex): serving reader — vindex → resident storage (#403)
- `49449396` chore: cargo fmt --all for rustfmt 1.9.0 / rust 1.95 (#402)
- `82cfba06` chore: DSv4 vs llama.cpp bench plan (#404)
- `32a95659` feat(dsv4-vindex): full-model extraction orchestrator + round-trip (#401)
- `84bf7d0d` feat(dsv4-vindex): dsv4_head.bin — head/embed round-trip (#400)
- `75dc0c25` feat(dsv4-vindex): dsv4_moe.bin — MoE/FFN + hash routing round-trip (#399)
- `c9f24fb3` feat(dsv4-vindex): dsv4_mhc.bin — mHC bookend round-trip (#398)
- `e2822ae4` feat(dsv4-vindex): dsv4_hca.bin — HCA compressor + indexer round-trip (#397)
- `d38ad1cf` feat(dsv4-vindex): dsv4_attn.bin wire format + round-trip (#396)
- `c05d384d` feat(vindex): V1 DSv4 config metadata (#395)
- `de2facbf` feat(vindex): V0 DSv4 capabilities gate (#394)
- `949b980d` docs(openspec): propose dsv4-vindex-extraction (#393)
- `2eeff9a5` docs(openspec): propose dsv4-server-serving (#392)
- `f6c7cc51` chore(openspec): archive dsv4-ondisk-prefix-cache (#391)
- `17e453b7` feat(dsv4): P4 prefix-cache generate wire-up + cold-vs-warm bench (#390)
- `26fe8446` feat(dsv4): P3 Full-SWA prefix-cache reuse + latent coff2 fix (#389)
- `438e747c` feat(dsv4): P2 prefix-keyed on-disk KV store (#388)
- `5873aeac` feat(dsv4): P1 KV-cache serialization wire format (#387)
- `3e092981` docs(openspec): propose dsv4-ondisk-prefix-cache (#386)
- `8ba4e896` perf(dsv4): finish resident-Q4_K HCA path — indexer wq_b + compressor (#385)
- `eeb8c1d6` perf(dsv4): resident-Q4_K shared-expert weights (#384)
- `7002bcab` test(dsv4): add attention raw tensors to resident-builder fixture (#383)
- `3dcbec44` perf(dsv4): resident-Q4_K attention weights + decode profiler (#382)
- `085e866d` feat(dsv4): HF-transformers parity harness (#381)
- `da01876f` chore(openspec): archive dsv4-quant-residency — P0-P4 complete (#380)
- `3fd8883e` perf(dsv4): P4 hybrid (attn→GPU / FFN→CPU) — 1.65× over all-CPU (#379)
- `e9c93763` perf(dsv4): bench resident mode — 44.8× faster decode than streaming (#378)
- `36180bce` feat(dsv4): P3 resident loader + end-to-end quant-vs-f32 parity (#377)
- `e915bb05` feat(dsv4): P3 resident non-streaming forward entry point (#376)
- `f6a978a8` feat(dsv4): P2 quant-aware routed-MoE dispatch (#375)
- `f4a36698` feat(dsv4): P1 resident layer builder (quantized routed experts) (#374)
- `4b58eb45` feat(dsv4): P1 raw expert-tensor reader for quant residency (#373)
- `cc3623b0` feat(dsv4): P1 dual-storage QuantTensor fields for routed MoE experts (#371)
- `09a4cb5f` fix(vindex): Mixtral moe_intermediate_size + wire summary-K SVD (#372)
- `93938037` test(dsv4): P0 audit for quant-residency — tensor types + expert_slice packing (#370)
- `0d55cfc3` docs(openspec): propose dsv4-quant-residency change (#369)
- `bd80f7c6` perf(cuda): device-resident f32 weight cache (opt-in, off by default) (#368)
- `36c54b4c` fix(inference): allocate internal KvCache when direct_all_layers skips dequant (#276)
- `06245ea8` feat(dsv4-bench): LARQL_DSV4_BENCH_VERBOSE=1 for per-step decode telemetry (#367)
- `92e6d9e4` feat(dsv4-bench): separate warmup from steady-state in decode timing (#366)
- `33de974a` feat(dsv4-bench): split prefill vs decode timing (#365)
- `ad59a7b3` feat(dsv4-bench): cpu vs cuda tok/s + VRAM benchmark test (#363)
- `5e348bde` docs(dsv4-gpu): update milestone-test docstring (#362)
- `bc4f65fa` perf(dsv4-gpu): use matmul_gpu for masked_attn's second GEMM (#361)
- `8ac45276` perf(dsv4-gpu): rayon-parallelize masked_attn softmax (#360)

**DSv4 GPU (commits 51-100):**
- `35d6dba1`..`c169c3cc` — 50 more DSv4 GPU perf commits (all CUDA, all deferred)

**DSv4 forward/cache implementation (commits 101-200):**
- `16f99833`..`cd3e9ea4` — 100 DSv4 forward implementation commits (all DSv4-specific, all deferred)

**Qwen3.5 + CUDA (commits 201-350):**
- `f13eff2c` perf(qwen35): rayon-parallel CPU attention scan in prefill (portable; deferred pending Qwen3.5 support)
- `857a17d0`..`dbc62670` — Qwen3.5 implementation commits (all deferred, requires Qwen3.5 family support)
- Multiple CUDA perf commits: all deferred (CUDA hardware required)

**Vindex Q-passthrough (commits ~185-200):**
- `a63da761` feat(vindex): Q4_K → Q4_K bit-passthrough in extract writer (potentially useful; deferred pending conflict analysis)
- `4d494c4e` feat(vindex): Q8_0 bit-passthrough (potentially useful; deferred)
- `092cd8dc` feat(vindex): Q5_K bit-passthrough (potentially useful; deferred)
- `28ce580b` feat(vindex): MoE expert gate_up + down byte-passthrough (potentially useful; deferred)

**Attention service routes (commits ~540-560):**
- All deferred — requires significant server infrastructure changes

**CI fixes (oldest commits ~580-597):**
- `05b4b307` ci: mkdir target/llvm-cov before cargo llvm-cov writes report (potentially clean)
- `7fea381e` ci: install libopenblas-dev in coverage workflow (CI divergence)
- `b1209e4a` fix(scripts): sort orphan_tests deterministically in spec-trace (deferred)

---

## Summary

### Incorporated from chrishayuk (this PR)

| # | SHA | Description |
|---|-----|-------------|
| 1 | `7f905ead` | fix(probe): signed thresholding (upstream 32f78fe2) |
| 2 | `a7771147` | fix(gguf): MoE expert_feed_forward_length fallback (upstream 8c816dca) |
| 3 | `2205b8ad` | fix(larql-vindex): u64 overflow guard doc (upstream 834d0659) |
| 4 | `ba3a7c91` | fix(gguf): deepseek_v4 arch string (upstream d0b915b9) |
| 5 | `9d3e1f82` | fix(cli): HF cache scan model-repo pulls (upstream a4ea55f1) |
| 6 | `bf45a868` | feat(mla): qk_nope/rope/v_head_dim fields (upstream 9b56cd2e) |
| 7 | `137cdea6` | feat(vindex): MLA absorption (upstream 2a1fc079) |
| 8 | `d7c9d02e` | feat(gqa): gqa_attention_asym (upstream d93797fe) |
| 9 | `8be6693a` | feat(ggml): Q3_K/Q5_K dequant (upstream f2a4c348) |
| 10 | `97d6494d` | feat(ggml): BitNet TQ1/TQ2/I2S (upstream ebc50c34+9467ae11) |
| 11 | `0d6a1d38` | feat(gguf): MLA metadata surface (upstream 8f1c8f3f) |
| 12 | `2ada1693` | feat(vindex): MLA wire (upstream 2d10daa8) |
| 13 | `c769b4b1` | fix(extract): GGUF file-or-dir input (upstream 58c849fa) |

### Deferred from chrishayuk (141 commits)

Root causes for deferral:
1. **kv_generate.rs deleted in fork** — 15+ commits touch this file (merged into different structure)
2. **larql-compute Metal extraction** — ~40 WIP commits; this large refactor (mega split) needs a dedicated PR after the fork's `larql-compute` structure is aligned
3. **Multi-modal** — 2 large PRs (Phase 1 + 2); conflict in 15+ files including deleted kv files
4. **Accuracy scoring** — accuracy_cmd.rs, bench/run.rs deleted in fork; needs investigation
5. **WIP commits** — 30+ "working on..." commits that are intermediate states
6. **Coverage policy divergence** — fork has different coverage baselines
7. **V1 falsification** — ROADMAP_STATUS.md and kquant_forward/hidden.rs deleted in fork

### Deferred from ianblenke (597 commits)

All 597 commits deferred. Root causes:
1. **CUDA hardware not available** (Chromebook Plus 2025, no NVIDIA GPU)
2. **DSv4 vindex format not in fork** (100+ commits require DSv4-specific schema)
3. **Different research track** — ianblenke is focused on CUDA GPU perf; fork is focused on wasm32v1-none/Chromebook

### Future work tracking

The following items from both upstreams are high-value and should be tracked as issues:

1. **chrishayuk: larql-compute Metal extraction** (PRs #109/#120) — foundational for larql-gpu design; track as issue
2. **chrishayuk: Multi-modal Phase 1+2** — significant capability gap; track as issue
3. **chrishayuk: GGUF streaming refactor** (c54875db) — maintainability improvement; track as issue  
4. **chrishayuk: V1 falsification** (ada668ef, closes KU4) — ROADMAP.md needs updating
5. **chrishayuk: MoE shards fix** (49403543) — correctness fix once pipeline_layer.rs is aligned
6. **chrishayuk: MoE KV engines** (part of b61836bc) — new capability once KV refactor lands
7. **ianblenke: f16 subnormal fix** (b8192d45) — correctness fix, potentially portable
8. **ianblenke: vindex Q-passthrough** (a63da761+4d494c4e+092cd8dc) — bit-accurate quantization preservation; potentially portable
9. **ianblenke: server deadlock fix** (17da8e48) — portability to be verified

---

## License

SPDX-License-Identifier: Apache-2.0  
(This document covers Apache-2.0 upstream contributions being ported)
