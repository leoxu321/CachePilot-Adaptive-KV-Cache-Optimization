# CachePilot measured comparison

Only validated local measurements appear below. Missing quality is not a passing result.

Times show median [IQR]. Memory is the maximum across load, warmup, and measured runs.
The budget uses PyTorch reserved memory, not total device usage; desktop/driver allocations are excluded.

## Suite `bd5506a323f5af49d9378b650ec216c0e782b4991e17d74b117468e772c01a7a`

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 3060 Ti; 8.00 GiB. OS: Windows-11-10.0.26300-SP0.
Packages: `{'cachepilot': '0.2.0', 'torch': '2.10.0+cu128', 'transformers': '4.57.6', 'accelerate': '1.12.0', 'bitsandbytes': '0.49.2', 'huggingface-hub': '0.36.2', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.5.4', 'PyYAML': '6.0.3'}`.

Attention/settings: `{'backend': 'transformers', 'attention': 'sdpa', 'kv_cache': 'dynamic', 'kv_dtype': 'float16', 'compute_dtype': 'float16', 'generation': 'greedy', 'enable_thinking': False, 'stop_on_eos': False, 'cpu_offload': False, 'device': 'cuda:0', 'tf32': False, 'prompt_recipe': 'synthetic-chat-v1', 'measurement_version': 2}`.

| Workload | Config | Batch | n | TTFT ms [IQR] | Aggregate decode tok/s [IQR] | E2E s [IQR] | Alloc GiB | Reserved GiB | PPL | PPL change % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| capacity-batch_size-4 | fp16 | 4 | 6 | 62.122 [3.991] | 65.824 [2.262] | 0.983 [0.034] | 1.403 | 1.486 | 45.9707 | 0.0000 |
| capacity-batch_size-4 | nf4 | 4 | 6 | 89.888 [9.757] | 47.213 [5.336] | 1.361 [0.171] | 0.869 | 1.309 | 51.7780 | 12.6326 |
| capacity-batch_size-2 | fp16 | 2 | 3 | 61.516 [0.597] | 35.122 [0.299] | 0.915 [0.008] | 1.403 | 1.465 | 45.9707 | 0.0000 |
| capacity-batch_size-2 | nf4 | 2 | 3 | 91.690 [5.751] | 23.214 [0.743] | 1.378 [0.045] | 0.869 | 1.291 | 51.7780 | 12.6326 |
| capacity-batch_size-1 | fp16 | 1 | 3 | 58.807 [2.634] | 17.160 [0.036] | 0.936 [0.003] | 1.403 | 1.449 | 45.9707 | 0.0000 |
| capacity-batch_size-1 | nf4 | 1 | 3 | 84.868 [3.390] | 15.695 [0.165] | 1.046 [0.011] | 0.869 | 1.271 | 51.7780 | 12.6326 |

Workload IDs:

- capacity-batch_size-4: `3d63e191d41cfdbec9e0e6d4231bcc8bcd074814502999dd05e2d54271ab5e6e`
- capacity-batch_size-2: `6584a288842e3595b2eb16e02aad5f5d5abecfd53e76386514e487900051e560`
- capacity-batch_size-1: `ec36c3c2d980afc19d126b2e9fdc9b75dd130d0ea9cf492b3a167a8c71e0bb4a`

Quality: 256 scored targets; context 128; stride 64; corpus SHA256 `d6ef29523203aa8d93c7365b1d34d79bec15c3c3b8cdfa94df34aed438ba83ef`.

Raw evidence (UTC timestamps):

- [benchmark/fp16](6c65cc7f21b64e86b68ee081ca9098b6.result.json) — 2026-11-08T23:56:38.818976+00:00; load 5.151 s.
- [benchmark/fp16](1dcfd9ac90f04ee9a305d7d5375a0b63.result.json) — 2026-11-08T23:56:51.804063+00:00; load 5.073 s.
- [benchmark/fp16](034c15588caf4d559140ae8a534dc406.result.json) — 2026-11-08T23:57:04.635941+00:00; load 5.138 s.
- [benchmark/fp16](10e8ebd00d594784a0fd3f5b8afd511f.result.json) — 2026-11-08T23:57:17.521837+00:00; load 5.028 s.
- [benchmark/nf4](54fc655426e9468fa11949b7dd14e2d7.result.json) — 2026-11-08T23:57:30.378392+00:00; load 6.404 s.
- [benchmark/nf4](34e74594ae2c4747975f1759913dd7f0.result.json) — 2026-11-08T23:57:45.084377+00:00; load 6.639 s.
- [benchmark/nf4](bc44783221d64a1d995bef9e2b016de0.result.json) — 2026-11-08T23:58:01.671184+00:00; load 6.319 s.
- [benchmark/nf4](0014d3b7b0d3448cb84cf8bddbe137cc.result.json) — 2026-11-08T23:58:17.456974+00:00; load 7.455 s.
- [evaluate/fp16](8beb004870814479b69a36fa9c5b3119.result.json) — 2026-11-08T23:59:09.752470+00:00; load 5.402 s.
- [evaluate/nf4](c0f653638e3c42bebea5a4bb67819c90.result.json) — 2026-11-08T23:59:19.964427+00:00; load 7.879 s.

## Capacity search

Status: **complete**. Axis: `batch_size`. Effective reserved-memory budget after safety margin/device cap: 1.900 GiB.

Search assumes approximately monotonic feasibility; adjacent candidates are checked. A range-exhausted result is only a lower bound on hardware capacity. Capacity alone does not verify quality.

- fp16: confirmed selected 4; next infeasible None; range exhausted: True.
- nf4: confirmed selected 4; next infeasible None; range exhausted: True.

| Configuration | Candidate | Confirmation | Feasible | Reason | Raw run |
|---|---:|---:|---|---|---|
| fp16 | 1 | 0 | True | within_budget | [6c65cc7f21b64e86b68ee081ca9098b6](6c65cc7f21b64e86b68ee081ca9098b6.result.json) |
| fp16 | 2 | 0 | True | within_budget | [1dcfd9ac90f04ee9a305d7d5375a0b63](1dcfd9ac90f04ee9a305d7d5375a0b63.result.json) |
| fp16 | 4 | 0 | True | within_budget | [034c15588caf4d559140ae8a534dc406](034c15588caf4d559140ae8a534dc406.result.json) |
| fp16 | 4 | 1 | True | within_budget | [10e8ebd00d594784a0fd3f5b8afd511f](10e8ebd00d594784a0fd3f5b8afd511f.result.json) |
| nf4 | 1 | 0 | True | within_budget | [54fc655426e9468fa11949b7dd14e2d7](54fc655426e9468fa11949b7dd14e2d7.result.json) |
| nf4 | 2 | 0 | True | within_budget | [34e74594ae2c4747975f1759913dd7f0](34e74594ae2c4747975f1759913dd7f0.result.json) |
| nf4 | 4 | 0 | True | within_budget | [bc44783221d64a1d995bef9e2b016de0](bc44783221d64a1d995bef9e2b016de0.result.json) |
| nf4 | 4 | 1 | True | within_budget | [0014d3b7b0d3448cb84cf8bddbe137cc](0014d3b7b0d3448cb84cf8bddbe137cc.result.json) |

## Interpretation

This is a local engineering measurement, not a claim of a general speedup. The small quality corpus is a regression check, not a task-accuracy benchmark. A smoke run is too small for robust performance conclusions. Desktop activity, clocks, temperature, and fixed FP16-first ordering can influence results.

Repeated successful invocations in the same suite/workload are pooled. Memory uses the largest observed peak; quality uses the largest observed perplexity per configuration. JSON retains individual samples, load time, warmups, generated IDs, protocol, and provenance.
