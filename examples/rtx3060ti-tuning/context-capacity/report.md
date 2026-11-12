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
| capacity-prompt_tokens-4096 | fp16 | 1 | 3 | 1508.908 [61.928] | 15.867 [2.310] | 2.451 [0.229] | 4.090 | 4.240 | 45.9707 | 0.0000 |
| capacity-prompt_tokens-4096 | nf4 | 1 | 3 | 1510.681 [55.093] | 14.038 [0.883] | 2.579 [0.129] | 3.499 | 3.750 | 51.7780 | 12.6326 |
| capacity-prompt_tokens-512 | fp16 | 1 | 3 | 74.614 [16.377] | 13.641 [0.295] | 1.175 [0.011] | 1.403 | 1.486 | 45.9707 | 0.0000 |
| capacity-prompt_tokens-512 | nf4 | 1 | 3 | 107.827 [3.395] | 12.809 [0.449] | 1.283 [0.042] | 0.869 | 1.309 | 51.7780 | 12.6326 |
| capacity-prompt_tokens-1024 | fp16 | 1 | 6 | 162.328 [4.743] | 14.356 [0.735] | 1.214 [0.050] | 1.445 | 1.475 | 45.9707 | 0.0000 |
| capacity-prompt_tokens-1024 | nf4 | 1 | 6 | 166.555 [10.398] | 13.462 [0.203] | 1.282 [0.030] | 0.869 | 1.250 | 51.7780 | 12.6326 |
| capacity-prompt_tokens-2048 | fp16 | 1 | 3 | 460.252 [28.566] | 11.558 [1.248] | 1.754 [0.187] | 2.038 | 2.117 | 45.9707 | 0.0000 |
| capacity-prompt_tokens-2048 | nf4 | 1 | 3 | 447.354 [0.770] | 13.232 [0.506] | 1.581 [0.042] | 1.448 | 1.752 | 51.7780 | 12.6326 |
| capacity-prompt_tokens-256 | fp16 | 1 | 3 | 68.152 [0.864] | 15.974 [0.188] | 1.008 [0.012] | 1.403 | 1.465 | 45.9707 | 0.0000 |
| capacity-prompt_tokens-256 | nf4 | 1 | 3 | 105.982 [4.819] | 12.928 [0.862] | 1.269 [0.081] | 0.869 | 1.291 | 51.7780 | 12.6326 |
| capacity-prompt_tokens-128 | fp16 | 1 | 3 | 68.063 [1.380] | 15.876 [0.117] | 1.014 [0.008] | 1.403 | 1.449 | 45.9707 | 0.0000 |
| capacity-prompt_tokens-128 | nf4 | 1 | 3 | 102.086 [25.378] | 12.863 [0.787] | 1.261 [0.094] | 0.869 | 1.271 | 51.7780 | 12.6326 |

Workload IDs:

- capacity-prompt_tokens-4096: `2703c167befe8a73ca46f7861aa5563680d5fdd0dd6a2f606334e33d1f796d37`
- capacity-prompt_tokens-512: `3d117233dbc3b7b6fc3d99b8fea4a6b226855f82d8e493e72b854827e2a20710`
- capacity-prompt_tokens-1024: `7b33a4b9c51e61bc59408db4d60de0cd95137d2bcfdd1678ccecd09443d3853c`
- capacity-prompt_tokens-2048: `836c9f5748f08c597b8cc9c1d00e495617280f203d782f16280b4e82cd92a386`
- capacity-prompt_tokens-256: `cc3b14430326edccea8da54e504d4394844aa7c407018e023c8d85821d3a86bb`
- capacity-prompt_tokens-128: `ec36c3c2d980afc19d126b2e9fdc9b75dd130d0ea9cf492b3a167a8c71e0bb4a`

Quality: 256 scored targets; context 128; stride 64; corpus SHA256 `d6ef29523203aa8d93c7365b1d34d79bec15c3c3b8cdfa94df34aed438ba83ef`.

Raw evidence (UTC timestamps):

- [evaluate/fp16](8beb004870814479b69a36fa9c5b3119.result.json) — 2026-11-08T23:59:09.752470+00:00; load 5.402 s.
- [evaluate/nf4](c0f653638e3c42bebea5a4bb67819c90.result.json) — 2026-11-08T23:59:19.964427+00:00; load 7.879 s.
- [benchmark/fp16](a80b09e4f20749f393d28aca19d94dee.result.json) — 2026-11-08T23:59:39.045759+00:00; load 5.669 s.
- [benchmark/fp16](71d46ca4f01c44a5a9f013bfeb1f2075.result.json) — 2026-11-08T23:59:52.798269+00:00; load 5.201 s.
- [benchmark/fp16](caab9440e6704e0a922e343d771d71f5.result.json) — 2026-11-09T00:00:05.882577+00:00; load 5.680 s.
- [benchmark/fp16](bd13577986dc4d54a2d470efe4c4cab8.result.json) — 2026-11-09T00:00:20.548936+00:00; load 5.406 s.
- [benchmark/fp16](791f51821ca84497910e0eab173f8705.result.json) — 2026-11-09T00:00:40.888786+00:00; load 6.502 s.
- [benchmark/fp16](4cc1315f5ece4a01b07250362433c8f8.result.json) — 2026-11-09T00:01:00.051032+00:00; load 6.044 s.
- [benchmark/fp16](9659c650d08c4df3adfb0981f90c3afd.result.json) — 2026-11-09T00:01:15.424794+00:00; load 5.839 s.
- [benchmark/nf4](e808375b1dc0482b9d41266b7ffdb03f.result.json) — 2026-11-09T00:01:30.684984+00:00; load 8.004 s.
- [benchmark/nf4](0ce2a61defb84a3a9a416959f09baa5d.result.json) — 2026-11-09T00:01:48.170352+00:00; load 7.614 s.
- [benchmark/nf4](0eea7d2ca94a40a5a997dafcac94e598.result.json) — 2026-11-09T00:02:05.299835+00:00; load 7.075 s.
- [benchmark/nf4](51ae630e25a54f3493e721dd2ff8b5f0.result.json) — 2026-11-09T00:02:21.475265+00:00; load 7.064 s.
- [benchmark/nf4](ea42c5e7b3fd431695ea431ca24f0f88.result.json) — 2026-11-09T00:02:43.088270+00:00; load 7.056 s.
- [benchmark/nf4](8ea7beae40904554814a64cc54feb41e.result.json) — 2026-11-09T00:03:00.404033+00:00; load 6.961 s.
- [benchmark/nf4](d3ce086d445e48cbba15e776bf6eb96d.result.json) — 2026-11-09T00:03:16.845328+00:00; load 7.576 s.

## Capacity search

Status: **complete**. Axis: `prompt_tokens`. Effective reserved-memory budget after safety margin/device cap: 1.600 GiB.

Search assumes approximately monotonic feasibility; adjacent candidates are checked. A range-exhausted result is only a lower bound on hardware capacity. Capacity alone does not verify quality.

- fp16: confirmed selected 1024; next infeasible 2048; range exhausted: False.
- nf4: confirmed selected 1024; next infeasible 2048; range exhausted: False.

| Configuration | Candidate | Confirmation | Feasible | Reason | Raw run |
|---|---:|---:|---|---|---|
| fp16 | 128 | 0 | True | within_budget | [a80b09e4f20749f393d28aca19d94dee](a80b09e4f20749f393d28aca19d94dee.result.json) |
| fp16 | 256 | 0 | True | within_budget | [71d46ca4f01c44a5a9f013bfeb1f2075](71d46ca4f01c44a5a9f013bfeb1f2075.result.json) |
| fp16 | 1024 | 0 | True | within_budget | [caab9440e6704e0a922e343d771d71f5](caab9440e6704e0a922e343d771d71f5.result.json) |
| fp16 | 4096 | 0 | False | memory_budget | [bd13577986dc4d54a2d470efe4c4cab8](bd13577986dc4d54a2d470efe4c4cab8.result.json) |
| fp16 | 2048 | 0 | False | memory_budget | [791f51821ca84497910e0eab173f8705](791f51821ca84497910e0eab173f8705.result.json) |
| fp16 | 512 | 0 | True | within_budget | [4cc1315f5ece4a01b07250362433c8f8](4cc1315f5ece4a01b07250362433c8f8.result.json) |
| fp16 | 1024 | 1 | True | within_budget | [9659c650d08c4df3adfb0981f90c3afd](9659c650d08c4df3adfb0981f90c3afd.result.json) |
| nf4 | 128 | 0 | True | within_budget | [e808375b1dc0482b9d41266b7ffdb03f](e808375b1dc0482b9d41266b7ffdb03f.result.json) |
| nf4 | 256 | 0 | True | within_budget | [0ce2a61defb84a3a9a416959f09baa5d](0ce2a61defb84a3a9a416959f09baa5d.result.json) |
| nf4 | 1024 | 0 | True | within_budget | [0eea7d2ca94a40a5a997dafcac94e598](0eea7d2ca94a40a5a997dafcac94e598.result.json) |
| nf4 | 4096 | 0 | False | memory_budget | [51ae630e25a54f3493e721dd2ff8b5f0](51ae630e25a54f3493e721dd2ff8b5f0.result.json) |
| nf4 | 2048 | 0 | False | memory_budget | [ea42c5e7b3fd431695ea431ca24f0f88](ea42c5e7b3fd431695ea431ca24f0f88.result.json) |
| nf4 | 512 | 0 | True | within_budget | [8ea7beae40904554814a64cc54feb41e](8ea7beae40904554814a64cc54feb41e.result.json) |
| nf4 | 1024 | 1 | True | within_budget | [d3ce086d445e48cbba15e776bf6eb96d](d3ce086d445e48cbba15e776bf6eb96d.result.json) |

## Interpretation

This is a local engineering measurement, not a claim of a general speedup. The small quality corpus is a regression check, not a task-accuracy benchmark. A smoke run is too small for robust performance conclusions. Desktop activity, clocks, temperature, and fixed FP16-first ordering can influence results.

Repeated successful invocations in the same suite/workload are pooled. Memory uses the largest observed peak; quality uses the largest observed perplexity per configuration. JSON retains individual samples, load time, warmups, generated IDs, protocol, and provenance.
