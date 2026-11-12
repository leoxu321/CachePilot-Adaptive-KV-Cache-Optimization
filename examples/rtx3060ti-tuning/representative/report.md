# CachePilot measured comparison

Only validated local measurements appear below. Missing quality is not a passing result.

Times show median [IQR]. Memory is the maximum across load, warmup, and measured runs.
The budget uses PyTorch reserved memory, not total device usage; desktop/driver allocations are excluded.

## Suite `1c7157cc81de3d3053e2804b2fdb73aa0f05268953706938120508a499848a25`

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 3060 Ti; 8.00 GiB. OS: Windows-11-10.0.26300-SP0.
Packages: `{'cachepilot': '0.2.0', 'torch': '2.10.0+cu128', 'transformers': '4.57.6', 'accelerate': '1.12.0', 'bitsandbytes': '0.49.2', 'huggingface-hub': '0.36.2', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.5.4', 'PyYAML': '6.0.3'}`.

Attention/settings: `{'backend': 'transformers', 'attention': 'sdpa', 'kv_cache': 'dynamic', 'kv_dtype': 'float16', 'compute_dtype': 'float16', 'generation': 'greedy', 'enable_thinking': False, 'stop_on_eos': False, 'cpu_offload': False, 'device': 'cuda:0', 'tf32': False, 'prompt_recipe': 'synthetic-chat-v1', 'measurement_version': 2}`.

| Workload | Config | Batch | n | TTFT ms [IQR] | Aggregate decode tok/s [IQR] | E2E s [IQR] | Alloc GiB | Reserved GiB | PPL | PPL change % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| medium-512x128 | fp16 | 1 | 10 | 84.940 [12.626] | 18.146 [0.696] | 7.077 [0.253] | 1.403 | 1.486 | 39.2228 | 0.0000 |
| medium-512x128 | nf4 | 1 | 10 | 127.526 [6.451] | 16.062 [0.390] | 8.035 [0.195] | 0.869 | 1.309 | 45.7912 | 16.7463 |

Workload IDs:

- medium-512x128: `9d2e34e42a0db581fe672153854e31c7fcf690c3c55e366a95ad75268a77fe43`

Quality: 1024 scored targets; context 256; stride 128; corpus SHA256 `d6ef29523203aa8d93c7365b1d34d79bec15c3c3b8cdfa94df34aed438ba83ef`.

Raw evidence (UTC timestamps):

- [benchmark/fp16](44cd3f8b167f457f8e329fe928f61602.result.json) — 2026-11-09T00:05:51.444859+00:00; load 4.565 s.
- [benchmark/nf4](aaf37de0e3444184af25471281a07498.result.json) — 2026-11-09T00:07:24.913859+00:00; load 5.973 s.
- [evaluate/fp16](c43f3cfebeb84c7992828e7a65be22a4.result.json) — 2026-11-09T00:09:14.939725+00:00; load 4.443 s.
- [evaluate/nf4](2c1ef083b4e64859a50ec3dc7a9758d7.result.json) — 2026-11-09T00:09:23.191323+00:00; load 5.949 s.

## Interpretation

This is a local engineering measurement, not a claim of a general speedup. The small quality corpus is a regression check, not a task-accuracy benchmark. A smoke run is too small for robust performance conclusions. Desktop activity, clocks, temperature, and fixed FP16-first ordering can influence results.

Repeated successful invocations in the same suite/workload are pooled. Memory uses the largest observed peak; quality uses the largest observed perplexity per configuration. JSON retains individual samples, load time, warmups, generated IDs, protocol, and provenance.
