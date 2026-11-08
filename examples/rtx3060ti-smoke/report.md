# CachePilot measured comparison

Only validated local measurements appear below. Missing quality is not a passing result.

Times show median [IQR]. Memory is the maximum across load, warmup, and measured runs.
The budget uses PyTorch reserved memory, not total device usage; desktop/driver allocations are excluded.

## Suite `84ba77c399c15665012ad60f284966847e97a8581cdee3fa4f702a0ba630a783`

Model: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
GPU: NVIDIA GeForce RTX 3060 Ti; 8.00 GiB. OS: Windows-11-10.0.26300-SP0.
Packages: `{'cachepilot': '0.1.0', 'torch': '2.10.0+cu128', 'transformers': '4.57.6', 'accelerate': '1.12.0', 'bitsandbytes': '0.49.2', 'huggingface-hub': '0.36.2', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.5.4', 'PyYAML': '6.0.3'}`.

| Workload | Config | n | TTFT ms [IQR] | Decode tok/s [IQR] | E2E s [IQR] | Alloc GiB | Reserved GiB | PPL | PPL change % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke-128x16 | fp16 | 3 | 57.488 [0.731] | 18.906 [0.230] | 0.851 [0.010] | 1.403 | 1.449 | 45.9707 | 0.0000 |
| smoke-128x16 | nf4 | 3 | 83.429 [1.218] | 16.400 [0.234] | 0.998 [0.014] | 0.869 | 1.271 | 51.7780 | 12.6326 |

Workload IDs:

- smoke-128x16: `11634d8c758d6d25f8f9e9ebe253b8ebe049fd040dc8a81908de2729086c4d8e`

Quality: 256 scored targets; context 128; stride 64; corpus SHA256 `d6ef29523203aa8d93c7365b1d34d79bec15c3c3b8cdfa94df34aed438ba83ef`.

Raw evidence (UTC timestamps):

- [benchmark/fp16](da76ea7b305c4623bdc672507ba4df8c.result.json) — 2026-11-08T22:32:53.822224+00:00; load 19.978 s.
- [benchmark/nf4](ce6b08fc905845869c1290b12bed5f05.result.json) — 2026-11-08T22:35:19.264505+00:00; load 6.479 s.
- [evaluate/fp16](fce10b2d139b4340b73f5605227a69e5.result.json) — 2026-11-08T22:38:08.334121+00:00; load 4.966 s.
- [evaluate/nf4](09562938cf67459ab13361c38246c2dc.result.json) — 2026-11-08T22:38:17.095446+00:00; load 6.271 s.

## Interpretation

This is a local engineering measurement, not a claim of a general speedup. The small quality corpus is a regression check, not a task-accuracy benchmark. A smoke run is too small for robust performance conclusions. Desktop activity, clocks, temperature, and fixed FP16-first ordering can influence results.

Repeated successful invocations in the same suite/workload are pooled. Memory uses the largest observed peak; quality uses the largest observed perplexity per configuration. JSON retains individual samples, load time, warmups, generated IDs, protocol, and provenance.
