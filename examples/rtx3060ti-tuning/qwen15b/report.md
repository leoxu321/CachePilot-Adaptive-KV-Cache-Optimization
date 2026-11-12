# CachePilot measured comparison

Only validated local measurements appear below. Missing quality is not a passing result.

Times show median [IQR]. Memory is the maximum across load, warmup, and measured runs.
The budget uses PyTorch reserved memory, not total device usage; desktop/driver allocations are excluded.

## Suite `468b6d2495c5892983e02b8e1a9a7201e164aefba57fe0112e8ff64934635b7a`

Model: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
GPU: NVIDIA GeForce RTX 3060 Ti; 8.00 GiB. OS: Windows-11-10.0.26300-SP0.
Packages: `{'cachepilot': '0.2.0', 'torch': '2.10.0+cu128', 'transformers': '4.57.6', 'accelerate': '1.12.0', 'bitsandbytes': '0.49.2', 'huggingface-hub': '0.36.2', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.5.4', 'PyYAML': '6.0.3'}`.

Attention/settings: `{'backend': 'transformers', 'attention': 'sdpa', 'kv_cache': 'dynamic', 'kv_dtype': 'float16', 'compute_dtype': 'float16', 'generation': 'greedy', 'enable_thinking': False, 'stop_on_eos': False, 'cpu_offload': False, 'device': 'cuda:0', 'tf32': False, 'prompt_recipe': 'synthetic-chat-v1', 'measurement_version': 2}`.

| Workload | Config | Batch | n | TTFT ms [IQR] | Aggregate decode tok/s [IQR] | E2E s [IQR] | Alloc GiB | Reserved GiB | PPL | PPL change % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke-128x16 | fp16 | 1 | 3 | 51.229 [0.944] | 21.761 [1.667] | 0.741 [0.062] | 2.919 | 2.998 | 21.8201 | 0.0000 |
| smoke-128x16 | nf4 | 1 | 3 | 76.242 [1.733] | 18.484 [0.215] | 0.891 [0.010] | 1.438 | 1.539 | 23.6730 | 8.4916 |

Workload IDs:

- smoke-128x16: `337fd41a04aa86bf7633c5959f91e84c2d16f881cc0135ef0981f6161f6527b0`

Quality: 256 scored targets; context 128; stride 64; corpus SHA256 `d6ef29523203aa8d93c7365b1d34d79bec15c3c3b8cdfa94df34aed438ba83ef`.

Raw evidence (UTC timestamps):

- [benchmark/fp16](855157f65df6472e8c78a96345d361a4.result.json) — 2026-11-09T00:09:36.239184+00:00; load 5.368 s.
- [benchmark/nf4](baa64ab3d1774ec386e72c4ccfe5fc1f.result.json) — 2026-11-09T00:09:48.556260+00:00; load 7.069 s.
- [evaluate/fp16](e47da73effd04a03a6d66e866cadd5ab.result.json) — 2026-11-09T00:10:05.739499+00:00; load 5.199 s.
- [evaluate/nf4](2d98cd6d24d04db48147674f399a0cba.result.json) — 2026-11-09T00:10:14.616384+00:00; load 6.989 s.

## Interpretation

This is a local engineering measurement, not a claim of a general speedup. The small quality corpus is a regression check, not a task-accuracy benchmark. A smoke run is too small for robust performance conclusions. Desktop activity, clocks, temperature, and fixed FP16-first ordering can influence results.

Repeated successful invocations in the same suite/workload are pooled. Memory uses the largest observed peak; quality uses the largest observed perplexity per configuration. JSON retains individual samples, load time, warmups, generated IDs, protocol, and provenance.
