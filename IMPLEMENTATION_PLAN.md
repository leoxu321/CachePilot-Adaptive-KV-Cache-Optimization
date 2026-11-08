# CachePilot implementation plan

1. Inspect host and official documentation; use native Windows CUDA if supported.
2. Build a vertical slice: validated configuration, isolated FP16 worker, fixed token
   workload, synchronized timings, environment metadata, raw JSON.
3. Run FP16 before enabling bitsandbytes NF4 with the same FP16 dynamic KV cache.
4. Add fixed-corpus perplexity, strict comparison identities, constrained selection,
   CSV and Markdown reports, and focused CPU tests.
5. Run the smoke comparison on the actual GPU if dependencies/model downloads work;
   document verified results and exact reproduction commands. Keep extended runs
   and optional future KV quantization clearly separate.

## Initial inspection (2026-11-08)

- Repository initially contained only README.md; no repository agent instructions.
- Windows 11 x64, Python 3.12.6, system PyTorch 2.10.0+cpu.
- NVIDIA RTX 3060 Ti, 8192 MiB, driver 610.88; approximately 2402 MiB already used
  by desktop applications. Do not terminate these applications.
- WMI and WSL discovery were denied in the sandbox; RAM and WSL status need a
  different check. CUDA runtime availability is distinct from nvidia-smi visibility.
- Use a repository-local virtual environment and caches; leave system Python alone.

## Compatibility sources checked

- https://huggingface.co/Qwen/Qwen3-0.6B (Qwen3 and enable_thinking=False)
- https://huggingface.co/docs/transformers/quantization/bitsandbytes
- https://huggingface.co/docs/bitsandbytes/installation (Windows x64, CUDA 12.8,
  sm86, NF4 supported)
- https://pytorch.org/get-started/locally/ (Windows CUDA wheels)

These sources support the chosen approach; actual compatibility will be verified
by running it. No speedup, quality threshold, or benchmark outcome is assumed.
