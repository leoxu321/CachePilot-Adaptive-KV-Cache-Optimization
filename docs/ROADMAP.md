# Roadmap

## Implemented

- Transformers FP16 and bitsandbytes NF4 weights; constant dynamic FP16 KV cache.
- Isolated CUDA workers, explicit prefill/decode timing, repeated measurements.
- Fixed-corpus perplexity, validated raw JSON, CSV/Markdown reporting.
- Constrained selection of observed configurations for a measured workload.
- Homogeneous batched generation and aggregate decode throughput.
- Input-length and batch-size capacity search with resume and confirmations.
- Latency, throughput and largest-context recommendation objectives.
- Qwen2.5-1.5B-Instruct configuration using the same backend.

## Optional next phases (not implemented)

1. Add several natural prompts per workload and a larger, licensed evaluation
   corpus; randomize configuration ordering after the required FP16 preflight.
2. Add optional GPU telemetry and repeated sessions to study desktop interference,
   temperature, clock behavior, and confidence intervals.
3. **KV-cache quantization**: compare an existing supported implementation against
   the same FP16 KV baseline, with separate quality and context-length checks.
   Treat this as a new protocol and new experiment, not an assumed improvement.
4. Broaden model support only after testing chat templates, cache behavior,
   precision, quantization compatibility, and tokenization/evaluation rules.

Custom CUDA/Triton kernels, training, serving APIs, dashboards, learned selection
policies, and cloud GPUs are outside the MVP. No novel algorithm is required.
