# Roadmap

## Implemented MVP

- Transformers FP16 and bitsandbytes NF4 weights; constant dynamic FP16 KV cache.
- Isolated CUDA workers, explicit prefill/decode timing, repeated measurements.
- Fixed-corpus perplexity, validated raw JSON, CSV/Markdown reporting.
- Constrained selection of observed configurations for a measured workload.

## Optional next phases (not implemented)

1. Add several natural prompts per workload and a larger, licensed evaluation
   corpus; randomize configuration ordering after the required FP16 preflight.
2. Add optional GPU telemetry and repeated sessions to study desktop interference,
   temperature, clock behavior, and confidence intervals.
3. **KV-cache quantization**: compare an existing supported implementation against
   the same FP16 KV baseline, with separate quality and context-length checks.
   Treat this as a new protocol and new experiment, not an assumed improvement.
4. Support another small model only after testing its chat template, cache behavior,
   precision, quantization compatibility, and tokenization/evaluation rules.

Custom CUDA/Triton kernels, training, serving APIs, dashboards, learned selection
policies, and cloud GPUs are outside the MVP. No novel algorithm is required.
