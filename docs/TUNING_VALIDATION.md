# Hardware-aware tuning validation

Final verification: **2026-11-11**. The GPU experiments completed on the evening
of 2026-11-08 America/New_York (raw timestamps are UTC). All experiments ran on
the local RTX 3060 Ti under native Windows CUDA. No driver or system-wide CUDA
installation was changed. Existing v0.1 artifacts remain unchanged.

## Implemented and verified

- Batched Transformers inference with FP16 dynamic KV cache and explicit CUDA
  placement; FP16 and bitsandbytes NF4 weight configurations.
- Resumable input-length and batch-size capacity search: discrete exponential
  bracketing/binary refinement, safety margins, context limits including output
  tokens, adjacent checks and independent boundary confirmations.
- Atomic journals, source/environment/config compatibility checks, OS locks,
  completed-result reuse, orphaned completed-result recovery and worker timeouts.
- Latency, aggregate decode throughput and largest measured context objectives;
  matched workload rules, quality gating and deterministic tie-breaking.
- Failed matched attempts make recommendations unverified, including an OOM or
  timeout during confirmation. A prior success cannot hide such a failure.
- Versioned quality corpus, schema v1 backward compatibility, validated raw JSON,
  CSV and Markdown reports, two model configurations and a generated README table.

## Actual capacity measurements

The quick batch search used 128 input tokens, 16 output tokens, one warmup and
three measured runs per worker. Candidates were 1, 2 and 4; the effective memory
budget was 1.9 GiB (2 minus a 0.1 GiB margin). Both FP16 and NF4 passed every
candidate and an additional fresh-process confirmation at batch 4. This is a
tested lower bound. A second invocation resumed without launching new workers.

The input-length search used batch one, 16 output tokens, the same repetition
counts, and candidates 128, 256, 512, 1024, 2048 and 4096. Its effective allowance
was 1.6 GiB (1.65 minus 0.05). Both precisions confirmed **1024**; **2048** and
**4096** exceeded the reserved-memory budget despite completing without OOM.
The search correctly distinguished budget infeasibility from CUDA failure.
At the 5% quality tolerance, the largest-context recommendation is FP16/1024.

Sources: [batch report](../examples/rtx3060ti-tuning/batch-capacity/report.md),
[context report](../examples/rtx3060ti-tuning/context-capacity/report.md).
Boundary confirmation samples remain separate raw runs and are pooled in summaries.

## Bounded representative experiment and second model

Before launch, the 512-input/128-output experiment was estimated at 3–5 minutes
for both precisions, including loading and two warmups plus ten measurements each.
The raw worker start timestamps span about 3.6 minutes through the final quality
worker start, consistent with that estimate. It completed successfully with
1,024 quality targets, context 256 and stride 128.

- Qwen3 FP16: median latency **7.077 s**, peak reserved **1.486 GiB**, PPL **39.223**.
- Qwen3 NF4: median latency **8.035 s**, peak reserved **1.309 GiB**, PPL **45.791**,
  relative increase **16.75%**.

The second model, `Qwen/Qwen2.5-1.5B-Instruct`, was loaded at pinned revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. FP16 completed before NF4. Both benchmark
and quality evaluation succeeded: 128 input / 16 output, one warmup, three measured
runs, 256 quality targets.

- 1.5B FP16: median latency **0.741 s**, peak reserved **2.998 GiB**, PPL **21.820**.
- 1.5B NF4: median latency **0.891 s**, peak reserved **1.539 GiB**, PPL **23.673**,
  relative increase **8.49%**.

Both model experiments select FP16 at 7 GiB / 5%. Different models and different
workloads are not ranked against each other. Reports include IQR, TTFT, throughput,
allocated and reserved peaks, model revision, environment and raw evidence links:
[representative](../examples/rtx3060ti-tuning/representative/report.md),
[1.5B smoke](../examples/rtx3060ti-tuning/qwen15b/report.md).

## Checks

- 80 CPU tests passed, covering old functionality, every monotonic boundary on
  several grid sizes, logarithmic termination, nonmonotonic/unstable observations,
  OOM versus other failures, budget/margins, resume, compatibility and objectives.
- Six opt-in GPU correctness tests passed for Qwen3: both precisions and batches
  one/two match explicitly greedy Transformers generation; sliding-window quality
  agrees with independently masked built-in label loss for both precisions.
- The original +12.63% NF4 smoke PPL was reproduced with identical evaluation
  hashes and independently checked loss. No corpus or evaluation rules were
  adjusted to obtain a more favorable result. See [audit](TUNING_AUDIT.md).
- `pip check` passed. Original archived result files still validate.
- The v0.2 wheel built successfully with its corpus included. The six GPU checks
  were rerun successfully during final verification on 2026-11-11.

## Limits and remaining work

These are local measurements with desktop programs running, short synthetic
prompts and a tiny prose quality set. The batch grid stops at 4. The 1.6 GiB input
boundary is an artificial budget boundary, not exhaustion of the 8 GiB device.
Actual GPU OOM was not forced; OOM, timeout and unrelated failures are covered with
controlled test doubles. WDDM paging and non-PyTorch allocations remain limitations.
The single first-token synchronization remains inside latency measurement.

The full 512/2048/4096-by-128, ten-repetition sweep, 7 GiB capacity search and
Linux/WSL execution were **not** run. CI is configured but remote CI completion is
not claimed. KV-cache quantization remains planned, not implemented.

Remaining commands from the project root (after README setup):

```powershell
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
.\.venv\Scripts\cachepilot benchmark --config configs/extended.yaml
.\.venv\Scripts\cachepilot evaluate --config configs/extended.yaml
.\.venv\Scripts\cachepilot recommend --results results/extended --objective latency --workload long-4096x128 --memory-budget-gb 7 --max-perplexity-increase-pct 5
.\.venv\Scripts\cachepilot capacity --config configs/capacity.yaml
.\.venv\Scripts\cachepilot evaluate --config configs/capacity.yaml
.\.venv\Scripts\cachepilot recommend --results results/capacity --objective context --memory-budget-gb 6.5 --max-perplexity-increase-pct 5
```
