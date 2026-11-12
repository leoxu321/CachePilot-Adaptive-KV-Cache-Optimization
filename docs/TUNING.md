# Capacity and recommendation protocol (v0.2)

## Workload matching and batch accounting

Schema v2 retains the audited v1 timing boundaries. Batch size now belongs to the
workload, not the suite identity. All batch rows contain the same synthetic prompt
IDs; this is a homogeneous shape benchmark, not heterogeneous request batching.
Each row generates exactly N tokens. Per-sequence decode rate is `(N-1)/decode_s`;
aggregate rate is `B*(N-1)/decode_s`. The interval begins after all first tokens
are selected and CUDA synchronized, ending after all final tokens are selected
and synchronized. No prefill tokens or first generated tokens enter the numerator.
TTFT measures completion of the first token for the whole batch.

Schema v1 remains supported without rewriting its original files. V1 and v2 cannot
be joined into one suite. A v2 suite fixes model revision, device/software,
generation/cache/attention settings, seed and quality protocol; its workloads fix
exact prompt hash, prompt/output lengths and batch size. Precision descriptors
explicitly record FP16 or NF4 (FP16 compute/KV, uint8 storage, no double quantization).
Validation checks actual serialized NF4 quantization settings before quality joins.
Quality is evaluated at batch one, independently of inference batch; it is a
weight-precision regression check, not batch-specific generated-output accuracy.

The current benchmark retains one synchronization at the first-token boundary.
It avoids per-token synchronization, hooks and CPU transfers. That extra boundary
can affect latency and is a measurement limitation; results describe this timed
loop rather than an uninstrumented serving implementation. Warmup, allocator
behavior, CUDA synchronization and lifecycle peaks follow [v1 methodology](METHODOLOGY.md).

## Capacity workflow

`capacity` searches one axis over ordered discrete candidates. Every measured
candidate and each boundary confirmation is a separate worker containing the
configured warmups and measured repetitions. Effective reserved-memory allowance:

`min(memory_budget_gb * 2^30, detected_total_memory_bytes) - safety_margin_gb * 2^30`

This is not a measurement of all board usage. Desktop, CUDA context and driver
allocations still need headroom. The margin is explicit, not an automatic estimate
of Windows display usage. Native WDDM may page CUDA allocations; explicit device
placement cannot rule out driver paging.

Candidates whose input **plus output** length exceeds the pinned model's actual
`max_position_embeddings` are recorded as context-infeasible without a worker.
Other candidates run and use the maximum reserved peak across loading, warmup and
measured repetitions. A successful CUDA run over budget is memory-infeasible.
CUDA OOM is separately labeled. An unsupported operation, exception, timeout or
invalid result stops search as inconclusive, never as an inferred size boundary.

Search probes indices 0,1,3,7,... to bracket the boundary, then binary-searches
indices. It checks the selected predecessor, successor and the candidate beyond
the first infeasible point when available. Independent selected-boundary runs must
agree. An observed feasibility reversal or failed confirmation makes the result
inconclusive. The approximately monotonic assumption is stated in every boundary.
No claims are made for unsampled gaps or beyond the largest configured candidate.

`capacity.json` stores identities, attempts, raw hashes and boundaries. Atomic
replacement protects journal updates; OS advisory locks release on controller
exit/death. Before launching a candidate its run ID is saved. A completed raw
result surviving an interrupted journal update is adopted on resume. Incomplete
attempts without a final raw file are rerun; identical completed attempts are not.
Worker timeouts kill and reap the worker and preserve a timeout status/log. A
recorded unrelated failure is reused on resume; after fixing it, choose a new
results directory for a deliberate retry.

Identity includes full config, resolved model SHA, model context limit, effective
budget, stable hardware/software environment, precision/protocol settings and a
digest of package source files. Model `main` is resolved once for the search and
stays pinned when resuming. Changing settings or code refuses resume. Transient
free VRAM is recorded by workers but is not part of resume compatibility.

## Quality and ranking

The unchanged corpus bytes are named `cachepilot-prose-v1` and guarded by SHA256
`d6ef29523203aa8d93c7365b1d34d79bec15c3c3b8cdfa94df34aed438ba83ef`.
Token budgets select a prefix; too-large budgets error instead of repeating text.
The NF4 audit is in [TUNING_AUDIT.md](TUNING_AUDIT.md). No protocol was changed to
improve its observed quality. Changing corpus bytes requires a new version/hash.

Latency/throughput ranking requires one exact workload ID. Context ranking permits
prompt lengths to differ but requires one batch size/output length and suite.
Every objective first filters memory and matched perplexity tolerance; missing
quality is unverified. Only actual successful benchmark records are candidates.
Each winner includes contributing run IDs, measured metrics and quality evidence.
Ties use configuration name then workload ID, never incidental filesystem order.
Feasible non-winners are reported as ranked behind the winner; constraint failures
list their reasons. No feasible candidate returns exit 2 and no selection.
Any failed matched attempt, including a failed boundary confirmation, makes that
configuration/workload unverified in the results directory. Successful reruns do
not erase it. Use a fresh experiment directory after investigating a transient
failure. Failures without a successful numeric row are also listed as alternatives.

Report tables pool successful repetitions per exact suite/workload/configuration,
including confirmations, and use maximum observed memory. Raw files retain each
attempt. Historical quality reruns use maximum observed perplexity per precision;
the reference remains the measured FP16 value, not a quality guarantee.

## Remaining experiment commands

With the project venv installed and HF_HOME pointing at the local cache:

```powershell
cachepilot benchmark --config configs/extended.yaml
cachepilot evaluate --config configs/extended.yaml
cachepilot recommend --results results/extended --objective latency --workload long-4096x128 --memory-budget-gb 7 --max-perplexity-increase-pct 5
cachepilot capacity --config configs/capacity.yaml
cachepilot evaluate --config configs/capacity.yaml
cachepilot recommend --results results/capacity --objective context --memory-budget-gb 6.5 --max-perplexity-increase-pct 5
cachepilot benchmark --config configs/qwen15b.yaml
cachepilot evaluate --config configs/qwen15b.yaml
```

For a larger batch search, copy `capacity-smoke.yaml`, extend `candidates`, set the
desired fixed input/output lengths and budget, and use a fresh `results_dir`.
KV-cache quantization remains optional future work. No custom kernels, training,
serving system, dashboard, learned policy or cloud hardware is included.
