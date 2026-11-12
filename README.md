# CachePilot

### Hardware-Aware LLM Inference Tuning

**Benchmark local LLM inference configurations and select the fastest measured
configuration that meets a GPU memory budget and a perplexity constraint.**

CachePilot is an ML engineering portfolio project, not a new inference algorithm.
It compares FP16 weights with bitsandbytes NF4 weights on `Qwen/Qwen3-0.6B` and
`Qwen/Qwen2.5-1.5B-Instruct` while
holding the dynamic FP16 KV cache constant. All inference and evaluation run on
your local NVIDIA GPU. No paid API, cloud GPU, training, or performance promise.

## Status

Implemented: CLI, strict YAML configs, Transformers FP16/NF4 backend, isolated
workers, synchronized prefill/decode timings, GPU memory peaks, fixed-corpus
perplexity, constrained selection, raw JSON, CSV, Markdown reports, and CPU tests.
Version 0.2 adds batched generation, resumable discrete capacity search, and three
explicit ranking objectives. Historical v0.1 results remain readable and unchanged.

FP16 and NF4 inference and matched quality evaluation ran successfully on the
actual RTX 3060 Ti under native Windows. The 7 GiB / 5% smoke selection is **FP16**:
NF4 was slower and increased perplexity by 12.63% on the tiny fixed corpus.
See [validation notes](docs/VALIDATION.md) for the tested scope and
[the example report](examples/rtx3060ti-smoke/report.md) for preserved measurements.
The full extended sweep is a supplied configuration, not an implied measured result.
See [the tuning audit](docs/TUNING_AUDIT.md) and [tuning methodology](docs/TUNING.md).
Optional future work is in the
[roadmap](docs/ROADMAP.md).

## Measured tuning results

<!-- measured-results:start -->
Generated from validated raw measurements; compare precisions within each workload/model only.

| Model / input x output | Config | Runs | Median E2E s [IQR] | Peak reserved GiB | PPL change |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B / 512 x 128 | fp16 | 10 | 7.077 [0.253] | 1.486 | +0.00% |
| Qwen3-0.6B / 512 x 128 | nf4 | 10 | 8.035 [0.195] | 1.309 | +16.75% |
| Qwen2.5-1.5B / 128 x 16 | fp16 | 3 | 0.741 [0.062] | 2.998 | +0.00% |
| Qwen2.5-1.5B / 128 x 16 | nf4 | 3 | 0.891 [0.010] | 1.539 | +8.49% |

Both experiments select FP16 at 7 GiB / 5% relative perplexity tolerance. The quality sets contain only 1,024 and 256 scored targets, respectively; these are local regression measurements, not broad quality or speedup claims.

[Qwen3 report](examples/rtx3060ti-tuning/representative/report.md) · [1.5B report](examples/rtx3060ti-tuning/qwen15b/report.md). Regenerate with `python scripts/update_readme_results.py`.
<!-- measured-results:end -->

The capacity validation confirmed 1,024 input tokens (batch one, 16 output tokens)
under an effective **1.6 GiB** budget for both precisions; the next tested size,
2,048, exceeded it. A separate batch search confirmed batch 4 under **1.9 GiB**,
which was the largest candidate tested, not a hardware maximum.
See [tuning validation](docs/TUNING_VALIDATION.md) and
[all tuning artifacts](examples/rtx3060ti-tuning/README.md).

## Capacity discovery and tuning

After setup, use the same shell/cache settings for all commands:

```powershell
.\.venv\Scripts\cachepilot capacity --config configs/capacity-smoke.yaml
.\.venv\Scripts\cachepilot evaluate --config configs/capacity-smoke.yaml
.\.venv\Scripts\cachepilot recommend --results results/tuning-batch --workload capacity-batch_size-4 --objective throughput --memory-budget-gb 1.9 --max-perplexity-increase-pct 5

# Input-length search; this is larger than the quick validation.
.\.venv\Scripts\cachepilot capacity --config configs/capacity.yaml
.\.venv\Scripts\cachepilot evaluate --config configs/capacity.yaml
.\.venv\Scripts\cachepilot recommend --results results/capacity --objective context --batch-size 1 --output-tokens 32 --memory-budget-gb 6.5 --max-perplexity-increase-pct 5
```

Set `capacity.axis` to `prompt_tokens` or `batch_size`, and provide a strictly
increasing discrete candidate list. The other dimensions stay fixed. The effective
budget is `min(configured budget, detected GPU capacity) - safety margin`, in GiB.
Exponential bracketing followed by binary refinement finds a boundary, checks
neighbors, and confirms the selected size in additional fresh processes.
Exceeding the memory budget is infeasible even without an OOM. Unsupported features,
timeouts and unrelated runtime errors stop the search as **inconclusive**.

Run the same command again to resume: completed attempts, including failures, are
not repeated. `capacity.json` is atomically updated after each attempt and preserves
the resolved model revision. Changing config, relevant environment or source code
requires a new `results_dir`. Completed raw files are checked against saved hashes.
An OS file lock prevents two controllers from sharing a search directory. A hung
worker is terminated at `timeout_seconds`; its reason and log are retained.

The search assumes approximately monotonic feasibility. It verifies adjacent
candidates but does not prove monotonicity over every untested size. A successful
largest candidate is a **lower bound**, not a discovered hardware maximum. Capacity
is separate from quality: run `evaluate` before requesting a quality-constrained
recommendation. Pass the effective budget (after margin) to `recommend`.

`recommend --objective` accepts:

- `latency`: lowest median end-to-end inference latency for one matched workload.
- `throughput`: highest median aggregate decode tokens/s for one matched workload,
  including the same batch size and output length.
- `context`: largest measured feasible prompt length, holding batch size and output
  length fixed. It does not compare latency across different prompt lengths.

Memory and relative perplexity constraints are applied before ranking. Missing
quality is unverified, never passing. Ties use configuration name, then workload
ID. Output includes the winner's samples and alternatives with rejection reasons.

## Setup: Windows PowerShell

Tested environment: Windows 11 x64, Python 3.12.6, RTX 3060 Ti (8192 MiB reported
by `nvidia-smi`), about 32 GiB system RAM. The original system PyTorch was CPU-only;
the project uses its own virtual environment. Native Windows CUDA and NF4 are
supported by the chosen libraries, so **WSL2 is not required** for this MVP.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e '.[inference,test]'

# Keep downloaded model files in the project. Set this in each new shell.
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'

.\.venv\Scripts\cachepilot doctor
.\.venv\Scripts\cachepilot benchmark --config configs/smoke.yaml --only fp16
# Attempt NF4 only after that FP16 benchmark succeeds.
.\.venv\Scripts\cachepilot benchmark --config configs/smoke.yaml --only nf4
.\.venv\Scripts\cachepilot evaluate --config configs/smoke.yaml
.\.venv\Scripts\cachepilot recommend --results results/ --memory-budget-gb 7 --max-perplexity-increase-pct 5
.\.venv\Scripts\cachepilot report --results results/
```

After activating the venv, the equivalent commands are simply `cachepilot ...`.
`python -m cachepilot ...` is also supported. Run `cachepilot --help` or
`cachepilot benchmark --help` for options. From a fresh results directory, a single
`cachepilot benchmark --config configs/smoke.yaml` runs FP16 first and only permits
NF4 after successful FP16. A failed baseline cannot silently produce a comparison.

The first setup downloads a roughly 2.9 GB CUDA PyTorch wheel, dependencies, and a
roughly 1.5 GB model weight file. Allow disk space for both the downloads and
installed/cached copies. Internet is used for public packages/model files only.
The smoke workload has one warmup and three measured repetitions per configuration
plus 256 quality targets. Cached runs are small; first-time installation/downloads
are not part of its inference measurements.

For analysis-only use, `pip install -e '.[test]'` is sufficient to validate results,
generate reports, and run CPU tests. Inference dependencies are an explicit extra
so installing the CLI does not silently install the wrong PyTorch build.

## Linux / WSL2 fallback

Use WSL2 Ubuntu if native Windows CUDA or bitsandbytes does not work on another
machine, or if you want to repeat the experiment under Linux. This project's
sandbox could not query WSL status; no distro was installed or changed.
Use the NVIDIA Windows driver with WSL CUDA support, then inside the distro:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[inference,test]'
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
cachepilot doctor
cachepilot benchmark --config configs/smoke.yaml --only fp16
cachepilot benchmark --config configs/smoke.yaml --only nf4
cachepilot evaluate --config configs/smoke.yaml
```

Never reuse a Windows venv inside WSL. Results across different operating systems
are separate comparison suites. `doctor` reports OS, RAM, package versions, CUDA
runtime, driver, GPU name/capacity/UUID, and available memory. It exits nonzero if
CUDA PyTorch or required inference packages are unavailable. Seeing a GPU in
`nvidia-smi` alone is insufficient.

## Workloads and experiments

- [Smoke](configs/smoke.yaml): 128 prompt tokens, 16 output tokens, 3 repetitions.
- [Extended](configs/extended.yaml): 512/2048/4096 prompt tokens, 128 output
  tokens, 10 repetitions each, 1024 quality targets. Not run automatically.

Prompt lengths include the complete non-thinking chat template. Prompt content is
synthetically constructed to the exact token count; IDs are saved. Generation is
greedy and fixed-length, ignoring EOS. This avoids unequal work when outputs differ.
Modify the YAML workload lengths, repetitions, seed, and evaluation budget to run
another measured workload. Unknown configuration keys are rejected. The bounded
implementation supports the two named models, configurable homogeneous batches,
cuda:0, and the two named precision configurations.
Paths in YAML are relative to the current working directory.

```powershell
.\.venv\Scripts\cachepilot benchmark --config configs/extended.yaml
.\.venv\Scripts\cachepilot evaluate --config configs/extended.yaml
.\.venv\Scripts\cachepilot recommend --results results/extended --workload medium-512x128 --memory-budget-gb 7 --max-perplexity-increase-pct 5
```

Multiple workload shapes require `--workload`; multiple environments/protocols
require `--suite` using the full ID in the report. A workload name that maps to
several different shapes is also ambiguous; use its full workload ID.

## Metrics and quality

| Metric | Meaning |
|---|---|
| TTFT | Synchronized time from device-resident prompt to first selected GPU token. |
| Decode throughput | Remaining `N - 1` tokens divided by post-first-token time; excludes prefill. |
| Aggregate decode throughput | `batch_size * (N - 1) / decode_seconds`; all sequences generate exactly N tokens. |
| End-to-end latency | The timed inference loop: prefill + fixed-length decode; excludes load, tokenization, input transfer, and text decoding. |
| Peak allocated / reserved | PyTorch allocator peaks; reports include both. Selection uses maximum reserved memory across load, warmup, and measurement. |
| Perplexity change | `100 * (candidate PPL / FP16 PPL - 1)` on identical fixed text and evaluation rules. |

Timing summaries include medians, quartiles, IQR, sample standard deviation,
min/max, and counts. Each configuration/workload runs in its own subprocess.
Memory budgets use GiB (`2^30` bytes), despite the short `--memory-budget-gb` name.
They measure this process's PyTorch reserved memory, not all VRAM used by Windows,
CUDA libraries, and other programs. Leave practical headroom on the desktop.
No CPU/disk offloading or precision fallback is permitted.

Quality uses the fixed `cachepilot-prose-v1` original English corpus, an explicit target-token
budget, and overlapping context windows. Each target is scored once. It is a tiny
regression signal, not a broad accuracy benchmark. Generated-text equality is
not used for quality. See [methodology](docs/METHODOLOGY.md) for exact boundaries,
tokenization, loss accounting, aggregation rules, caveats, and official sources.

## Results and selection

```text
results/
  <run-id>.result.json    # Raw samples, status, configuration and provenance
  requests/<run-id>.json # Exact worker request
  logs/<run-id>.log      # Model/library output or exception traceback
  summary.csv           # Successful benchmark comparisons
  report.md             # Tables, quality, environment and failure explanations
```

Benchmark/evaluate automatically refresh summary files. Raw files are never
overwritten by a later invocation. OOM, unsupported configurations, errors,
timeouts, and skipped comparisons are explicit statuses. Invalid result files
cause analysis to fail with a filename; they are not silently discarded.

`recommend` filters measured configurations by peak reserved memory and relative
perplexity increase before applying the selected objective. Missing quality makes
a configuration infeasible. If none pass, it prints
`no feasible configuration` and exits 2; errors exit 1 and success exits 0.
The JSON selection includes contributing run IDs. It never predicts unmeasured
performance. NF4 can save memory while being slower than FP16 on a small model.

## Reproducibility

The four primary inference dependencies are pinned in `pyproject.toml`. A complete
tested environment freeze is saved in `requirements/windows-tested.txt`. Install
CUDA PyTorch first, then use that freeze as constraints if reproducing this host:

```powershell
.\.venv\Scripts\python -m pip install -e '.[inference,test]' -c requirements/windows-tested.txt
```

Model `revision: main` is resolved to an immutable commit once per command; the
actual loaded commit is checked and recorded. To repeat a published run, replace
`main` in your config with its recorded 40-character SHA (the example reproduction
config does this). With that snapshot already cached, set
`$env:HF_HUB_OFFLINE = '1'` to run without network. Keep settings/environment
consistent across benchmark and evaluate; fingerprints prevent incompatible joins.
Random seeds are recorded; reproducibility does not mean identical timings under
different desktop load, clocks, driver versions, or operating systems.

## Architecture and tests

```text
CLI -> strict Config -> controller -> one subprocess per config/workload
                                       -> Transformers CUDA backend
                                       -> raw JSON + log
raw JSON -> validation -> matched comparison rows -> CSV / Markdown / recommendation
```

`backend.py` owns model loading, token construction, timing, and perplexity.
`worker.py` owns a single model lifetime and provenance. `runner.py` resolves
revisions, enforces FP16-first execution, and handles process failure.
`capacity.py` orchestrates search, compatibility checks and the resume journal.
`metrics.py` contains pure calculations; `results.py` validates and selects;
`report.py` emits artifacts without requiring GPU dependencies.

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pip check
```

Tests cover metric boundaries, variability, weighted perplexity, exact target
coverage, malformed results, incompatible comparisons, memory/quality constraints,
ambiguous workloads, and no-feasible outcomes. Test fixtures are synthetic and
remain inside tests; they are never published as measured results.

## Portfolio scope

The contribution is a reproducible measurement and decision pipeline with explicit
failure handling and validity limits. Describe only experiments actually run and
cite their raw artifacts. KV-cache quantization, custom kernels, training,
serving, dashboards, and learned selection policies are not implemented. See the
[roadmap](docs/ROADMAP.md) for optional follow-up work.
