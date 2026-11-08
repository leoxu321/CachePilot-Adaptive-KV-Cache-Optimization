# CachePilot

**Benchmark local LLM inference configurations and select the fastest measured
configuration that meets a GPU memory budget and a perplexity constraint.**

CachePilot is an ML engineering portfolio project, not a new inference algorithm.
It compares FP16 weights with bitsandbytes NF4 weights on `Qwen/Qwen3-0.6B` while
holding the dynamic FP16 KV cache constant. All inference and evaluation run on
your local NVIDIA GPU. No paid API, cloud GPU, training, or performance promise.

## Status

Implemented: CLI, strict YAML configs, Transformers FP16/NF4 backend, isolated
workers, synchronized prefill/decode timings, GPU memory peaks, fixed-corpus
perplexity, constrained selection, raw JSON, CSV, Markdown reports, and CPU tests.

FP16 and NF4 inference and matched quality evaluation ran successfully on the
actual RTX 3060 Ti under native Windows. The 7 GiB / 5% smoke selection is **FP16**:
NF4 was slower and increased perplexity by 12.63% on the tiny fixed corpus.
See [validation notes](docs/VALIDATION.md) for the tested scope and
[the example report](examples/rtx3060ti-smoke/report.md) for preserved measurements.
The extended experiment is a supplied
configuration, not an implied measured result. Optional future work is in the
[roadmap](docs/ROADMAP.md).

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
- [Extended](configs/extended.yaml): 128/512/2048 prompt tokens, 64/128/128 output
  tokens, 10 repetitions each, 1024 quality targets. Not run automatically.

Prompt lengths include the complete non-thinking chat template. Prompt content is
synthetically constructed to the exact token count; IDs are saved. Generation is
greedy and fixed-length, ignoring EOS. This avoids unequal work when outputs differ.
Modify the YAML workload lengths, repetitions, seed, and evaluation budget to run
another measured workload. Unknown configuration keys are rejected. The bounded
MVP supports only this model, batch one, cuda:0, and the two named configurations.
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
| End-to-end latency | The timed inference loop: prefill + fixed-length decode; excludes load, tokenization, input transfer, and text decoding. |
| Peak allocated / reserved | PyTorch allocator peaks; reports include both. Selection uses maximum reserved memory across load, warmup, and measurement. |
| Perplexity change | `100 * (candidate PPL / FP16 PPL - 1)` on identical fixed text and evaluation rules. |

Timing summaries include medians, quartiles, IQR, sample standard deviation,
min/max, and counts. Each configuration/workload runs in its own subprocess.
Memory budgets use GiB (`2^30` bytes), despite the short `--memory-budget-gb` name.
They measure this process's PyTorch reserved memory, not all VRAM used by Windows,
CUDA libraries, and other programs. Leave practical headroom on the desktop.
No CPU/disk offloading or precision fallback is permitted.

Quality uses the bundled original English corpus, an explicit target-token
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

`recommend` filters measured configurations for one exact workload/suite by peak
reserved memory and relative perplexity increase, then minimizes median inference
latency. Missing quality makes a configuration infeasible. If none pass, it prints
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
