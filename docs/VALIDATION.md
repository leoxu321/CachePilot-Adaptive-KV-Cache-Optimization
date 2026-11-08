# Verified implementation status

Local validation date: **2026-11-08**. No cloud inference or paid APIs were used.

## Hardware and environment observed

- Windows 11 x64, OS build 26300; Python 3.12.6.
- NVIDIA GeForce RTX 3060 Ti, compute capability 8.6, driver 610.88.
- `nvidia-smi`: 8192 MiB VRAM; CUDA reports 8,589,279,232 usable total bytes.
- System RAM via Windows `GlobalMemoryStatusEx`: 34,253,627,392 bytes.
- CUDA PyTorch 2.10.0+cu128; Transformers 4.57.6; Accelerate 1.12.0;
  bitsandbytes 0.49.2. Full resolved package freeze is in
  [windows-tested.txt](../requirements/windows-tested.txt).
- System Python originally had CPU-only PyTorch; it was not modified. Project
  dependencies are in `.venv/`, model/download caches in `.cache/`.
- Desktop applications were left running. Native Windows FP16 and NF4 both work.
  WSL2 was not needed; WSL discovery was denied by the sandbox and was not pursued.

## Actual GPU experiment

FP16 was benchmarked successfully **before** NF4 was attempted. Both ran in
isolated child processes. Both quality evaluations then ran in isolated processes.
Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
Workload: batch one, 128 input tokens, exactly 16 output tokens, one warmup,
three measured repetitions. Quality: 256 target tokens, 128-token windows,
64-target stride, bundled corpus, same tokenization hash for both configurations.

| Configuration | Median E2E s | Median decode tok/s | Lifecycle reserved GiB | Perplexity | PPL change |
|---|---:|---:|---:|---:|---:|
| FP16 | 0.850866 | 18.9065 | 1.449219 | 45.9707 | baseline |
| NF4 | 0.998069 | 16.3999 | 1.271484 | 51.7780 | +12.6326% |

These are small smoke measurements, not claims about other workloads. NF4 reduced
this benchmark's memory peak but did not improve its latency or satisfy a 5%
perplexity tolerance. At **7 GiB / 5%**, selection returned **FP16**.
At **1.3 GiB / 15%**, it returned **NF4**. At **0.5 GiB / 5%**, it returned
**no feasible configuration**. Budgets use process PyTorch reserved memory, not
total board usage. See [the generated report](../examples/rtx3060ti-smoke/report.md)
for variability, provenance, and links to the unmodified raw measurements.

## Tests actually run

- 51 CPU tests passed: metrics, target coverage, strict config/result validation,
  constraints, mismatched protocols, ambiguity, pooling, process crash/timeout,
  FP16-first guards, and worker failure classification.
- Two opt-in GPU correctness tests passed on FP16: the manual cached decode
  produced the same token IDs as Transformers `generate`, and the explicit
  perplexity loss agreed with Transformers' built-in shifted label loss.
- `pip check` reported no broken requirements.
- Built a Python wheel successfully, including the bundled quality corpus.
- All five CLI commands were exercised against the real local experiment.

The GPU reference tests are not additional performance samples and do not enter
the report. Reproduce them with the model cached:

```powershell
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:CACHEPILOT_GPU_TESTS = '1'
.\.venv\Scripts\python -m pytest tests/test_gpu_integration.py -q
Remove-Item Env:CACHEPILOT_GPU_TESTS
```

## Implemented but not hardware-tested in this session

- Extended workloads and their 1024-target quality evaluation.
- Linux/WSL2 execution and CI runners (workflow supplied, not remotely executed).
- Actual GPU out-of-memory or unsupported-device failures. Error classification
  and subprocess failures were tested with controlled test doubles; the working
  GPU was not intentionally exhausted or reconfigured.

KV-cache quantization and the other roadmap items are **not implemented**.

## Reproduce the smoke experiment

Use the README setup and then:

```powershell
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
.\.venv\Scripts\cachepilot benchmark --config configs/reproduce-rtx3060ti.yaml
.\.venv\Scripts\cachepilot evaluate --config configs/reproduce-rtx3060ti.yaml
.\.venv\Scripts\cachepilot recommend --results results/reproduction --memory-budget-gb 7 --max-perplexity-increase-pct 5
.\.venv\Scripts\cachepilot report --results results/reproduction
```

Once the pinned snapshot is cached, `HF_HUB_OFFLINE=1` permits fully offline runs.
Use the same setting for benchmark and evaluation in a reproduction directory.
The preserved example directory is separate from ignored scratch `results/`.
No synthetic test records are included in it.
