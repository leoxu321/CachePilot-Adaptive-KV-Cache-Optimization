# Real tuning validation

These directories contain actual local RTX 3060 Ti measurements, not test fixtures.
Original worker JSON files and capacity journals were copied without editing.
Reports and recommendations are regenerated from those raw files.

- [Batch capacity](batch-capacity/report.md): 128 input / 16 output, batches 1/2/4,
  effective reserved-memory allowance 1.9 GiB, one fresh-process boundary
  confirmation. Both variants reached the upper grid limit of 4; this is a lower
  bound on capacity, not a maximum. Recommendation also requires quality.
- [Context capacity](context-capacity/report.md): batch one / 16 output,
  128/256/512/1024/2048/4096 input tokens, effective allowance 1.6 GiB.
  Both variants confirmed 1024; 2048 and 4096 exceeded the memory budget without OOM.
  Matching quality artifacts are reused from the batch experiment, since model,
  revision, precision, dataset/evaluation settings, hardware and software agree.
- [Representative experiment](representative/report.md): Qwen3-0.6B, batch one,
  512 input / 128 output, two warmups and ten measured repetitions per precision;
  1,024 quality targets.
- [Second model](qwen15b/report.md): Qwen2.5-1.5B-Instruct, batch one,
  128 input / 16 output, three measured repetitions and 256 quality targets.

The batch search was rerun to verify resume without any new worker launches.
The original journals retain the exact code digest at measurement time. Later
selection-only changes deliberately make those old journals incompatible for
resuming with the final source tree; they remain valid measured/report artifacts.
The shipped validation configurations use fresh `results/tuning-batch` and
`results/tuning-context` directories for future runs. No old journal was rewritten
to pretend the code digest had not changed.

Desktop applications were left running. The second model download overlapped some
capacity probes; these are functional capacity checks, not isolated lab timing
claims. Longer performance and quality measurements are stored separately.

The [historical v0.1 smoke experiment](../rtx3060ti-smoke/report.md) is unchanged.
