# Real local smoke experiment

The four `*.result.json` files are unmodified outputs from local Windows GPU
workers on 2026-11-08. They contain three measured repetitions for each of FP16
and NF4 plus one fixed-corpus quality evaluation for each configuration.
No synthetic test fixtures are included.

See [report.md](report.md) for comparison tables and links to raw evidence.
`recommendation.json` records the 7 GiB / 5% selection.

Regenerate the report or recommendation without a GPU:

```powershell
cachepilot report --results examples/rtx3060ti-smoke
cachepilot recommend --results examples/rtx3060ti-smoke --memory-budget-gb 7 --max-perplexity-increase-pct 5
```

Repeat the measurements with `configs/reproduce-rtx3060ti.yaml` and the tested
dependency constraints in `requirements/windows-tested.txt`. Reproduction writes
to `results/reproduction`, preserving this example. Exact timing is sensitive to
desktop activity and other environmental conditions; this is a small smoke test.
