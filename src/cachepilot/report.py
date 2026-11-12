import csv
import json
from pathlib import Path

from .results import comparison_rows, load_results


def write_report(directory):
    root = Path(directory)
    records = load_results(root)
    rows = comparison_rows(records)
    fields = ["suite_id", "workload_id", "workload", "configuration", "batch_size", "prompt_tokens", "output_tokens",
              "aggregate_decode_median_tokens_per_s", "n", "ttft_median_ms",
              "ttft_iqr_ms", "decode_median_tokens_per_s", "decode_iqr_tokens_per_s",
              "e2e_median_s", "e2e_iqr_s", "peak_allocated_gib", "peak_reserved_gib",
              "perplexity", "perplexity_increase_pct", "run_ids"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"suite_id": row["suite_id"], "workload_id": row["workload_id"],
                "workload": row["workload"], "configuration": row["configuration"],
                "batch_size": row["batch_size"], "prompt_tokens": row["prompt_tokens"], "output_tokens": row["output_tokens"],
                "aggregate_decode_median_tokens_per_s": row["aggregate_decode_tokens_per_s"]["median"],
                "n": row["e2e_s"]["n"], "ttft_median_ms": row["ttft_s"]["median"] * 1000,
                "ttft_iqr_ms": row["ttft_s"]["iqr"] * 1000,
                "decode_median_tokens_per_s": row["decode_tokens_per_s"]["median"],
                "decode_iqr_tokens_per_s": row["decode_tokens_per_s"]["iqr"],
                "e2e_median_s": row["e2e_s"]["median"], "e2e_iqr_s": row["e2e_s"]["iqr"],
                "peak_allocated_gib": row["peak_allocated_bytes"] / 2**30,
                "peak_reserved_gib": row["peak_reserved_bytes"] / 2**30,
                "perplexity": row["perplexity"], "perplexity_increase_pct": row["perplexity_increase_pct"],
                "run_ids": ";".join(row["run_ids"])})
    lines = ["# CachePilot measured comparison", "",
             "Only validated local measurements appear below. Missing quality is not a passing result.", "",
             "Times show median [IQR]. Memory is the maximum across load, warmup, and measured runs.",
             "The budget uses PyTorch reserved memory, not total device usage; desktop/driver allocations are excluded.", ""]
    for suite in sorted({r["suite_id"] for r in rows}):
        example = next(r for r in records if r.get("suite_id") == suite and r["status"] == "ok")
        env = example["environment"]
        lines += [f"## Suite `{suite}`", "", f"Model: `{example['config']['model']}` at `{example['model_revision']}`.",
                  f"GPU: {env['gpu']['name']}; {env['gpu']['total_memory_bytes'] / 2**30:.2f} GiB. OS: {env['os']}.",
                  f"Packages: `{env['packages']}`.", "",
                  f"Attention/settings: `{example['settings']}`.", "",
                  "| Workload | Config | Batch | n | TTFT ms [IQR] | Aggregate decode tok/s [IQR] | E2E s [IQR] | Alloc GiB | Reserved GiB | PPL | PPL change % |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in [r for r in rows if r["suite_id"] == suite]:
            def stat(key, scale=1):
                return f"{row[key]['median'] * scale:.3f} [{row[key]['iqr'] * scale:.3f}]"
            def optional(value):
                return "not evaluated" if value is None else f"{value:.4f}"
            name = row["workload"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {row['configuration']} | {row['batch_size']} | {row['e2e_s']['n']} | {stat('ttft_s', 1000)} | "
                f"{stat('aggregate_decode_tokens_per_s')} | {stat('e2e_s')} | {row['peak_allocated_bytes']/2**30:.3f} | "
                f"{row['peak_reserved_bytes']/2**30:.3f} | {optional(row['perplexity'])} | {optional(row['perplexity_increase_pct'])} |")
        lines += ["", "Workload IDs:", ""]
        for workload_id in sorted({r["workload_id"] for r in rows if r["suite_id"] == suite}):
            name = next(r["workload"] for r in rows if r["workload_id"] == workload_id)
            lines.append(f"- {name}: `{workload_id}`")
        quality = [r for r in records if r["kind"] == "evaluate" and r["status"] == "ok" and r["suite_id"] == suite]
        if quality:
            protocol = quality[0]["quality"]["protocol"]
            lines += ["", f"Quality: {protocol['token_budget']} scored targets; context {protocol['context_tokens']}; "
                      f"stride {protocol['stride']}; corpus SHA256 `{protocol['corpus_sha256']}`."]
        lines += ["", "Raw evidence (UTC timestamps):", ""]
        for record in sorted((r for r in records if r.get("suite_id") == suite and r["status"] == "ok"),
                             key=lambda r: r["created_at"]):
            filename = f"{record['run_id']}.result.json"
            lines.append(f"- [{record['kind']}/{record['configuration']}]({filename}) — "
                         f"{record['created_at']}; load {record['load_s']:.3f} s.")
        lines.append("")
    if not rows:
        lines += ["No successful benchmark measurements available.", ""]
    capacity_path = root / "capacity.json"
    if capacity_path.exists():
        state = json.loads(capacity_path.read_text(encoding="utf-8"))
        identity = state["identity"]
        lines += ["## Capacity search", "", f"Status: **{state['status']}**. "
                  f"Axis: `{identity['config']['capacity']['axis']}`. Effective reserved-memory budget "
                  f"after safety margin/device cap: {identity['effective_budget_bytes']/2**30:.3f} GiB.", "",
                  "Search assumes approximately monotonic feasibility; adjacent candidates are checked. "
                  "A range-exhausted result is only a lower bound on hardware capacity. Capacity alone does not verify quality.", ""]
        if state.get("error"):
            lines += [f"Stopped: {state['error']}", ""]
        for configuration, boundary in state["boundaries"].items():
            lines.append(f"- {configuration}: confirmed selected {boundary['selected']}; next infeasible "
                         f"{boundary['next_infeasible']}; range exhausted: {boundary['range_exhausted']}.")
        lines += ["", "| Configuration | Candidate | Confirmation | Feasible | Reason | Raw run |",
                  "|---|---:|---:|---|---|---|"]
        for attempt in state["attempts"].values():
            run = attempt.get("run_id")
            evidence = f"[{run}]({run}.result.json)" if run else "not executed (context constraint)"
            lines.append(f"| {attempt['configuration']} | {attempt['candidate']} | {attempt['repeat']} | "
                         f"{attempt.get('feasible', 'pending')} | {attempt.get('reason', 'interrupted')} | {evidence} |")
        lines.append("")
    failures = [r for r in records if r["status"] != "ok"]
    if failures:
        lines += ["## Failed or skipped attempts", ""]
        for record in failures:
            lines.append(f"- `{record['run_id']}` {record['kind']}/{record['configuration']}: "
                         f"**{record['status']}** — {record['error']['message']}")
        lines.append("")
    lines += ["## Interpretation", "",
              "This is a local engineering measurement, not a claim of a general speedup. "
              "The small quality corpus is a regression check, not a task-accuracy benchmark. "
              "A smoke run is too small for robust performance conclusions. Desktop activity, clocks, "
              "temperature, and fixed FP16-first ordering can influence results.", "",
              "Repeated successful invocations in the same suite/workload are pooled. Memory uses the "
              "largest observed peak; quality uses the largest observed perplexity per configuration. "
              "JSON retains individual samples, load time, warmups, generated IDs, protocol, and provenance.", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
