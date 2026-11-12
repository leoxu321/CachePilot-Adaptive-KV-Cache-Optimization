"""Regenerate the small README table from validated, real example artifacts."""

from pathlib import Path

from cachepilot.results import comparison_rows, load_results


def main():
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples" / "rtx3060ti-tuning"
    lines = ["Generated from validated raw measurements; compare precisions within each workload/model only.", "",
             "| Model / input x output | Config | Runs | Median E2E s [IQR] | Peak reserved GiB | PPL change |",
             "|---|---|---:|---:|---:|---:|"]
    for directory, label in (("representative", "Qwen3-0.6B / 512 x 128"),
                             ("qwen15b", "Qwen2.5-1.5B / 128 x 16")):
        rows = comparison_rows(load_results(examples / directory))
        for row in rows:
            latency = row["e2e_s"]
            change = row["perplexity_increase_pct"]
            quality = "unverified" if change is None else f"{change:+.2f}%"
            lines.append(f"| {label} | {row['configuration']} | {latency['n']} | "
                         f"{latency['median']:.3f} [{latency['iqr']:.3f}] | "
                         f"{row['peak_reserved_bytes']/2**30:.3f} | {quality} |")
    lines += ["", "Both experiments select FP16 at 7 GiB / 5% relative perplexity tolerance. "
              "The quality sets contain only 1,024 and 256 scored targets, respectively; "
              "these are local regression measurements, not broad quality or speedup claims.", "",
              "[Qwen3 report](examples/rtx3060ti-tuning/representative/report.md) · "
              "[1.5B report](examples/rtx3060ti-tuning/qwen15b/report.md). "
              "Regenerate with `python scripts/update_readme_results.py`."]
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    begin, end = "<!-- measured-results:start -->", "<!-- measured-results:end -->"
    prefix, rest = text.split(begin, 1)
    _, suffix = rest.split(end, 1)
    path.write_text(prefix + begin + "\n" + "\n".join(lines) + "\n" + end + suffix, encoding="utf-8")
    print("Updated README from preserved real measurements")


if __name__ == "__main__":
    main()
