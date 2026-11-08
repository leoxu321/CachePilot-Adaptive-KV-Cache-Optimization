import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cachepilot", description="Benchmark local CUDA inference and select measured configurations.")
    parser.add_argument("--version", action="version", version="CachePilot 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect OS, RAM, GPU and installed inference dependencies")
    for name in ("benchmark", "evaluate"):
        command = sub.add_parser(name, help="Measure fixed-length generation" if name == "benchmark" else "Evaluate fixed-corpus perplexity")
        command.add_argument("--config", required=True, help="YAML configuration file")
        command.add_argument("--only", choices=["fp16", "nf4"], help="Run one enabled configuration (FP16 must succeed first)")
    command = sub.add_parser("recommend", help="Select the fastest feasible measured configuration")
    command.add_argument("--results", required=True)
    command.add_argument("--memory-budget-gb", required=True, type=float, help="Maximum lifecycle peak reserved GPU memory in GiB (2^30 bytes)")
    command.add_argument("--max-perplexity-increase-pct", required=True, type=float)
    command.add_argument("--workload", help="Workload name or full workload ID; required if ambiguous")
    command.add_argument("--suite", help="Full suite ID; required if results mix environments/protocols")
    command = sub.add_parser("report", help="Generate summary.csv and report.md from raw results")
    command.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .environment import inspect_environment
            env = inspect_environment()
            print(json.dumps(env, indent=2))
            return 0 if env["inference_ready"] else 1
        if args.command in ("benchmark", "evaluate"):
            from .config import load_config
            from .runner import execute
            records = execute(load_config(args.config), args.command, args.only)
            from .report import write_report
            write_report(load_config(args.config).results_dir)
            return 0 if all(r["status"] == "ok" for r in records) else 1
        if args.command == "report":
            from .report import write_report
            print(write_report(args.results))
            return 0
        from .results import load_results, recommend
        selection = recommend(load_results(args.results), args.memory_budget_gb,
                              args.max_perplexity_increase_pct, args.workload, args.suite)
        print(json.dumps(selection, indent=2))
        return 0 if selection["status"] == "ok" else 2
    except (ValueError, OSError, ImportError) as exc:
        print(f"cachepilot: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
