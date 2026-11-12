"""Validate raw measurements and select only comparable, observed configurations."""

import json
import math
from pathlib import Path
from datetime import datetime
import re

from .config import fingerprint
from .metrics import perplexity, positive, relative_increase, summarize, timing_metrics


def close(actual, expected, name):
    positive(actual, name, allow_zero=True)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"Inconsistent {name}")


def validate_record(record):
    if not isinstance(record, dict) or record.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported result schema")
    if record.get("kind") not in ("benchmark", "evaluate"):
        raise ValueError("Invalid result kind")
    if record.get("configuration") not in ("fp16", "nf4"):
        raise ValueError("Invalid result configuration")
    if record.get("status") not in ("ok", "oom", "unsupported", "error", "timeout", "skipped"):
        raise ValueError("Invalid result status")
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        raise ValueError("Missing run_id")
    if record["status"] != "ok":
        if not record.get("error", {}).get("message"):
            raise ValueError("Failed result must explain its error")
        return record
    try:
        if datetime.fromisoformat(record["created_at"]).tzinfo is None:
            raise ValueError("Result timestamp must include its timezone")
        if not re.fullmatch(r"[0-9a-f]{40}", record["model_revision"]):
            raise ValueError("Result must contain an immutable model commit")
        if record["suite_id"] != fingerprint(record["suite_identity"]):
            raise ValueError("Invalid suite fingerprint")
        identity = record["suite_identity"]
        from .environment import comparison_environment
        if identity["environment"] != comparison_environment(record["environment"]):
            raise ValueError("Environment provenance does not match suite")
        if any(record["config"][key] != identity[key] for key in ("model", "seed", "evaluation")):
            raise ValueError("Configuration provenance does not match suite")
        positive(record["environment"]["gpu"]["total_memory_bytes"], "GPU capacity")
        if not isinstance(record["environment"]["gpu"]["name"], str):
            raise ValueError("Missing GPU name")
        if record["model_revision"] != identity["revision"] or record["actual_model_revision"] != identity["revision"]:
            raise ValueError("Inconsistent model revision")
        if record["settings"] != identity["settings"]:
            raise ValueError("Inconsistent backend settings")
        from .backend import SETTINGS, SETTINGS_V2, PRECISIONS
        v2 = record["schema_version"] == 2
        if record["settings"] != (SETTINGS_V2 if v2 else SETTINGS):
            raise ValueError("Unsupported measurement protocol")
        if v2 and record["precision"] != PRECISIONS[record["configuration"]]:
            raise ValueError("Unsupported precision settings")
        if v2 and record["configuration"] == "nf4":
            quant = record["quantization"]
            for key, value in {"bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16",
                               "bnb_4bit_use_double_quant": False, "bnb_4bit_quant_storage": "uint8"}.items():
                if quant.get(key) != value:
                    raise ValueError("Actual quantization settings differ from declared precision")
        positive(record["load_s"], "load_s")
        if record["kind"] == "benchmark":
            samples = record["samples"]
            if len(samples) < 2 or len(samples) != record["config"]["measured_runs"]:
                raise ValueError("Measured run count mismatch")
            if len(record["warmup_runs"]) != record["config"]["warmup_runs"] or not record["warmup_runs"]:
                raise ValueError("Warmup run count mismatch")
            prompt = record["prompt"]
            if any(type(token) is not int or token < 0 for token in prompt["input_ids"]):
                raise ValueError("Input token IDs must be nonnegative integers")
            if len(prompt["input_ids"]) != record["workload"]["prompt_tokens"] or fingerprint(prompt["input_ids"]) != prompt["input_sha256"]:
                raise ValueError("Prompt length or fingerprint mismatch")
            batch = record["workload"].get("batch_size", 1)
            if type(batch) is not int or batch < 1 or (not v2 and batch != 1):
                raise ValueError("Invalid batch size")
            workload_identity = {"input_sha256": prompt["input_sha256"],
                    "output_tokens": record["workload"]["output_tokens"], "settings": record["settings"]}
            if v2:
                workload_identity["batch_size"] = batch
            if record["workload_id"] != fingerprint(workload_identity):
                raise ValueError("Invalid workload fingerprint")
            for sample in samples + record["warmup_runs"]:
                sequences = [sample["generated_ids"]] if batch == 1 else sample["generated_ids"]
                if sample["output_tokens"] != record["workload"]["output_tokens"] or len(sequences) != batch or any(len(seq) != sample["output_tokens"] for seq in sequences):
                    raise ValueError("Output token count mismatch")
                expected = timing_metrics(sample["ttft_s"], sample["decode_s"], sample["output_tokens"])
                for key, value in expected.items():
                    close(sample[key], value, key)
                if v2:
                    close(sample["batch_size"], batch, "batch_size")
                    close(sample["aggregate_decode_tokens_per_s"], batch * expected["decode_tokens_per_s"], "aggregate throughput")
            summary_keys = ["ttft_s", "decode_s", "decode_tokens_per_s", "e2e_s", "peak_allocated_bytes", "peak_reserved_bytes"]
            if v2:
                summary_keys.append("aggregate_decode_tokens_per_s")
            for key in summary_keys:
                expected = summarize([s[key] for s in samples])
                for stat, value in expected.items():
                    close(record["summary"][key][stat], value, f"summary.{key}.{stat}")
            phases = [record["load_memory"], *record["warmup_runs"], *samples]
        else:
            quality = record["quality"]
            if quality["protocol_id"] != fingerprint(quality["protocol"]):
                raise ValueError("Invalid evaluation protocol fingerprint")
            protocol = quality["protocol"]
            if protocol["corpus_sha256"] != identity["corpus_sha256"] or any(
                protocol[k] != v for k, v in identity["evaluation"].items()):
                raise ValueError("Quality protocol does not match suite")
            if quality["target_tokens"] != protocol["token_budget"]:
                raise ValueError("Quality token budget mismatch")
            from .metrics import evaluation_windows
            expected_windows = list(evaluation_windows(protocol["token_budget"], protocol["context_tokens"], protocol["stride"]))
            observed_windows = [(w["input_start"], w["input_end"], w["target_start"]) for w in quality["windows"]]
            if observed_windows != expected_windows:
                raise ValueError("Evaluation target coverage mismatch")
            if sum(w["target_tokens"] for w in quality["windows"]) != quality["target_tokens"]:
                raise ValueError("Evaluation target count mismatch")
            for window in quality["windows"]:
                positive(window["nll_sum"], "window nll_sum", allow_zero=True)
                if window["target_tokens"] != window["input_end"] - window["target_start"]:
                    raise ValueError("Window target count mismatch")
            close(quality["nll_sum"], sum(w["nll_sum"] for w in quality["windows"]), "nll_sum")
            close(quality["perplexity"], perplexity(quality["nll_sum"], quality["target_tokens"]), "perplexity")
            phases = [record["load_memory"], record["evaluation_memory"]]
        for phase in phases:
            allocated = positive(phase["peak_allocated_bytes"], "peak allocated")
            reserved = positive(phase["peak_reserved_bytes"], "peak reserved")
            if reserved < allocated:
                raise ValueError("Reserved memory cannot be less than allocated memory")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            close(record[key], max(p[key] for p in phases), key)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Incomplete or malformed successful result: {exc}") from exc
    return record


def load_results(directory):
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Results directory does not exist: {root}")
    records = []
    for path in sorted(root.glob("*.result.json")):
        try:
            record = validate_record(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid result {path.name}: {exc}") from exc
        records.append(record)
    if len({r["run_id"] for r in records}) != len(records):
        raise ValueError("Duplicate run IDs in results directory")
    return records


def quality_for(records, suite, configuration):
    evaluations = [r for r in records if r["status"] == "ok" and r["kind"] == "evaluate"
                   and r["suite_id"] == suite and r["configuration"] == configuration]
    if not evaluations:
        return None
    protocols = {r["quality"]["protocol_id"] for r in evaluations}
    if len(protocols) != 1:
        raise ValueError("Conflicting evaluation tokenizations within a suite")
    # Conservatively use the worst observed perplexity, never cherry-pick a rerun.
    return max(evaluations, key=lambda r: r["quality"]["perplexity"])


def comparison_rows(records):
    groups = {}
    for record in records:
        if record["status"] == "ok" and record["kind"] == "benchmark":
            key = (record["suite_id"], record["workload_id"], record["configuration"])
            groups.setdefault(key, []).append(record)
    rows = []
    for (suite, workload_id, configuration), runs in sorted(groups.items()):
        baseline = quality_for(records, suite, "fp16")
        quality = quality_for(records, suite, configuration)
        compatible = baseline and quality and baseline["quality"]["protocol_id"] == quality["quality"]["protocol_id"]
        row = {"suite_id": suite, "workload_id": workload_id, "workload": runs[0]["workload"]["name"],
               "prompt_tokens": runs[0]["workload"]["prompt_tokens"],
               "output_tokens": runs[0]["workload"]["output_tokens"],
               "batch_size": runs[0]["workload"].get("batch_size", 1),
               "configuration": configuration, "run_ids": [r["run_id"] for r in runs],
               "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in runs),
               "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in runs),
               "perplexity": quality["quality"]["perplexity"] if quality else None,
               "perplexity_increase_pct": relative_increase(quality["quality"]["perplexity"], baseline["quality"]["perplexity"]) if compatible else None,
               "quality_run_id": quality["run_id"] if quality else None,
               "baseline_quality_run_id": baseline["run_id"] if baseline else None}
        samples = [s for r in runs for s in r["samples"]]
        for key in ("ttft_s", "decode_tokens_per_s", "e2e_s"):
            row[key] = summarize([s[key] for s in samples])
        row["aggregate_decode_tokens_per_s"] = summarize([
            s.get("aggregate_decode_tokens_per_s", s["decode_tokens_per_s"]) for s in samples])
        rows.append(row)
    return rows


def recommend(records, memory_budget_gb, max_increase_pct, workload=None, suite=None,
              objective="latency", batch_size=None, output_tokens=None):
    positive(memory_budget_gb, "memory budget")
    positive(max_increase_pct, "maximum perplexity increase", allow_zero=True)
    rows = comparison_rows(records)
    if objective not in ("latency", "throughput", "context"):
        raise ValueError("objective must be latency, throughput, or context")
    if suite:
        rows = [r for r in rows if r["suite_id"] == suite]
    if workload:
        rows = [r for r in rows if workload in (r["workload"], r["workload_id"])]
    if batch_size is not None:
        rows = [r for r in rows if r["batch_size"] == batch_size]
    if output_tokens is not None:
        rows = [r for r in rows if r["output_tokens"] == output_tokens]
    if len({r["suite_id"] for r in rows}) > 1:
        raise ValueError("Multiple comparison suites found; use --suite with a full suite ID from the report")
    if objective != "context" and len({r["workload_id"] for r in rows}) > 1:
        raise ValueError("Multiple workloads found; specify --workload")
    if objective == "context" and len({(r["batch_size"], r["output_tokens"]) for r in rows}) > 1:
        raise ValueError("Context comparisons require matched batch and output lengths; use --batch-size and --output-tokens")
    feasible = []
    rejected = []
    failures = [r for r in records if r["kind"] == "benchmark" and r["status"] != "ok"]
    def matches_failure(row, failure):
        if failure["configuration"] != row["configuration"] or not failure.get("workload"):
            return False
        if failure.get("suite_id") and failure["suite_id"] != row["suite_id"]:
            return False
        reference = next(r for r in records if r["run_id"] == row["run_ids"][0])
        if (failure.get("model_revision") != reference["model_revision"]
                or failure.get("config", {}).get("model") != reference["config"]["model"]):
            return False
        shape = failure["workload"]
        return (shape.get("prompt_tokens"), shape.get("output_tokens"), shape.get("batch_size", 1)) == (
            row["prompt_tokens"], row["output_tokens"], row["batch_size"])
    for row in rows:
        reasons = []
        failed_attempts = [r for r in failures if matches_failure(row, r)]
        if failed_attempts:
            reasons.append("unverified after failed matched measurement: " + ", ".join(
                f"{r['status']} ({r['run_id']})" for r in failed_attempts))
        if row["peak_reserved_bytes"] > memory_budget_gb * 2**30:
            reasons.append("memory budget exceeded")
        if row["perplexity_increase_pct"] is None:
            reasons.append("missing matched quality evaluation and FP16 baseline")
        elif row["perplexity_increase_pct"] > max_increase_pct and not math.isclose(
                row["perplexity_increase_pct"], max_increase_pct, rel_tol=1e-12, abs_tol=1e-12):
            reasons.append("perplexity constraint exceeded")
        if reasons:
            rejected.append({"configuration": row["configuration"], "workload": row["workload"],
                             "run_ids": row["run_ids"], "reasons": reasons})
        else:
            feasible.append(row)
    def rank(row):
        score = {"latency": row["e2e_s"]["median"],
                 "throughput": -row["aggregate_decode_tokens_per_s"]["median"],
                 "context": -row["prompt_tokens"]}[objective]
        return score, row["configuration"], row["workload_id"]
    winner = min(feasible, key=rank) if feasible else None
    for row in feasible:
        if row is not winner:
            rejected.append({"configuration": row["configuration"], "workload": row["workload"],
                             "run_ids": row["run_ids"], "reasons": ["feasible but ranked behind winner"]})
    # Failed-only alternatives have no numeric row but still deserve an explanation.
    for failure in failures:
        shape = failure.get("workload") or {}
        if suite and failure.get("suite_id") != suite:
            continue
        if workload and workload not in (shape.get("name"), failure.get("workload_id")):
            continue
        if batch_size is not None and shape.get("batch_size", 1) != batch_size:
            continue
        if output_tokens is not None and shape.get("output_tokens") != output_tokens:
            continue
        if not any(matches_failure(row, failure) for row in rows):
            rejected.append({"configuration": failure["configuration"], "workload": shape.get("name"),
                             "run_ids": [failure["run_id"]],
                             "reasons": [f"no successful verified measurement: {failure['status']}", failure["error"]["message"]]})
    return {"status": "ok" if winner else "no feasible configuration",
            "objective": objective, "memory_budget_gib": memory_budget_gb, "max_perplexity_increase_pct": max_increase_pct,
            "selection": winner, "rejected": rejected,
            "scope": "Measured workload only; maximum lifecycle PyTorch reserved memory; no extrapolation"}
