"""Sequential controller. Model lifetime is bounded by a child process."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .storage import write_json


def child_environment(seed):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def run_isolated(request, root):
    """Exactly one worker, with a persistent request/result/log and bounded lifetime."""
    root = Path(root)
    run_id = request["run_id"]
    request_path = root / "requests" / f"{run_id}.json"
    result_path = root / f"{run_id}.result.json"
    log_path = root / "logs" / f"{run_id}.log"
    write_json(request_path, request)
    log_path.parent.mkdir(exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            # subprocess.run kills and waits for the child on timeout or interruption.
            proc = subprocess.run([sys.executable, "-m", "cachepilot.worker", str(request_path.resolve()),
                str(result_path.resolve())], stdout=log, stderr=subprocess.STDOUT,
                env=child_environment(request["config"]["seed"]), timeout=request["config"]["timeout_seconds"])
        if not result_path.exists():
            write_json(result_path, failure_record(request, "error", f"Worker exited {proc.returncode}; see {log_path}"))
    except subprocess.TimeoutExpired:
        write_json(result_path, failure_record(request, "timeout", f"Worker exceeded {request['config']['timeout_seconds']}s and was terminated; see {log_path}"))
    except OSError as exc:
        write_json(result_path, failure_record(request, "error", f"Cannot start worker: {exc}"))
    return json.loads(result_path.read_text(encoding="utf-8"))


def resolve_revision(config):
    if re.fullmatch(r"[0-9a-f]{40}", config.revision):
        return config.revision
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(config.model, revision=config.revision).sha
    except Exception as exc:
        raise ValueError(f"Cannot resolve model revision: {exc}. For offline runs use a cached 40-character commit SHA.") from exc


def execute(config, kind, only=None):
    from .environment import inspect_environment
    env = inspect_environment()
    if not env["cuda_available"]:
        raise ValueError(env["guidance"])
    revision = resolve_revision(config)
    root = Path(config.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    configs = [only] if only else config.configurations
    if only and only not in config.configurations:
        raise ValueError(f"{only} is not enabled in the configuration file")
    results = []
    baseline_ok = {}
    for configuration in configs:
        workloads = config.workloads if kind == "benchmark" else [None]
        for workload in workloads:
            run_id = uuid.uuid4().hex
            request = {"kind": kind, "configuration": configuration, "revision": revision,
                       "config": config.to_dict(), "workload": workload, "run_id": run_id}
            request_path = root / "requests" / f"{run_id}.json"
            result_path = root / f"{run_id}.result.json"
            log_path = root / "logs" / f"{run_id}.log"
            write_json(request_path, request)
            log_path.parent.mkdir(exist_ok=True)
            label = workload["name"] if workload else "fixed-corpus"
            print(f"{kind}: {configuration} / {label} (revision {revision[:12]})", flush=True)
            # FP16 must complete successfully before quantized loading is attempted.
            permitted = configuration == "fp16" or baseline_ok.get(label, False)
            if not permitted:
                permitted = prior_fp16_success(root, config, revision, workload)
            if not permitted:
                record = failure_record(request, "skipped", "Run a successful matched FP16 benchmark first")
                write_json(result_path, record)
            else:
                record = run_isolated(request, root)
            if configuration == "fp16" and kind == "benchmark":
                baseline_ok[label] = record["status"] == "ok"
            print(f"  {record['status']}: {result_path}", flush=True)
            if record.get("error"):
                print(f"  {record['error']['message']}", flush=True)
            results.append(record)
    return results


def prior_fp16_success(root, config, revision, workload):
    # This is only a run-order guard. Selection later enforces full suite identities.
    for path in root.glob("*.result.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("kind") == "benchmark" and record.get("status") == "ok"
                and record.get("configuration") == "fp16" and record.get("model_revision") == revision
                and record.get("config", {}).get("model") == config.model
                and (workload is None or record.get("workload") == workload)):
                return True
        except (OSError, ValueError):
            continue
    return False


def failure_record(request, status, message):
    return {"schema_version": 1, "kind": request["kind"], "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(), "run_id": request["run_id"],
            "configuration": request["configuration"], "model_revision": request["revision"],
            "config": request["config"], "workload": request["workload"],
            "error": {"type": status, "message": message}}
