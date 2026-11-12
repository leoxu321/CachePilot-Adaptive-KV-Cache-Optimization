"""Private subprocess entry point: exactly one model/config/workload per process."""

from datetime import datetime, timezone
import json
import sys
import time
import traceback

from .backend import (SETTINGS_V2, PRECISIONS, UnsupportedConfiguration, evaluate_model, generate_once,
                      load_backend, make_prompt, peak_memory, seed_runtime)
from .config import fingerprint
from .environment import comparison_environment, inspect_environment
from .metrics import summarize
from .storage import write_json


def run(request):
    record = {"schema_version": 2, "kind": request["kind"], "status": "error",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "run_id": request["run_id"], "configuration": request["configuration"],
              "config": request["config"], "model_revision": request["revision"],
              "settings": SETTINGS_V2, "precision": PRECISIONS[request["configuration"]],
              "workload": request.get("workload"),
              "measurement_boundary": "device-resident input through last device token; no text decoding"}
    try:
        import torch
        env = inspect_environment()
        record["environment"] = env
        if not env["cuda_available"]:
            raise UnsupportedConfiguration(env["guidance"])
        seed_runtime(request["config"]["seed"])
        from importlib.resources import files
        import hashlib
        identity = {"model": request["config"]["model"], "revision": request["revision"],
                    "settings": SETTINGS_V2, "seed": request["config"]["seed"],
                    "evaluation": request["config"]["evaluation"],
                    "corpus_sha256": hashlib.sha256(files("cachepilot").joinpath("data/quality.txt").read_bytes()).hexdigest(),
                    "environment": comparison_environment(env)}
        record["suite_identity"] = identity
        record["suite_id"] = fingerprint(identity)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        model, tokenizer = load_backend(request["config"]["model"], request["revision"], request["configuration"])
        torch.cuda.synchronize(0)
        record["load_s"] = time.perf_counter() - start
        record["load_memory"] = peak_memory()
        record["actual_model_revision"] = getattr(model.config, "_commit_hash", None)
        if record["actual_model_revision"] != request["revision"]:
            raise UnsupportedConfiguration("Loaded model commit differs from the resolved revision")
        record["quantization"] = (model.config.quantization_config.to_dict()
            if request["configuration"] == "nf4" else None)
        if request["kind"] == "benchmark":
            workload = request["workload"]
            batch = workload.get("batch_size", 1)
            if workload["prompt_tokens"] + workload["output_tokens"] > model.config.max_position_embeddings:
                raise ValueError("Input plus output tokens exceed actual model context limit")
            ids, prompt = make_prompt(tokenizer, workload["prompt_tokens"])
            record["prompt"] = prompt
            record["workload_id"] = fingerprint({"input_sha256": prompt["input_sha256"],
                "output_tokens": workload["output_tokens"], "batch_size": batch, "settings": SETTINGS_V2})
            inputs = torch.tensor([ids] * batch, dtype=torch.long, device="cuda:0")
            warmups = []
            for _ in range(request["config"]["warmup_runs"]):
                torch.cuda.reset_peak_memory_stats(0)
                warmups.append(generate_once(model, inputs, workload["output_tokens"]))
            record["warmup_runs"] = warmups
            samples = []
            for index in range(request["config"]["measured_runs"]):
                torch.cuda.reset_peak_memory_stats(0)
                sample = generate_once(model, inputs, workload["output_tokens"])
                samples.append({"index": index, **sample})
            record["samples"] = samples
            record["summary"] = {key: summarize([x[key] for x in samples]) for key in (
                "ttft_s", "decode_s", "decode_tokens_per_s", "e2e_s",
                "peak_allocated_bytes", "peak_reserved_bytes", "aggregate_decode_tokens_per_s")}
            phases = [record["load_memory"], *warmups, *samples]
        else:
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            record["quality"] = evaluate_model(model, tokenizer, request["config"]["evaluation"])
            torch.cuda.synchronize(0)
            record["evaluation_s"] = time.perf_counter() - start
            record["evaluation_memory"] = peak_memory()
            phases = [record["load_memory"], record["evaluation_memory"]]
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            record[key] = max(phase[key] for phase in phases)
        record["status"] = "ok"
    except Exception as exc:
        try:
            import torch
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        except ImportError:
            is_oom = False
        record["status"] = "oom" if is_oom else ("unsupported" if isinstance(exc, (UnsupportedConfiguration, ImportError, NotImplementedError)) else "error")
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc(file=sys.stderr)
    return record


def main():
    request_path, result_path = sys.argv[1:]
    with open(request_path, encoding="utf-8") as stream:
        request = json.load(stream)
    record = run(request)
    write_json(result_path, record)
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
