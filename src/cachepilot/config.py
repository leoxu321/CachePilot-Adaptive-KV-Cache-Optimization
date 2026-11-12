"""Small, strict configuration schema. Unknown settings are errors."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import yaml


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def keys(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")


@dataclass(frozen=True)
class Config:
    model: str
    revision: str
    configurations: list
    workloads: list
    warmup_runs: int
    measured_runs: int
    seed: int
    evaluation: dict
    results_dir: str
    timeout_seconds: int
    capacity: dict | None = None

    def to_dict(self):
        return asdict(self)


def load_config(path):
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {exc}") from exc
    return parse_config(raw)


def parse_config(raw):
    keys(raw, Config.__dataclass_fields__, "config")
    defaults = dict(model="Qwen/Qwen3-0.6B", revision="main",
                    configurations=["fp16", "nf4"], warmup_runs=1,
                    measured_runs=3, seed=42, evaluation={"token_budget": 256,
                    "context_tokens": 128, "stride": 64}, results_dir="results",
                    timeout_seconds=1800)
    raw = defaults | raw
    if raw["model"] not in ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-1.5B-Instruct"):
        raise ValueError("Supported models: Qwen/Qwen3-0.6B and Qwen/Qwen2.5-1.5B-Instruct")
    if not isinstance(raw["revision"], str) or not raw["revision"].strip():
        raise ValueError("revision must be a nonempty model revision")
    configs = raw["configurations"]
    if not isinstance(configs, list) or not configs or any(x not in ("fp16", "nf4") for x in configs):
        raise ValueError("configurations must contain fp16 and optionally nf4")
    if len(set(configs)) != len(configs) or "fp16" not in configs:
        raise ValueError("configurations must be unique and include fp16 as the baseline")
    raw["configurations"] = ["fp16"] + (["nf4"] if "nf4" in configs else [])
    workloads = raw.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be a nonempty list")
    seen = set()
    for workload in workloads:
        keys(workload, ("name", "prompt_tokens", "output_tokens", "batch_size"), "workload")
        name = workload.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Workload names must be nonempty and unique")
        seen.add(name)
        integer(workload.get("prompt_tokens"), "prompt_tokens", 32)
        integer(workload.get("output_tokens"), "output_tokens", 2)
        integer(workload.get("batch_size", 1), "batch_size")
        if workload["prompt_tokens"] + workload["output_tokens"] > 32768:
            raise ValueError("Workload exceeds the supported 32768-token context")
    integer(raw["warmup_runs"], "warmup_runs")
    integer(raw["measured_runs"], "measured_runs", 2)
    integer(raw["seed"], "seed", 0)
    integer(raw["timeout_seconds"], "timeout_seconds")
    ev = raw["evaluation"]
    keys(ev, ("token_budget", "context_tokens", "stride"), "evaluation")
    for key in ("token_budget", "context_tokens", "stride"):
        integer(ev.get(key), key, 2 if key == "context_tokens" else 1)
    if ev["context_tokens"] > 32768 or ev["stride"] >= ev["context_tokens"]:
        raise ValueError("Evaluation requires stride < context_tokens <= 32768")
    if not isinstance(raw["results_dir"], str) or not raw["results_dir"]:
        raise ValueError("results_dir must be a nonempty path (relative to working directory)")
    if raw.get("capacity") is not None:
        cap = raw["capacity"]
        keys(cap, ("axis", "candidates", "memory_budget_gb", "safety_margin_gb", "confirmation_runs"), "capacity")
        if len(workloads) != 1:
            raise ValueError("Capacity search requires exactly one fixed base workload")
        if cap.get("axis") not in ("prompt_tokens", "batch_size"):
            raise ValueError("capacity.axis must be prompt_tokens or batch_size")
        candidates = cap.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("capacity.candidates must be a nonempty increasing list")
        for value in candidates:
            integer(value, "capacity candidate", 32 if cap["axis"] == "prompt_tokens" else 1)
        if candidates != sorted(set(candidates)):
            raise ValueError("capacity candidates must be unique and increasing")
        from .metrics import positive
        positive(cap.get("memory_budget_gb"), "capacity memory budget")
        positive(cap.get("safety_margin_gb"), "capacity safety margin", allow_zero=True)
        if cap["safety_margin_gb"] >= cap["memory_budget_gb"]:
            raise ValueError("Safety margin must be smaller than the memory budget")
        integer(cap.get("confirmation_runs"), "confirmation_runs")
    return Config(**raw)
