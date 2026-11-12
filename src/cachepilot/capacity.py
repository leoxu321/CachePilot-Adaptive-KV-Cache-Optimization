"""Resumable discrete capacity search; errors are never interpreted as boundaries."""

from contextlib import contextmanager
import json
from pathlib import Path
import os
import uuid

from .backend import SETTINGS_V2, PRECISIONS
from .config import fingerprint
from .environment import comparison_environment, inspect_environment
from .metrics import positive
from .results import validate_record
from .runner import resolve_revision, run_isolated, child_environment
from .storage import write_json


class SearchStopped(ValueError):
    pass


def effective_budget(memory_gib, margin_gib, total_bytes):
    positive(memory_gib, "memory budget")
    positive(margin_gib, "safety margin", allow_zero=True)
    result = min(memory_gib * 2**30, total_bytes) - margin_gib * 2**30
    if result <= 0:
        raise ValueError("Safety margin consumes the entire device/budget allowance")
    return int(result)


def classify(record, budget_bytes):
    if record["status"] == "oom":
        return {"feasible": False, "reason": "cuda_oom"}
    if record["status"] != "ok":
        return {"feasible": None, "reason": record["status"], "error": record.get("error")}
    peak = record["peak_reserved_bytes"]
    return {"feasible": peak <= budget_bytes, "reason": "within_budget" if peak <= budget_bytes else "memory_budget",
            "peak_reserved_bytes": peak}


def search_boundary(candidates, probe, confirmations):
    """probe(index, repetition) must persist and reuse completed observations.

    Exponential indices 0,1,3,7,... followed by binary refinement. Adjacent
    points and independent boundary confirmations are checked before completion.
    """
    if not candidates or confirmations < 1:
        raise ValueError("Candidates and at least one confirmation are required")
    observed = {}
    def check(index, repetition=0):
        result = probe(index, repetition)
        if result["feasible"] is None:
            raise SearchStopped(f"Candidate {candidates[index]}: {result['reason']}; not a capacity boundary")
        value = result["feasible"]
        if index in observed and observed[index] != value:
            raise SearchStopped("Boundary confirmation disagreed; feasibility is unstable")
        observed[index] = value
        if any(i < j and not a and b for i, a in observed.items() for j, b in observed.items()):
            raise SearchStopped("Observed non-monotonic feasibility; no reliable boundary")
        return value
    low, high, index = -1, len(candidates), 0
    while True:
        if not check(index):
            high = index
            break
        low = index
        if index == len(candidates) - 1:
            break
        index = min(index * 2 + 1, len(candidates) - 1)
    while high - low > 1:
        middle = (low + high) // 2
        if check(middle):
            low = middle
        else:
            high = middle
    # Check both neighboring points, including an extra point beyond a failed
    # boundary when available, to catch common non-monotonic observations.
    for neighbor in sorted({low - 1, low + 1, high + 1}):
        if 0 <= neighbor < len(candidates):
            check(neighbor)
    if low >= 0:
        for repeat in range(1, confirmations + 1):
            check(low, repeat)
    return {"selected": candidates[low] if low >= 0 else None,
            "selected_index": low if low >= 0 else None,
            "next_infeasible": candidates[high] if high < len(candidates) else None,
            "range_exhausted": low == len(candidates) - 1,
            "observed": {str(candidates[i]): feasible for i, feasible in sorted(observed.items())},
            "assumption": "approximately monotonic feasibility; unsampled gaps are not measurements"}


@contextmanager
def search_lock(root):
    """OS lock releases on process death, so an interrupted search can resume."""
    path = root / "capacity.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another capacity search owns this results directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_state(path, identity):
    expected = fingerprint(identity)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("search_id") != expected or state.get("identity") != identity:
            raise ValueError("Capacity resume incompatible with config, model, code, or environment; use a new results_dir")
        return state
    return {"schema_version": 1, "search_id": expected, "identity": identity,
            "status": "running", "attempts": {}, "boundaries": {}}


def execute_capacity(config):
    if not config.capacity:
        raise ValueError("The config requires a capacity section")
    env = inspect_environment()
    if not env["cuda_available"]:
        raise ValueError(env["guidance"])
    root = Path(config.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    with search_lock(root):
        path = root / "capacity.json"
        # Resolve mutable revisions once for the whole resumable search.
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        revision = previous["identity"]["resolved_revision"] if previous else resolve_revision(config)
        from transformers import AutoConfig
        model_context = AutoConfig.from_pretrained(config.model, revision=revision, trust_remote_code=False).max_position_embeddings
        budget = effective_budget(config.capacity["memory_budget_gb"], config.capacity["safety_margin_gb"], env["gpu"]["total_memory_bytes"])
        stable_env = comparison_environment(env)
        child_env = child_environment(config.seed)
        stable_env["environment"] = {k: child_env.get(k) for k in stable_env["environment"]}
        # Code digest prevents resuming a measurement after implementation changes.
        source = {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")}
        identity = {"config": config.to_dict(), "resolved_revision": revision, "model_context": model_context,
                    "effective_budget_bytes": budget, "environment": stable_env,
                    "settings": SETTINGS_V2, "precisions": PRECISIONS, "code_sha256": fingerprint(source)}
        state = load_state(path, identity)
        for attempt in state["attempts"].values():
            if attempt.get("result_sha256"):
                raw = root / f"{attempt['run_id']}.result.json"
                record = validate_record(json.loads(raw.read_text(encoding="utf-8")))
                if fingerprint(record) != attempt["result_sha256"]:
                    raise ValueError("Completed capacity measurement changed; refusing resume")
        write_json(path, state)
        axis = config.capacity["axis"]
        candidates = config.capacity["candidates"]
        baseline_loaded = any(a.get("worker_status") == "ok" and a["configuration"] == "fp16" for a in state["attempts"].values())
        for configuration in config.configurations:
            if configuration in state["boundaries"]:
                continue
            def probe(index, repeat):
                nonlocal baseline_loaded
                size = candidates[index]
                key = f"{configuration}:{size}:{repeat}"
                workload = {**config.workloads[0], "batch_size": config.workloads[0].get("batch_size", 1),
                            axis: size, "name": f"capacity-{axis}-{size}"}
                attempt = state["attempts"].get(key)
                if attempt and "feasible" in attempt:
                    if attempt.get("run_id"):
                        raw = root / f"{attempt['run_id']}.result.json"
                        record = validate_record(json.loads(raw.read_text(encoding="utf-8")))
                        if fingerprint(record) != attempt["result_sha256"]:
                            raise ValueError("Completed capacity measurement changed; refusing resume")
                    return attempt
                if workload["prompt_tokens"] + workload["output_tokens"] > model_context:
                    attempt = {"configuration": configuration, "candidate": size, "repeat": repeat,
                               "feasible": False, "reason": "model_context_limit", "workload": workload}
                elif configuration == "nf4" and not baseline_loaded:
                    raise SearchStopped("A successful FP16 benchmark is required before NF4 search")
                else:
                    # Save the run ID before starting: an orphaned completed raw
                    # result can be adopted after interruption without remeasurement.
                    attempt = attempt or {"configuration": configuration, "candidate": size, "repeat": repeat,
                                          "run_id": uuid.uuid4().hex, "workload": workload}
                    state["attempts"][key] = attempt
                    write_json(path, state)
                    request = {"kind": "benchmark", "configuration": configuration, "revision": revision,
                               "config": config.to_dict(), "workload": workload, "run_id": attempt["run_id"]}
                    raw = root / f"{attempt['run_id']}.result.json"
                    print(f"capacity: {configuration} {axis}={size}, confirmation={repeat}", flush=True)
                    record = (json.loads(raw.read_text(encoding="utf-8")) if raw.exists() else run_isolated(request, root))
                    validate_record(record)
                    if (record["configuration"] != configuration or record["workload"] != workload
                            or record["config"] != config.to_dict() or record["model_revision"] != revision):
                        raise ValueError("Raw capacity result does not match its request")
                    attempt.update(classify(record, budget))
                    attempt.update(worker_status=record["status"], result_sha256=fingerprint(record))
                    if configuration == "fp16" and record["status"] == "ok":
                        baseline_loaded = True
                state["attempts"][key] = attempt
                write_json(path, state)
                print(f"  {attempt['reason']} (feasible={attempt['feasible']})", flush=True)
                return attempt
            try:
                state["boundaries"][configuration] = search_boundary(candidates, probe, config.capacity["confirmation_runs"])
            except SearchStopped as exc:
                state.update(status="inconclusive", error=str(exc))
                write_json(path, state)
                return state
            write_json(path, state)
        state.update(status="complete")
        state.pop("error", None)
        write_json(path, state)
        return state
