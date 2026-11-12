from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cachepilot import capacity
from cachepilot.config import parse_config
from cachepilot.storage import write_json


@pytest.fixture
def controller(tmp_path, monkeypatch):
    config = parse_config({"revision": "a" * 40, "configurations": ["fp16"],
        "workloads": [{"name": "test", "prompt_tokens": 128, "output_tokens": 16}],
        "results_dir": str(tmp_path), "capacity": {"axis": "prompt_tokens", "candidates": [128, 256, 512],
            "memory_budget_gb": 2, "safety_margin_gb": .1, "confirmation_runs": 1}})
    monkeypatch.setattr(capacity, "inspect_environment", lambda: {"cuda_available": True,
        "os": "test", "python": "test", "cpu": "test", "packages": {}, "environment": {},
        "gpu": {"total_memory_bytes": 8 * 2**30}})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=SimpleNamespace(
        from_pretrained=lambda *a, **k: SimpleNamespace(max_position_embeddings=256))))
    # Schema validation is tested separately; these are controller-only doubles.
    monkeypatch.setattr(capacity, "validate_record", lambda r: r)
    return config


def test_context_includes_output_and_completed_resume_avoids_workers(controller, monkeypatch):
    calls = []
    def run(request, root):
        calls.append(request)
        result = request | {"status": "ok", "model_revision": request["revision"], "peak_reserved_bytes": 2**30}
        write_json(root / f"{request['run_id']}.result.json", result)
        return result
    monkeypatch.setattr(capacity, "run_isolated", run)
    state = capacity.execute_capacity(controller)
    assert state["status"] == "complete"
    assert state["boundaries"]["fp16"]["selected"] == 128
    assert len(calls) == 2  # candidate and independent confirmation
    assert all(r["workload"]["prompt_tokens"] == 128 for r in calls)
    assert capacity.execute_capacity(controller)["status"] == "complete"
    assert len(calls) == 2
    first = calls[0]
    write_json(Path(controller.results_dir) / f"{first['run_id']}.result.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        capacity.execute_capacity(controller)


def test_orphaned_complete_result_is_adopted_after_interruption(controller, monkeypatch):
    calls = []
    def run(request, root):
        calls.append(request["run_id"])
        result = request | {"status": "ok", "model_revision": request["revision"], "peak_reserved_bytes": 2**30}
        write_json(root / f"{request['run_id']}.result.json", result)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return result
    monkeypatch.setattr(capacity, "run_isolated", run)
    with pytest.raises(KeyboardInterrupt):
        capacity.execute_capacity(controller)
    assert capacity.execute_capacity(controller)["status"] == "complete"
    assert len(calls) == len(set(calls)) == 2


def test_runtime_error_stops_and_is_reused_on_resume(controller, monkeypatch):
    calls = []
    def run(request, root):
        calls.append(request)
        result = request | {"status": "error", "model_revision": request["revision"], "error": {"message": "unrelated failure"}}
        write_json(root / f"{request['run_id']}.result.json", result)
        return result
    monkeypatch.setattr(capacity, "run_isolated", run)
    assert capacity.execute_capacity(controller)["status"] == "inconclusive"
    assert capacity.execute_capacity(controller)["status"] == "inconclusive"
    assert len(calls) == 1
