import subprocess
import sys
from types import SimpleNamespace

import pytest

from cachepilot.config import Config
from cachepilot import runner, worker


def config(tmp_path):
    return Config(model="Qwen/Qwen3-0.6B", revision="a" * 40, configurations=["fp16", "nf4"],
        workloads=[{"name": "test", "prompt_tokens": 128, "output_tokens": 16}], warmup_runs=1,
        measured_runs=3, seed=42, evaluation={"token_budget": 256, "context_tokens": 128, "stride": 64},
        results_dir=str(tmp_path), timeout_seconds=30)


def test_failed_fp16_worker_prevents_nf4_and_preserves_logs(tmp_path, monkeypatch):
    monkeypatch.setattr("cachepilot.environment.inspect_environment", lambda: {"cuda_available": True})
    invocations = []
    def crash(command, **kwargs):
        invocations.append(command)
        kwargs["stdout"].write("simulated worker crash, test only")
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runner.subprocess, "run", crash)
    results = runner.execute(config(tmp_path), "benchmark")
    assert len(invocations) == 1
    assert [r["status"] for r in results] == ["error", "skipped"]
    assert "exited 7" in results[0]["error"]["message"]
    assert len(list(tmp_path.glob("*.result.json"))) == 2
    assert next((tmp_path / "logs").glob("*.log")).read_text() == "simulated worker crash, test only"


def test_timeout_is_recorded_and_next_variant_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("cachepilot.environment.inspect_environment", lambda: {"cuda_available": True})
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(runner.subprocess, "run", timeout)
    results = runner.execute(config(tmp_path), "benchmark")
    assert [r["status"] for r in results] == ["timeout", "skipped"]


def test_nf4_only_without_fp16_does_not_start_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("cachepilot.environment.inspect_environment", lambda: {"cuda_available": True})
    def unexpected(*args, **kwargs):
        pytest.fail("NF4 worker must not start without a successful FP16 benchmark")
    monkeypatch.setattr(runner.subprocess, "run", unexpected)
    assert runner.execute(config(tmp_path), "benchmark", only="nf4")[0]["status"] == "skipped"


@pytest.mark.parametrize("exception,status", [(worker.UnsupportedConfiguration("test unsupported"), "unsupported"),
                                              (ValueError("test invalid"), "error")])
def test_worker_preserves_failure_type(tmp_path, monkeypatch, exception, status):
    class FakeOOM(RuntimeError):
        pass
    cuda = SimpleNamespace(OutOfMemoryError=FakeOOM, synchronize=lambda _: None,
                           reset_peak_memory_stats=lambda _: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(worker, "inspect_environment", lambda: {"cuda_available": True,
        "os": "test", "python": "test", "cpu": "test", "packages": {}, "environment": {}})
    monkeypatch.setattr(worker, "seed_runtime", lambda _: None)
    def fail(*args):
        raise exception
    monkeypatch.setattr(worker, "load_backend", fail)
    request = {"kind": "benchmark", "run_id": "test", "configuration": "fp16", "revision": "a" * 40,
               "config": config(tmp_path).to_dict(), "workload": config(tmp_path).workloads[0]}
    record = worker.run(request)
    assert record["status"] == status
    assert record["error"]["message"] == str(exception)


def test_worker_classifies_oom(tmp_path, monkeypatch):
    class FakeOOM(RuntimeError):
        pass
    cuda = SimpleNamespace(OutOfMemoryError=FakeOOM, synchronize=lambda _: None,
                           reset_peak_memory_stats=lambda _: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(worker, "inspect_environment", lambda: {"cuda_available": True,
        "os": "test", "python": "test", "cpu": "test", "packages": {}, "environment": {}})
    monkeypatch.setattr(worker, "seed_runtime", lambda _: None)
    def fail(*args):
        raise FakeOOM("synthetic out of memory")
    monkeypatch.setattr(worker, "load_backend", fail)
    record = worker.run({"kind": "benchmark", "run_id": "test", "configuration": "fp16", "revision": "a" * 40,
                         "config": config(tmp_path).to_dict(), "workload": config(tmp_path).workloads[0]})
    assert record["status"] == "oom"
    assert "summary" not in record
