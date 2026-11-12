"""Synthetic feasibility functions only; no fabricated hardware results."""
import pytest

from cachepilot.capacity import SearchStopped, classify, effective_budget, load_state, search_boundary
from cachepilot.storage import write_json


@pytest.mark.parametrize("length", [1, 2, 3, 10, 31])
def test_every_possible_monotonic_boundary(length):
    candidates = list(range(1, length + 1))
    for boundary in range(-1, length):
        measured = {}
        def probe(index, repeat):
            measured.setdefault((index, repeat), {"feasible": index <= boundary, "reason": "test"})
            return measured[index, repeat]
        result = search_boundary(candidates, probe, 2)
        assert result["selected"] == (boundary + 1 if boundary >= 0 else None)
        assert result["range_exhausted"] == (boundary == length - 1)
        if boundary >= 0:
            assert (boundary, 1) in measured and (boundary, 2) in measured
        assert len(measured) <= length + 2


def test_search_is_logarithmic_before_neighbor_checks():
    calls = set()
    def probe(index, repeat):
        calls.add((index, repeat))
        return {"feasible": index < 512, "reason": "test"}
    assert search_boundary(list(range(1024)), probe, 1)["selected"] == 511
    assert len(calls) < 25


@pytest.mark.parametrize("status", ["unsupported", "error", "timeout", "skipped"])
def test_non_oom_failure_never_becomes_capacity_boundary(status):
    result = classify({"status": status, "error": {"message": "test"}}, 100)
    assert result["feasible"] is None
    with pytest.raises(SearchStopped, match="not a capacity boundary"):
        search_boundary([1, 2, 4], lambda i, r: result, 1)


def test_oom_and_budget_failures_are_distinct():
    assert classify({"status": "oom"}, 100) == {"feasible": False, "reason": "cuda_oom"}
    assert classify({"status": "ok", "peak_reserved_bytes": 101}, 100)["reason"] == "memory_budget"
    assert classify({"status": "ok", "peak_reserved_bytes": 100}, 100)["feasible"]
    assert effective_budget(7, .5, 8 * 2**30) == int(6.5 * 2**30)
    assert effective_budget(12, 1, 8 * 2**30) == 7 * 2**30
    with pytest.raises(ValueError):
        effective_budget(7, 8, 8 * 2**30)


def test_unstable_confirmation_is_not_claimed_success():
    with pytest.raises(SearchStopped, match="confirmation disagreed"):
        search_boundary([1, 2, 4], lambda i, r: {"feasible": i < 2 and r == 0, "reason": "test"}, 1)


def test_nonmonotonic_neighbor_stops_search():
    with pytest.raises(SearchStopped, match="non-monotonic"):
        search_boundary([1, 2, 4], lambda i, r: {"feasible": i != 1, "reason": "test"}, 1)


def test_partial_resume_reuses_completed_measurements(tmp_path):
    identity = {"model": "test", "budget": 100, "code": "v1"}
    path = tmp_path / "capacity.json"
    state = load_state(path, identity)
    real_calls = []
    def probe(index, repeat):
        key = f"{index}:{repeat}"
        if key in state["attempts"]:
            return state["attempts"][key]
        if len(real_calls) == 2:
            raise KeyboardInterrupt
        real_calls.append(key)
        state["attempts"][key] = {"feasible": index < 3, "reason": "test"}
        write_json(path, state)
        return state["attempts"][key]
    with pytest.raises(KeyboardInterrupt):
        search_boundary([1, 2, 4, 8], probe, 1)
    saved = set(real_calls)
    state = load_state(path, identity)
    def resume(index, repeat):
        key = f"{index}:{repeat}"
        if key not in state["attempts"]:
            assert key not in saved
            state["attempts"][key] = {"feasible": index < 3, "reason": "test"}
        return state["attempts"][key]
    assert search_boundary([1, 2, 4, 8], resume, 1)["selected"] == 4
    with pytest.raises(ValueError, match="incompatible"):
        load_state(path, identity | {"budget": 101})
