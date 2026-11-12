from copy import deepcopy

import pytest

from cachepilot.results import recommend


def test_throughput_ranks_post_prefill_aggregate_rate(records):
    # Slow prefill but fast decoding makes throughput differ from E2E objective.
    for sample in records[0]["samples"]:
        sample["decode_tokens_per_s"] = 100
    assert recommend(records, 7, 5, objective="throughput")["selection"]["configuration"] == "fp16"
    assert recommend(records, 7, 5, objective="latency")["selection"]["configuration"] == "nf4"


def test_context_requires_matched_batch_and_outputs(records):
    other = deepcopy(records[0])
    other["workload"] = {"name": "long", "prompt_tokens": 2048, "output_tokens": 16}
    other.update(workload_id="long", run_id="long-run")
    records.append(other)
    winner = recommend(records, 7, 5, objective="context")["selection"]
    assert winner["prompt_tokens"] == 2048
    other["workload"]["output_tokens"] = 128
    with pytest.raises(ValueError, match="matched batch"):
        recommend(records, 7, 5, objective="context")
    assert recommend(records, 7, 5, objective="context", output_tokens=16)["status"] == "ok"


def test_context_applies_quality_and_memory_first(records):
    records[1]["workload"]["prompt_tokens"] = 4096
    records[1]["workload_id"] = "long"
    assert recommend(records, 7, 3, objective="context")["selection"]["configuration"] == "fp16"
    assert recommend(records, .5, 5, objective="context")["selection"] is None


def test_throughput_refuses_unmatched_workloads(records):
    records[1]["workload_id"] = "different"
    with pytest.raises(ValueError, match="Multiple workloads"):
        recommend(records, 7, 5, objective="throughput")


def test_tie_break_is_deterministic_and_reports_other_feasible(records):
    for s in records[1]["samples"]:
        s["e2e_s"] = 1
    result = recommend(records, 7, 5)
    assert result["selection"]["configuration"] == "fp16"
    assert result["rejected"][0]["reasons"] == ["feasible but ranked behind winner"]


@pytest.mark.parametrize("status", ["oom", "error", "timeout", "unsupported"])
def test_failed_confirmation_cannot_be_hidden_by_successful_sample(records, status):
    failed = deepcopy(records[1])
    failed.update(status=status, run_id="failed-confirmation", error={"message": "test failure"})
    records.append(failed)
    result = recommend(records, 7, 5)
    assert result["selection"]["configuration"] == "fp16"
    assert any(status in reason for r in result["rejected"] for reason in r["reasons"])


def test_failed_only_alternative_has_explanation(records):
    failed = deepcopy(records[0])
    failed.update(status="oom", run_id="failed-only", error={"message": "test OOM"})
    result = recommend([failed], 7, 5)
    assert result["status"] == "no feasible configuration"
    assert "oom" in result["rejected"][0]["reasons"][0]
