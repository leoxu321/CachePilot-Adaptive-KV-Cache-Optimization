from copy import deepcopy
import json

import pytest

from cachepilot.results import load_results, recommend, validate_record


def test_valid_results(records):
    for record in records:
        assert validate_record(record) is record


@pytest.mark.parametrize("mutation", [
    lambda r: r["samples"][0].update(e2e_s=99),
    lambda r: r["samples"][0].update(ttft_s=float("nan")),
    lambda r: r["samples"][0].update(output_tokens=15),
    lambda r: r["samples"][0].update(peak_reserved_bytes=1),
    lambda r: r["summary"]["e2e_s"].update(median=.0001),
    lambda r: r.update(peak_reserved_bytes=1),
    lambda r: r["prompt"]["input_ids"].append(7),
    lambda r: r.update(suite_id="wrong"),
    lambda r: r.pop("samples"),
    lambda r: r.pop("environment"),
    lambda r: r["environment"].update(cpu="different machine"),
    lambda r: r["config"].update(seed=100),
    lambda r: r.pop("created_at"),
])
def test_corrupt_measurements_are_rejected(records, mutation):
    record = deepcopy(records[0])
    mutation(record)
    with pytest.raises(ValueError):
        validate_record(record)


def test_wrong_quality_token_count_is_rejected(records):
    records[2]["quality"]["target_tokens"] = 3
    with pytest.raises(ValueError):
        validate_record(records[2])


def test_select_fastest_feasible(records):
    result = recommend(records, 7, 5)
    assert result["selection"]["configuration"] == "nf4"
    assert recommend(records, 7, 3)["selection"]["configuration"] == "fp16"


def test_no_feasible_for_memory_or_quality_missing(records):
    assert recommend(records, .5, 5)["status"] == "no feasible configuration"
    assert recommend(records[:2], 7, 5)["selection"] is None
    assert recommend(records, 7, 5, workload="unmeasured")["selection"] is None


def test_memory_uses_peak_reserved_not_allocated(records):
    result = recommend(records, 1.0, 5)
    assert result["selection"]["configuration"] == "nf4"
    assert recommend(records, .9, 5)["selection"] is None


def test_exact_constraint_boundary_is_inclusive(records):
    records[3]["quality"]["perplexity"] = 10.5
    assert recommend(records, 1, 5)["selection"]["configuration"] == "nf4"


def test_mismatched_evaluation_protocol_is_not_joined(records):
    records[3]["quality"]["protocol_id"] = "other-tokenization"
    assert recommend(records, 7, 5)["selection"]["configuration"] == "fp16"


def test_different_suite_quality_is_not_joined(records):
    records[3]["suite_id"] = "other-machine"
    assert recommend(records, 7, 5)["selection"]["configuration"] == "fp16"


def test_multiple_workloads_require_explicit_choice(records):
    other = deepcopy(records[0])
    other.update(workload_id="different-workload", run_id="second")
    other["workload"]["name"] = "long"
    records.append(other)
    with pytest.raises(ValueError, match="Multiple workloads"):
        recommend(records, 7, 5)
    assert recommend(records, 7, 5, workload="smoke")["status"] == "ok"


def test_multiple_suites_require_explicit_choice(records):
    other = deepcopy(records[0])
    other.update(suite_id="other-suite", run_id="other")
    records.append(other)
    with pytest.raises(ValueError, match="Multiple comparison suites"):
        recommend(records, 7, 5)


def test_reruns_pool_all_samples_and_use_largest_peak(records):
    other = deepcopy(records[1])
    other["run_id"] = "nf4-rerun"
    other["peak_reserved_bytes"] = 8 * 2**30
    records.append(other)
    assert recommend(records, 7, 5)["selection"]["configuration"] == "fp16"
    result = recommend(records, 9, 5)
    assert result["selection"]["e2e_s"]["n"] == 6


def test_invalid_json_fails_closed(tmp_path):
    (tmp_path / "bad.result.json").write_text('{"schema_version": 100}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid result"):
        load_results(tmp_path)


def test_duplicate_ids_fail_closed(tmp_path, records):
    for name in ("first", "duplicate"):
        (tmp_path / f"{name}.result.json").write_text(json.dumps(records[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_results(tmp_path)
