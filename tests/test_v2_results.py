from copy import deepcopy

import pytest

from cachepilot.backend import PRECISIONS, SETTINGS_V2
from cachepilot.config import fingerprint
from cachepilot.metrics import summarize
from cachepilot.results import validate_record


def batched_record(record):
    record = deepcopy(record)
    record.update(schema_version=2, settings=SETTINGS_V2, precision=PRECISIONS["fp16"])
    record["suite_identity"]["settings"] = SETTINGS_V2
    record["suite_id"] = fingerprint(record["suite_identity"])
    record["workload"]["batch_size"] = 2
    record["workload_id"] = fingerprint({"input_sha256": record["prompt"]["input_sha256"],
        "output_tokens": 16, "batch_size": 2, "settings": SETTINGS_V2})
    for s in record["samples"] + record["warmup_runs"]:
        s.update(batch_size=2, aggregate_decode_tokens_per_s=s["decode_tokens_per_s"] * 2,
                 generated_ids=[s["generated_ids"], s["generated_ids"]])
    record["summary"]["aggregate_decode_tokens_per_s"] = summarize([s["aggregate_decode_tokens_per_s"] for s in record["samples"]])
    return record


def test_valid_batched_result_and_aggregate_numerator(records):
    record = batched_record(records[0])
    validate_record(record)
    record["samples"][0]["aggregate_decode_tokens_per_s"] /= 2
    with pytest.raises(ValueError, match="aggregate throughput"):
        validate_record(record)


def test_wrong_precision_rejected(records):
    record = batched_record(records[0])
    record["precision"] = PRECISIONS["nf4"]
    with pytest.raises(ValueError, match="precision"):
        validate_record(record)
