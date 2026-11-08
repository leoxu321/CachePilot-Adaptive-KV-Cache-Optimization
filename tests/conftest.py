"""Synthetic records exist only in tests; never represent hardware measurements."""

from copy import deepcopy
import math

import pytest

from cachepilot.backend import SETTINGS
from cachepilot.config import fingerprint
from cachepilot.metrics import summarize, timing_metrics
from cachepilot.environment import comparison_environment


def benchmark(configuration="fp16", latency=1.0, memory=2 * 2**30):
    protocol = {"token_budget": 2, "context_tokens": 3, "stride": 2,
                "corpus_sha256": "test-corpus", "token_ids_sha256": "test-tokens"}
    environment = {"os": "test", "python": "test", "cpu": "test", "packages": {},
                   "environment": {}, "gpu": {"name": "test-only", "total_memory_bytes": 8 * 2**30}}
    identity = {"model": "Qwen/Qwen3-0.6B", "revision": "a" * 40, "seed": 42,
                "environment": comparison_environment(environment),
                "settings": SETTINGS, "evaluation": {k: protocol[k] for k in ("token_budget", "context_tokens", "stride")},
                "corpus_sha256": "test-corpus"}
    prompt = {"input_ids": [1] * 128, "input_sha256": fingerprint([1] * 128)}
    samples = [{**timing_metrics(latency * .2, latency * .8, 16), "generated_ids": [1] * 16,
                "peak_allocated_bytes": memory // 2, "peak_reserved_bytes": memory} for _ in range(3)]
    return {"schema_version": 1, "kind": "benchmark", "status": "ok", "configuration": configuration,
            "run_id": configuration + "-benchmark", "suite_identity": identity,
            "created_at": "2026-11-08T00:00:00+00:00", "environment": environment,
            "suite_id": fingerprint(identity), "model_revision": "a" * 40, "actual_model_revision": "a" * 40,
            "settings": SETTINGS, "load_s": 1.0, "config": {"measured_runs": 3, "warmup_runs": 1,
                "model": identity["model"], "seed": 42, "evaluation": identity["evaluation"]},
            "workload": {"name": "smoke", "prompt_tokens": 128, "output_tokens": 16}, "prompt": prompt,
            "workload_id": fingerprint({"input_sha256": prompt["input_sha256"], "output_tokens": 16, "settings": SETTINGS}),
            "samples": samples, "warmup_runs": [deepcopy(samples[0])],
            "summary": {key: summarize([s[key] for s in samples]) for key in ("ttft_s", "decode_s", "decode_tokens_per_s", "e2e_s", "peak_allocated_bytes", "peak_reserved_bytes")},
            "load_memory": {"peak_allocated_bytes": memory // 2, "peak_reserved_bytes": memory},
            "peak_allocated_bytes": memory // 2, "peak_reserved_bytes": memory}


def evaluation(configuration="fp16", ppl=10.0):
    result = benchmark(configuration)
    result["kind"] = "evaluate"
    result["run_id"] = configuration + "-quality"
    protocol = {**result["suite_identity"]["evaluation"], "corpus_sha256": "test-corpus",
                "token_ids_sha256": "test-tokens"}
    result["quality"] = {"protocol": protocol, "protocol_id": fingerprint(protocol),
                         "target_tokens": 2, "nll_sum": math.log(ppl) * 2, "perplexity": ppl,
                         "windows": [{"input_start": 0, "input_end": 3, "target_start": 1,
                                      "target_tokens": 2, "nll_sum": math.log(ppl) * 2}]}
    result["evaluation_memory"] = result["load_memory"].copy()
    return result


@pytest.fixture
def records():
    return [benchmark(), benchmark("nf4", .75, 2**30), evaluation(), evaluation("nf4", 10.4)]
