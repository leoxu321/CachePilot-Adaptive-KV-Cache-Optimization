from pathlib import Path

import pytest
import yaml

from cachepilot.config import load_config


def test_shipped_configs_load():
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "configs/smoke.yaml").measured_runs == 3
    assert len(load_config(root / "configs/extended.yaml").workloads) == 3


@pytest.mark.parametrize("change", [
    {"unknown": True}, {"configurations": ["nf4"]}, {"measured_runs": 1},
    {"warmup_runs": 0}, {"seed": True},
    {"evaluation": {"token_budget": 20, "context_tokens": 16, "stride": 16}},
    {"workloads": [{"name": "bad", "prompt_tokens": 128, "output_tokens": 1}]},
])
def test_bad_configs_have_clear_errors(tmp_path, change):
    raw = {"workloads": [{"name": "test", "prompt_tokens": 128, "output_tokens": 16}], **change}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
