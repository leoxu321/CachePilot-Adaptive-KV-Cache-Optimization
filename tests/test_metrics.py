import math

import pytest

from cachepilot.metrics import evaluation_windows, perplexity, relative_increase, summarize, timing_metrics


def test_decode_excludes_first_token_and_prefill():
    measured = timing_metrics(0.25, 0.75, 16)
    assert measured["decode_tokens_per_s"] == 20
    assert measured["e2e_s"] == 1


def test_summary_has_interpolated_quartiles_and_sample_stdev():
    summary = summarize([4, 1, 3, 2])
    assert summary["median"] == 2.5
    assert summary["p25"] == 1.75
    assert summary["p75"] == 3.25
    assert summary["iqr"] == 1.5
    assert summary["stdev"] == pytest.approx(math.sqrt(5 / 3))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, True])
def test_invalid_measurements_are_rejected(invalid):
    with pytest.raises(ValueError):
        timing_metrics(invalid, 1, 2)


def test_weighted_perplexity_uses_total_nll_not_mean_window_ppl():
    assert perplexity(2 * math.log(2) + math.log(8), 3) == pytest.approx(32 ** (1 / 3))
    assert relative_increase(10.5, 10) == pytest.approx(5)


@pytest.mark.parametrize("budget,context,stride", [(1, 2, 1), (256, 128, 64), (137, 64, 31), (1024, 256, 128)])
def test_windows_score_every_target_once_with_context(budget, context, stride):
    targets = []
    for start, end, target_start in evaluation_windows(budget, context, stride):
        assert start < target_start < end
        assert end - start <= context
        targets.extend(range(target_start, end))
    assert targets == list(range(1, budget + 1))
