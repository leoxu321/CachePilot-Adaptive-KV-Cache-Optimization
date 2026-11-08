"""Pure calculations shared by measurement, validation, and reporting."""

import math
import statistics


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def timing_metrics(ttft_s, decode_s, output_tokens):
    positive(ttft_s, "ttft_s")
    positive(decode_s, "decode_s")
    if type(output_tokens) is not int or output_tokens < 2:
        raise ValueError("At least two output tokens are required to measure decode")
    return {"ttft_s": ttft_s, "decode_s": decode_s,
            "decode_tokens_per_s": (output_tokens - 1) / decode_s,
            "e2e_s": ttft_s + decode_s, "output_tokens": output_tokens}


def summarize(values):
    values = sorted(positive(v, "sample", allow_zero=True) for v in values)
    if len(values) < 2:
        raise ValueError("At least two measured samples are required")
    def percentile(p):
        at = (len(values) - 1) * p
        low = int(at)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (at - low)
    median = statistics.median(values)
    return {"median": median, "p25": percentile(.25), "p75": percentile(.75),
            "iqr": percentile(.75) - percentile(.25), "min": values[0],
            "max": values[-1], "stdev": statistics.stdev(values), "n": len(values)}


def perplexity(nll_sum, target_tokens):
    positive(nll_sum, "nll_sum", allow_zero=True)
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    return math.exp(nll_sum / target_tokens)


def relative_increase(value, baseline):
    return (positive(value, "perplexity") / positive(baseline, "baseline perplexity") - 1) * 100


def evaluation_windows(token_budget, context_tokens, stride):
    """Yield (input start, input end exclusive, first scored target index).

    Token zero is context only; target indices 1..token_budget are scored once.
    """
    if token_budget < 1 or not 1 <= stride < context_tokens:
        raise ValueError("Invalid evaluation window parameters")
    for target_start in range(1, token_budget + 1, stride):
        end = min(target_start + stride, token_budget + 1)
        yield max(0, end - context_tokens), end, target_start
