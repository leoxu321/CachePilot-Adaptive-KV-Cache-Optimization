"""Opt-in correctness checks against Transformers' own generation/loss paths.

Run only with a cached snapshot and CACHEPILOT_GPU_TESTS=1. Not timing results.
"""

from importlib.resources import files
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CACHEPILOT_GPU_TESTS") != "1",
                                reason="opt-in local GPU/reference checks")
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@pytest.fixture(scope="module")
def backend():
    from cachepilot.backend import load_backend, seed_runtime
    seed_runtime(42)
    return load_backend("Qwen/Qwen3-0.6B", REVISION, "fp16")


def test_manual_decode_matches_transformers_generate(backend):
    import torch
    from transformers import GenerationConfig
    from cachepilot.backend import generate_once, make_prompt
    model, tokenizer = backend
    ids, prompt = make_prompt(tokenizer, 128)
    assert len(ids) == 128
    inputs = torch.tensor([ids], device="cuda:0")
    measured = generate_once(model, inputs, 16)
    with torch.inference_mode():
        expected = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            generation_config=GenerationConfig(max_new_tokens=16, do_sample=False,
                eos_token_id=None, pad_token_id=tokenizer.pad_token_id,
                use_cache=True, cache_implementation="dynamic"))
    assert measured["generated_ids"] == expected[0, 128:].tolist()


def test_perplexity_matches_transformers_label_loss(backend):
    import torch
    from cachepilot.backend import evaluate_model
    model, tokenizer = backend
    quality = evaluate_model(model, tokenizer, {"token_budget": 32, "context_tokens": 64, "stride": 32})
    text = files("cachepilot").joinpath("data/quality.txt").read_text(encoding="utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False)[:33]
    inputs = torch.tensor([ids], device="cuda:0")
    with torch.inference_mode():
        reference = model(input_ids=inputs, labels=inputs, use_cache=False).loss.item()
    assert quality["nll_sum"] / quality["target_tokens"] == pytest.approx(reference, rel=1e-5)
