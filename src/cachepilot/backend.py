"""One explicit Transformers/CUDA backend. No offloading or precision fallback."""

import hashlib
from importlib.resources import files
import time

from .config import fingerprint
from .metrics import evaluation_windows, perplexity, timing_metrics


SETTINGS = {"backend": "transformers", "attention": "sdpa", "batch_size": 1,
            "kv_cache": "dynamic", "kv_dtype": "float16", "compute_dtype": "float16",
            "generation": "greedy", "enable_thinking": False, "stop_on_eos": False,
            "cpu_offload": False, "device": "cuda:0", "tf32": False,
            "prompt_recipe": "synthetic-chat-v1", "measurement_version": 1}
PROMPT_TEXT = ("Explain how a library organizes books and helps readers find reliable information. "
               "Compare careful record keeping with guessing, using a simple example. ")


class UnsupportedConfiguration(RuntimeError):
    pass


def seed_runtime(seed):
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def load_backend(model_id, revision, configuration):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise UnsupportedConfiguration("CUDA PyTorch is unavailable. Run cachepilot doctor; CPU fallback is forbidden.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    kwargs = dict(revision=revision, dtype=torch.float16, device_map={"": 0},
                  attn_implementation="sdpa", trust_remote_code=False, use_safetensors=True)
    if configuration == "nf4":
        try:
            import bitsandbytes  # noqa: F401
        except (ImportError, OSError, RuntimeError) as exc:
            raise UnsupportedConfiguration(f"bitsandbytes unavailable: {exc}") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False)
    elif configuration != "fp16":
        raise UnsupportedConfiguration(f"Unknown configuration: {configuration}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if any(t.device.type != "cuda" or t.device.index != 0
           for t in list(model.parameters()) + list(model.buffers())):
        raise UnsupportedConfiguration("Model tensors are not all on cuda:0; offloading is forbidden")
    if configuration == "fp16" and any(p.dtype != torch.float16 for p in model.parameters()):
        raise UnsupportedConfiguration("FP16 model contains non-FP16 weights")
    if configuration == "nf4" and not getattr(model, "is_loaded_in_4bit", False):
        raise UnsupportedConfiguration("4-bit loading was not applied")
    return model, tokenizer


def make_prompt(tokenizer, target_tokens):
    marker = "CACHEPILOT_CONTENT_PLACEHOLDER"
    template = tokenizer.apply_chat_template([{"role": "user", "content": marker}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if template.count(marker) != 1:
        raise UnsupportedConfiguration("Cannot construct the fixed-length chat template")
    prefix, suffix = template.split(marker)
    pre = tokenizer.encode(prefix, add_special_tokens=False)
    post = tokenizer.encode(suffix, add_special_tokens=False)
    room = target_tokens - len(pre) - len(post)
    if room < 1:
        raise ValueError("prompt_tokens is too small for the complete non-thinking chat template")
    body = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    # Deliberate synthetic construction, not truncation of a user's natural prompt.
    ids = pre + (body * ((room + len(body) - 1) // len(body)))[:room] + post
    return ids, {"recipe": SETTINGS["prompt_recipe"], "template_tokens": len(pre) + len(post),
                 "content_tokens": room, "input_ids": ids, "input_sha256": fingerprint(ids)}


def peak_memory():
    import torch
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def generate_once(model, input_ids, output_tokens):
    import torch
    from transformers import DynamicCache
    # Cache allocation and Python loop are inside timing; CPU tokenization/H2D are outside.
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    cache = DynamicCache(config=model.config)
    generated = []
    with torch.inference_mode():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(token)
        del out
        torch.cuda.synchronize(0)
        first = time.perf_counter()
        for _ in range(output_tokens - 1):
            out = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(token)
            del out
        torch.cuda.synchronize(0)
        end = time.perf_counter()
    metrics = timing_metrics(first - start, end - first, output_tokens)
    metrics.update(peak_memory())
    # Verify the cache precision rather than merely claiming it was unchanged.
    if any(layer.keys.dtype != torch.float16 or layer.values.dtype != torch.float16 for layer in cache.layers):
        raise UnsupportedConfiguration("KV cache is not FP16")
    metrics["generated_ids"] = torch.cat(generated, dim=1).cpu().tolist()[0]
    return metrics


def evaluate_model(model, tokenizer, config):
    import torch
    import torch.nn.functional as F
    corpus = files("cachepilot").joinpath("data/quality.txt").read_bytes()
    ids = tokenizer.encode(corpus.decode("utf-8"), add_special_tokens=False)
    budget = config["token_budget"]
    if len(ids) < budget + 1:
        raise ValueError(f"Corpus has {len(ids)} tokens; needs {budget + 1}. Reduce token_budget.")
    selected = ids[:budget + 1]
    nll_sum = 0.0
    windows = []
    with torch.inference_mode():
        for start, end, target_start in evaluation_windows(budget, config["context_tokens"], config["stride"]):
            inputs = torch.tensor([selected[start:end]], dtype=torch.long, device="cuda:0")
            out = model(input_ids=inputs, use_cache=False)
            offset = target_start - start - 1
            logits = out.logits[:, offset:-1, :].float()
            targets = inputs[:, offset + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
            value = loss.item()
            nll_sum += value
            windows.append({"input_start": start, "input_end": end, "target_start": target_start,
                            "target_tokens": end - target_start, "nll_sum": value})
            del out, logits, loss, inputs, targets
    protocol = {**config, "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
                "token_ids_sha256": fingerprint(selected), "available_tokens": len(ids),
                "add_special_tokens": False, "chat_template": False,
                "rule": "sliding-window-v1; first token context only; each target scored once"}
    return {"perplexity": perplexity(nll_sum, budget), "nll_sum": nll_sum,
            "target_tokens": budget, "protocol": protocol,
            "protocol_id": fingerprint(protocol), "windows": windows}
