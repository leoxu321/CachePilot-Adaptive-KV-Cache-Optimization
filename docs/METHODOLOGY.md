# Measurement protocol v1

## Scope and controlled variables

CachePilot compares `Qwen/Qwen3-0.6B` in FP16 with bitsandbytes NF4 linear weights.
NF4 uses FP16 computation, no double quantization, and the library's default
unquantized modules (including embedding/output layers); it does not make every
parameter four bits. The actual quantization configuration is saved in JSON.
Both variants use a Transformers dynamic **FP16** KV cache, SDPA attention, batch
size one, greedy argmax, no sampling, no TF32, no compilation, and cuda:0 only.
The model is in evaluation mode and forward passes use inference mode.
The non-thinking chat template sets `enable_thinking=False`.

Each workload has one synthetic prompt of exactly the requested number of tokens,
including the complete chat prefix and non-thinking assistant suffix. The content
is constructed by repeating a fixed text's token IDs, taking the explicitly
requested content budget, and adding those template tokens. This is a length
microbenchmark, not a natural prompt distribution. It does not silently truncate
user text. Full input IDs, their SHA256, template overhead, and generated IDs are
saved. Output length is fixed; EOS does not stop generation, even if emitted.
The model card's preferred sampling recipe is deliberately not used: this tool
measures controlled greedy workloads, not best conversational output quality.

## Timing boundaries

All times are host `time.perf_counter()` wall-clock seconds. CUDA is synchronized
immediately before the first forward, immediately after selecting the first
generated token, and after selecting the last token. Synchronization is **not**
performed between every decode token. These timings include Python dispatch,
cache construction, forward computation, argmax selection, and synchronization
overhead. They are not CUDA-kernel-only measurements or serving-system latency.

| Metric | Definition |
|---|---|
| Load seconds | Synchronized tokenizer/model loading and placement; includes download on a cold cache and NF4 conversion when applicable. Excludes environment inspection. |
| TTFT | From just before cache creation/prefill until first selected token is available on GPU. Includes the complete prompt forward and argmax. |
| Decode seconds | From that first-token boundary to completion of the final selected token. |
| Decode tokens/s | `(output_tokens - 1) / decode_seconds`; first generated token and prefill are excluded. |
| End-to-end seconds | TTFT + decode seconds, for device-resident input through last device token. |

CPU prompt construction/tokenization, initial host-to-device input transfer,
returning token IDs to CPU, text decoding, model load, warmup, file IO, and
process startup are outside inference timing. This definition of end-to-end
is the **inference loop**, not user-visible request latency. At least two output
tokens and two measured repetitions are required. Warmup results are stored
separately and never included in measured summaries. No `model.generate()` call
or streaming callback is used; the explicit greedy forward loop exposes the
prefill/decode boundary without per-token CPU copies.

## GPU memory

The worker records `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()` after resetting peak counters before loading,
each warmup, and each measured run. Resetting counters does not discard resident
weights or cached allocator blocks. Allocator caches are retained across warmups
and repetitions, as in steady inference. Each new configuration/workload uses a
fresh subprocess, which exits before the next begins.

The top-level benchmark peak is the **maximum across loading, warmup, and all
measured repetitions**. Selection compares this peak **reserved** value to the
budget. This includes a configuration's transient load/quantization requirements
and is deliberately stricter than only considering steady-state allocated bytes.
Reports include allocated bytes as a separate metric. The option name uses `gb`,
but its documented unit is **GiB = 2^30 bytes**.

PyTorch peaks exclude some CUDA context/library/driver memory and other programs.
They are not total board VRAM measurements. `nvidia-smi` and CUDA free/total memory
are recorded at worker startup as context, not as the budget metric. Windows WDDM
can virtualize GPU residency; CUDA placement checks rule out explicit CPU/disk
offloading but cannot guarantee absence of driver paging. Leave headroom and avoid
other GPU workloads for stronger experiments. The driver-reported CUDA version is
not the installed PyTorch runtime version; both are recorded separately.

## Quality

`src/cachepilot/data/quality.txt` is a fixed, original English prose corpus provided
with the project under CC0. It requires no downloaded dataset or paid service.
The smoke test uses 256 scored target tokens; extended uses 1024. These are small
regression checks, not representative accuracy or generalization benchmarks.

Rules are identical for FP16 and NF4:

1. Decode corpus bytes as UTF-8; tokenize the whole text without special tokens or
   a chat template. Require at least `token_budget + 1` available tokens.
2. Select that explicit prefix. Token zero is context only. Score targets
   1 through `token_budget` exactly once.
3. Advance by `stride` targets. Each window ends after its new targets and includes
   up to `context_tokens` total input tokens. Overlapping context is not rescored.
   Early windows have less context. Require `stride < context_tokens`.
4. Forward with `use_cache=False`; use FP32 logits for summed cross-entropy on the
   newly scored targets. Accumulate window sums in Python double precision.
5. `perplexity = exp(total_negative_log_likelihood / scored_target_tokens)`.
   Relative increase is `100 * (candidate_PPL / FP16_PPL - 1)`.

Each window's boundaries, target count and loss sum, corpus SHA256, token-ID SHA256,
budget, context, and stride are saved. Quality must match both the suite and the
evaluation protocol before selection can use it. Generated text equality is not
used as a quality metric. Evaluation memory/time are recorded separately and do
not enter the inference budget or latency objective.

## Aggregation and selection

Each timing metric has a median, linearly interpolated p25/p75, IQR, sample standard
deviation, min/max, and count. These quantify observed variability; they are not
confidence intervals. Raw repetitions are retained. Repeated successful files
within a suite/configuration/workload pool all measured samples, not only the
fastest file. Memory uses the maximum across those files. Evaluation reruns use
the highest observed perplexity per configuration, with the FP16 value as baseline.

A suite fingerprints model commit, backend settings, quality settings/corpus,
seed, hardware identity, OS, Python, package versions, and relevant environment.
A workload fingerprints exact prompt IDs and generation settings/length.
Mutable model revisions are resolved once per command before spawning workers;
all workers load that exact commit. Offline reproduction should use a literal
commit SHA. Results from incompatible suites are never combined.

For the requested workload, keep only measured configurations whose lifecycle
peak reserved memory is within budget and whose matched relative perplexity
increase is within tolerance. Select the lowest median end-to-end latency.
Ties break by configuration name. Inclusive comparisons allow floating-point
roundoff at a perplexity boundary (1e-12 tolerance). Missing quality is infeasible.
Unknown workloads or no passing candidates produce `no feasible configuration`.
Multiple workloads/suites require explicit disambiguation. There is no learned
policy, extrapolation, or recommendation for an unmeasured workload.

## Validity limits

Small samples, one synthetic prompt per shape, a tiny prose corpus, fixed FP16-first
ordering, GPU clocks, thermal conditions, and desktop contention limit inference
from these results. Greedy output may differ and changes the generated histories,
though the number of forward steps is identical. Subsequent tokens are cached in
FP16 in both variants. Weight quantization may reduce memory and still be slower.
An observed failure is saved as OOM, unsupported, error, timeout, or skipped and
does not become a numerical result. Reports reject malformed raw result files.
Fingerprints prevent accidental joins; they are not cryptographic attestations
against deliberately fabricated data.

## Official references checked on 2026-11-08

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B): Qwen3 support and non-thinking template.
- [Transformers bitsandbytes integration](https://huggingface.co/docs/transformers/quantization/bitsandbytes): NF4 and compute dtype.
- [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation): Windows x64, CUDA 12.8 and sm86 support.
- [PyTorch installation](https://pytorch.org/get-started/locally/) and [versioned wheels](https://pytorch.org/get-started/previous-versions/).
- [Transformers perplexity](https://huggingface.co/docs/transformers/v4.57.1/en/perplexity): fixed-context evaluation and overlapping windows.
