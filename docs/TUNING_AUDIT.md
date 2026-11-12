# Tuning extension audit and plan

The existing v0.1 smoke artifacts are preserved unchanged. Initial inspection
found 51 passing CPU tests and two opt-in GPU tests. Missing features were batch
workloads, capacity search/resume, multiple ranking objectives, and a second model.

Timing uses CUDA synchronization before prefill, after first-token argmax, and
after final-token argmax; no per-token host copies occur inside timing. Model
load and warmup are excluded. Memory includes lifecycle peaks, not only steady
inference. This remains intentional and conservative for configuration selection.

The FP16 and NF4 quality artifacts have identical protocol ID
`eb0d45c19971db0e0355f17811739762c3a0862a4f3530baebcd448eeaab0a87`.
Both score 256 targets with windows `(0,65,1)`, `(1,129,65)`, `(65,193,129)`,
`(129,257,193)` (input start/end, first target), 64 targets each. NLL sums are
979.9690704345703 (FP16) and 1010.4231567382812 (NF4). Token hashes, model revision,
SDPA, FP16 compute and FP16 KV settings match. Summed cross-entropy is divided by
total scored targets, not number of windows. The +12.63% is consistent with these
observations; there is no evidence here that it is a loss-accounting bug.

The expanded GPU reference check initially failed because Transformers inherited
Qwen's model-specific `do_sample=True` when a default-valued GenerationConfig was
passed. This affected the reference test, not CachePilot's explicit argmax loop.
The reference now sets `use_model_defaults=False`, `do_sample=False` and
`eos_token_id=None` explicitly. All six GPU checks then passed: batch-one and
batch-two generated IDs match Transformers for both FP16 and NF4, and each model's
256-target sliding-window NLL agrees with independently masked built-in label
loss. The evaluation protocol and corpus were not changed to alter perplexity.

Implementation sequence:

1. Add batched workloads and versioned measurement settings while reading v1 files.
2. Add isolated candidate execution, budget/margin enforcement, exponential/binary
   discrete search, confirmation/neighbor checks, atomic resume journal and locks.
3. Add matched-workload latency/aggregate-throughput and fixed-batch/output context
   objectives, quality gating and explicit alternatives.
4. Test the workflow on the GPU, then run a bounded 512x128 representative case;
   leave the larger full sweep as a reproducible command.
5. Add Qwen2.5-1.5B-Instruct after Qwen3 works. Its official model card and
   Transformers Qwen2 documentation support the existing backend; no dependency
   or system-wide driver changes are needed.

References checked: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct and
https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2 .
