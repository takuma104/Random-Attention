# Qwen3.5 replication (RTX 5090)

Plan: [`docs/plans/qwen35-full-attention-random-eviction.md`](../../docs/plans/qwen35-full-attention-random-eviction.md).
Progress: [`docs/plans/qwen35-progress.md`](../../docs/plans/qwen35-progress.md).

Isolated from the original Qwen3/VaSE code. Python 3.12, BF16, Transformers SDPA.
Only Full Attention KV is evicted. Upstream DeltaNet, attention output gate and
partial RoPE remain intact. FLA chunk/recurrent kernels are required and audited;
causal conv currently uses the upstream torch fallback. No optimized serving
speed claims should be made from this prototype.

## Environment / artifacts

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r scripts/qwen35/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/bin/python scripts/qwen35/prepare_artifacts.py
```

The artifact script pins model/data revisions before downloading and saves the
50 pilot IDs. Full weights and data are excluded from Git.

## Validate

```bash
.venv/bin/python -m pytest kvcompress/qwen35 -v
.venv/bin/python scripts/qwen35/validate_model.py
```

## Run / resume

```bash
# Smoke only (2k cap does NOT establish long-reasoning accuracy).
.venv/bin/python scripts/qwen35/run_math.py --out results/qwen35/smoke_v1 \
  --limit 2 --max-new-tokens 2048 --cells native,random_pp:1024,recency_pp:1024
# Fixed pilot: 50 questions x 7 conditions x 1 rollout, 8k output cap.
.venv/bin/python scripts/qwen35/run_math.py --out results/qwen35/pilot_v2
```

Repeat the **exact same command** to resume. The runner refuses changes in
settings or source hashes; use a new output directory for a changed experiment.
Only one inference job should use the GPU at a time. Each answer is atomic JSON
with tokens, exact EOS/cap reason, seeds, prompt IDs, cache metrics and grades.
`summary.json` is updated after each answer. `.ungraded` records recover expensive
generations if the CPU grading subprocess is interrupted.

`C` includes the recent `r=64` slots. Prompt must fit strictly within `C-r`.
Initial implementation supports unpadded batch=1, prefill followed by single-token
decode. The full vision-language checkpoint is loaded, but only text is used.
Presence penalty applies to generated tokens only. Sampling and eviction use
separate explicit generators. All generation settings are in each manifest.

Stop at the union of model-config EOS and tokenizer/chat EOS: for the pinned
checkpoint these are `<|endoftext|>` (248044) and `<|im_end|>` (248046).
The initial `smoke_v1` and interrupted `pilot_v1` used only the former and are
INVALIDATED as accuracy evidence; keep them only for implementation diagnostics.
Use a fresh smoke output directory when running updated source.

The main metric is a boxed answer after `</think>`; the paper-compatible diagnostic
also scores unfinished thinking using the original repository extractor/grader.
Both inherit that grader's mathematical normalizations and tolerances. An answer
at the output cap stays in the denominator. Token count includes sampled EOS;
the final sampled token has not entered KV yet. Report eviction exposure and
truncation rates with accuracy. Do not call the 8k pilot the paper's 32k protocol.

Per-answer tok/s is a workload diagnostic, not a fixed-workload speed benchmark.
Native uses DynamicCache, compressed uses preallocated bounded storage; allocator
and cache-growth costs therefore differ. An iso-kernel/allocator efficiency study
is a separate later step.
