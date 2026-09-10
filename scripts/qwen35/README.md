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
The B1 runner supports unpadded batch=1, prefill followed by single-token
decode. The separate B2 runner below supports two rollouts of the same prompt. The full vision-language checkpoint is loaded, but only text is used.
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

## Same-prompt B2 runner (main protocol)

```bash
# B2 smoke: 2 questions x 3 methods x 2 rollouts.
.venv/bin/python scripts/qwen35/run_math_batch.py --out results/qwen35/smoke_b2_v3 \
  --limit 2 --max-new-tokens 8192 --cells native,random_pp:1024,snapkv_pp:1024
# Main: 500 questions x 5 methods x 2 rollouts, all at B=2.
.venv/bin/python scripts/qwen35/run_math_batch.py --out results/qwen35/main32k_b2_v3 \
  --subset all --limit 500 --runs 2 --max-new-tokens 32768
# Audit every expected problem/run, EOS/cap, cache counters, paired settings;
# then compute problem-clustered paired confidence intervals.
.venv/bin/python scripts/qwen35/analyze_main.py results/qwen35/main32k_b2_v3 \
  --out docs/plans/qwen35-main32k-analysis.json
```

`analyze_main.py` adds the preregistered Random–SnapKV contrast, two-comparison
Holm adjustment, and strict -2pp accuracy-preservation criterion. It refuses
incomplete main runs. During execution, `audit_progress.py ROOT --problems N
--out REPORT.json` audits an immutable completed prefix using a temporary derived
snapshot, without changing the live manifest or running interim hypothesis tests.
CPU-only tests: `.venv/bin/python -m pytest scripts/qwen35/test_statistics.py -q`.

B2 uses separate generation and eviction generators per row, never mixes different
prompts, and keeps finished rows resident. Per-row `elapsed_seconds` is batch wall
time divided by two (GPU time accounting), while `request_latency_seconds` is that
row's actual EOS/cap latency. Both rollouts are atomically journaled before either
is graded; partial grading can resume without repeating inference. Do not edit
frozen source mid-run. BF16 batch shape affects rounding and therefore sampled
trajectories: do not pool B1 pilot scores with B2 main scores.

Per-answer tok/s is a workload diagnostic, not a fixed-workload speed benchmark.
Native uses DynamicCache, compressed uses preallocated bounded storage; allocator
and cache-growth costs therefore differ. An iso-kernel/allocator efficiency study
is a separate later step.
