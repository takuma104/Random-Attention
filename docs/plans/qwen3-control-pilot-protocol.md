# Official Qwen3-4B control: exploratory MATH500 pilot

Fixed 2026-09-19 **before any Qwen3 MATH500 generation**, including the operational smoke. This is an isolated control, not a replacement for the original Qwen3 reproduction or frozen Qwen3.5 experiments.

## Scope and prior exposure

Qwen3.5 main, capacity frontier, retrieval, efficiency and grading-sensitivity results are already known. The same MATH500 questions and historical 50 pilot IDs have been used with Qwen3.5. Qwen3 numerical validation and the off-dataset arithmetic smoke `17+25` are known: native EOS at1132/975 tokens, Random1024 EOS at1175/975; all four legacy-graded correct. These are implementation checks, **not benchmark accuracy evidence**. No Qwen3 MATH outcomes were used to choose this pilot.

This is exploratory within-model compression evaluation on an official all-Full-Attention model. Differences from Qwen3.5 also involve training, tokenizer, layer/head layout, gate, sampling and generated lengths. It is **not a causal DeltaNet ablation**, an independent replication of the Qwen3.5 findings, or a test isolating architecture alone.

## Fixed design

- Official `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, official Transformers5.16.1 SDPA forward; no quantization, offload, MTP, YaRN or library upgrades.
- 36 FA layers, 32 Q/8 KV heads, head_dim128, full RoPE, no attention output gate. Maximum logical context40960; 32768 output cap plus all audited prompts fits.
- Data: same pinned `HuggingFaceH4/MATH-500` revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; JSONL SHA `67cb00bd5c8704069d6e104c66391dfd08109074cc644fff746a6a4f060ec6c5`.
- Exactly the ordered50 `pilot_ids` in `qwen3-control-artifacts.json`, two runs (0/1) each, same-prompt B2. **300 answers / 150 batches**, native, Random-pp C1024, Random-pp C2048.
- Same protected full prompt and recent64 convention: C includes both, choose C−P−64 older generated tokens independently per KV head/row/layer; retain chronological order and absolute positions. Compact after attention at C+64 to C. All500 Qwen3 prompts pass P<960, maximum814; no truncation/exclusion.
- Thinking chat template plus the unchanged step-by-step/boxed instruction. Released thinking sampling: temperature.6/top_p.95/top_k20, presence_penalty0, repetition_penalty1, min_p0. EOS union `[151645,151643]`. **No penalty change based on loops or scores.** Qwen3.5's temperature1/presence1.5 is a disclosed difference.
- Same frozen `stable_seed(problem_id,run,stream)`, including its historical `qwen35-experiment-v1` string namespace. Generation and eviction streams separated. Same seeds across conditions, independent rows. No pooling B1/B2 sampled trajectories.
- Fixed order: question order → native → Random1024 → Random2048, each with rows0/1. No adaptive scheduling or early stopping for accuracy. Finished rows remain resident; each record captures its own EOS/cap endpoint.

## Operational smoke and acceptance

First run the first **two** fixed questions × two runs × all three conditions at **8192** output cap in `results/qwen3_control/smoke8k_b2_v1`. These12 answers are operational only and not pooled into the32768 pilot. The complete32768 pilot in `results/qwen3_control/pilot32k_b2_v1` replays all50 questions with the same source/seeds; all overlapping8k prefixes must be exact (including EOS). Smoke accuracy does not select methods, capacities, sampler, questions, cap, repetitions or sample size.

Require strict CPU audit, no pending journal/grading errors, native-versus-compressed pre-eviction prefix equality, and a same-command smoke resume leaving every raw-answer SHA unchanged. If an implementation fault is found, preserve/invalidate affected artifacts, document/fix and revalidate before restarting under a new version; do not silently mix source versions. CUDA/Xid recurrence requires investigation, not blind retry. A genuine blocker may require consultation.

## Numerical evidence and reproducibility

- `qwen3-control-numerical-investigation.md` preserves the **failed original BF16 eager gate**, the separate full-FP32 diagnosis, and the explicitly post-diagnostic v2 criterion. It was fixed before held-out C1024/C2048 validation, which passed without further threshold changes.
- v2 exact native/noop/preeviction full-logit checks, per-row/head FP32-oracle tests, prompt/recent/position/cache checks and generation replay passed. Both real Qwen3 EOS IDs and finished-row isolation are also covered by a scripted CUDA test; full-checkpoint arithmetic EOS/grading/prefix gate passed.
- New scripts/modules only; original Qwen3 engine, Qwen3.5 Python sources/tests and grader/parser/normalizer unchanged. Generic cache selection, sampler, atomic writer and two-answer grading recovery are reused read-only, **not the Qwen3.5-specific attention hook**.
- Runner hashes all model assets before startup, freezes dataset/artifact settings, source hashes, official installed Qwen3/cache/SDPA source hashes, full package freeze, this protocol and validation sidecars. Resumes require identical settings. Local lock prevents two writers to an output directory; run only one GPU job overall.
- Both answers atomically journaled before either is graded. Partial grading resumes without regenerating durable trajectories. Raw outputs/checkpoints remain outside Git; reports and SHA-linked aggregates are tracked. Push only `origin/experiments`.

## Analysis fixed before outcomes

- Preserve legacy grader primary `final_correct`, `paper_correct`, EOS-required correctness; report cap rate, generated length, eviction exposure, EOS-wrong count, cache bytes and shared peak allocation. No inaccurate/long/capped-answer exclusions.
- At final50-question completion: Random1024−native and Random2048−native paired **problem-cluster** differences, averaging the two runs within each problem.10,000 bootstrap draws with seed20260919, same resampled clusters for both contrasts; descriptive percentile95% intervals and problem win/loss/tie counts.
- These two pilot intervals are **not multiplicity-adjusted confirmatory tests**. No significance claim or ≤2pp preservation decision from this small pilot, including a degenerate bootstrap interval; failure to detect a difference is not equivalence. No post-hoc best-capacity selection presented as confirmation.
- Operational audits at10/25/50 questions only; no interim hypothesis tests or adaptation. Further500-question controls require a new prospectively fixed protocol and are not authorized by this pilot's statistical design alone.
- Any grading sensitivity remains a separate disclosed overlay/review, never a rewrite of legacy scores or raw files. Existing Qwen3.5 review corrections are not automatically transferred to new answers.
- `elapsed_seconds=batch_wall_seconds/2` is accounting, not per-request latency. Report actual request latency separately if used; shared peak is not per-row peak. Native KV144KiB/token/sequence; all GDN state is zero. Unequal sampled lengths and row utilization are not fixed-workload or serving speedup evidence. No causal architectural interpretation from memory or accuracy correlations.
