# Qwen3-4B control progress

## 2026-09-19 10:02 JST: smoke/recovery accepted; proceed to fixed pilot

Protocol committed **before Qwen3 MATH outcomes**: `13802cd`, `qwen3-control-pilot-protocol.md`. No sampling, budget, question, replication or output-cap changes based on smoke scores.

- Official checkpoint pinned; Qwen3-specific hooks preserve upstream full-RoPE/no-gate forward. Original Qwen3 reproduction and frozen Qwen3.5 remain untouched.
- Numeric/implementation gates complete. The initial BF16 eager-reference failure and FP32-based revised gate are disclosed in `qwen3-control-numerical-investigation.md`, not erased. Held-out C1024/C2048 gates passed.
- Operational first2 questions ×2 runs ×3 cells at8192 cap:12 answers,6 batches, job `qwen3-control-smoke8k-42d8`,09:54:43–10:01:51 including audit and identical-command resume. Generation itself used .114881 GPU hours (~6m54s).
- Strict CPU audits passed: no missing/extra answers, pending journals or grading errors;36-layer cache/EOS/pairing/source accounting correct. Eight native-versus-compressed pre-eviction prefixes, **11,788 tokens exact**.
- Resume preserved all12 raw files, data SHA `d421472be8ccd916ecf1da0416f8cf975946306f97f42ebfad8af0fa227eb632`. Manifest SHA `0a89429efab729ec739959eb558b989e9d6ca1f381c24fc502cc123dc6673e64`.
- Smoke native/Random1024/Random2048 each had3/4 final-correct,4/4 paper-correct and1/4 capped; all EOS answers correct. These tiny operational results are **not comparative accuracy evidence**, not pooled into the32768 pilot and not used for parameter selection.
- Mean tokens5865/6037.5/5579.75; shared peak allocated9.813/7.829/8.114GiB. Unequal sampled lengths/row residency do not establish a speedup.
- Reports: `qwen3-control-smoke-analysis.json`, `qwen3-control-smoke-resume-audit.json`. Numerical, sampler, accounting and shared recovery/RNG tests passed16 tests; new CPU output-cap prefix tests add3.

The unchanged-code fixed50 questions ×2 runs ×native/Random1024/Random2048 at32768 cap (`results/qwen3_control/pilot32k_b2_v1`) started10:04:35, job `qwen3-control-pilot32k-9025`. Operational audits at10/25/50, plus exact8k→32k overlap audit. Final problem-cluster intervals are descriptive/exploratory, not confirmatory preservation or causal architecture tests.

## 10:29 JST: first10-question operational audit

- 60 answers /30 batches complete; no pending journals or grading exceptions in the audited prefix. Source/environment-related file hashes,36-layer state accounting and paired pre-eviction prefixes passed:40 prefixes /59,949 tokens exact.
- All60 audited answers reached EOS. No interim contrasts/hypothesis tests or parameter/sample-size changes were made. Generation accounting .401799 GPU hours; deterministic-prefix runtime extrapolation is provisional.
- The separate8k→32k audit passed all12 overlapping trajectories, **69,929 tokens exact**, including unchanged termination for all9 already-EOS answers. Smoke raw SHA remains unchanged.
- Pilot manifest SHA `b6f4de7349587205b14ad04980572567f3b9fec314fc69aa6d26c996a2aaa0aa`; first10 data SHA `7de329448326d3f9c6c8630763b53c2794daf1dae944926fbac9da72dee037f6`.
- Reports: `qwen3-control-progress-10.json`, `qwen3-control-8k-32k-prefix-early.json`. No generation-source edits or competing GPU workloads.

## 11:46 JST: first25-question operational audit

- 150 answers /75 batches complete. Strict source, dataset,36-layer cache, pairing and grading-status audit passed;100 pre-eviction prefixes /149,635 tokens exact.
- First10 raw data SHA remains unchanged. First25 data SHA: `8ad7b64e4b1da05fe27177c0d94eba6fc8b6c8ba246e93f12b4c744cbccc8456`.
- Generation accounting1.684811 GPU hours. Capped and EOS-wrong answers remain included; no exclusions, source/sampling changes, interim hypothesis tests or adaptation.
- Report: `qwen3-control-progress-25.json`. Continue unchanged to all50 questions and the preregistered final exploratory analysis.
