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

Next: unchanged-code fixed50 questions ×2 runs ×native/Random1024/Random2048 at32768 cap (`results/qwen3_control/pilot32k_b2_v1`). Operational audits at10/25/50, plus exact8k→32k overlap audit. Final problem-cluster intervals are descriptive/exploratory, not confirmatory preservation or causal architecture tests.
