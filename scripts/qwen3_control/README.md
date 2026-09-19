# Isolated official Qwen3 control

See `docs/plans/qwen3-control-pilot-protocol.md` before running. Original Qwen3 engine and frozen Qwen3.5 sources are not modified. One GPU workload at a time; no environment updates.

```bash
.venv/bin/python scripts/qwen3_control/run_math_batch.py --phase smoke --out results/qwen3_control/smoke8k_b2_v1
.venv/bin/python scripts/qwen3_control/analyze_math.py results/qwen3_control/smoke8k_b2_v1 --out docs/plans/qwen3-control-smoke-analysis.json
# Repeat the SAME smoke command, then re-audit and require identical data SHA.
# Only after smoke + recovery acceptance:
.venv/bin/python scripts/qwen3_control/run_math_batch.py --phase pilot --out results/qwen3_control/pilot32k_b2_v1
.venv/bin/python scripts/qwen3_control/analyze_math.py results/qwen3_control/pilot32k_b2_v1 --problems 10 --out docs/plans/qwen3-control-progress-10.json
.venv/bin/python scripts/qwen3_control/analyze_math.py results/qwen3_control/pilot32k_b2_v1 --final --out docs/plans/qwen3-control-pilot-analysis.json
```

The runner freezes sources, protocol, environment and validation sidecars. Do not edit these during active/resumable experiments. It hashes checkpoint files before each startup and journals both expensive answers before grading either. A grading interruption must recover the journal, not regenerate. The CPU auditor handles all36 FA layers and verifies per-row state, paired prefixes and dataset/source identity; it does not use the Qwen3.5 eight-FA-layer auditor.

`--phase smoke` fixes2 questions/2 runs/3 cells/8192 cap. `--phase pilot` fixes50 questions/2 runs/3 cells/32768 cap. Neither permits ad-hoc parameter selection. Smoke records are not pooled into pilot accuracy. Final clustered intervals are exploratory only. Outputs under `work/` and `results/` remain ignored.

Numerical provenance includes the initial BF16 eager-reference failure, FP32 investigation and prospectively fixed held-out v2 gates; see `docs/plans/qwen3-control-numerical-investigation.md`. Do not relabel the original failed gate as passed.
