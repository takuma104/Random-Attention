# Qwen3-4B対照: 準備段階

2026-09-19。Qwen3.5の主評価・retrieval・効率・C4096/C8192 frontierと表記感度解析が完了したため、当初計画にあるQwen3-4B対照へ進む。

## 目的と限界

- 原論文で使われた`Qwen/Qwen3-4B`のofficial checkpointを固定し、同じprompt保護/eviction規約を全Full Attention層へ適用できる別adapterを検証する。
- Qwen3.5では8 FA層のみ、Qwen3では全FA層が対象。モデル間の比較にはtraining、tokenizer、layer/head数、生成長、推奨sampling等の差がある。**DeltaNetの因果効果を単離した実験ではない。**
- 原論文の結果を引用するだけで、自前Qwen3.5との差をarchitectureの差とは断定しない。
- 元のQwen3 reproduction、frozen Qwen3.5の生成ソース/環境/既存dataは変更しない。新規scripts/control moduleと別artifact/results directoryを使う。

## 次のgate

1. `scripts/qwen3_control/prepare_artifacts.py`でmodel revisionをdownload前に固定し、config・generation_config・各download file SHAを保存する。Qwen3.5と同じMATH500 revision/固定pilot IDsを再利用する。
2. 実config、model cardの推奨sampling、現在のTransformers公式Qwen3 forwardを確認する。新しいlibrary installや既存環境の更新は行わない。
3. 公式native/no-eviction adapterのlogits一致、prompt/recent/absolute positions、独立row/head RNG、post-eviction eager参照、B2・EOS・採点を検証する。公式forwardをなるべく保持し、Qwen3.5用output gate/partial RoPEを誤用しない。
4. 精度試行の前にpilot条件・sampling・対象・run数・解析の位置づけを固定する。まず動作/数値gateであり、まだQwen3の新規精度結果はない。

GPUは引き続きRTX5090一枚。downloadはcheckpoint取得だけで、9B/別datasetを同時に走らせない。Qwen3.5 C8192はreview後でもnative比−2.0pp、97.5% CI [−3.8,−0.4]で2pp精度維持は未確立。この結論を今回の対照によって遡って変更しない。
