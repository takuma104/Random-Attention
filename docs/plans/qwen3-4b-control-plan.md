# Qwen3-4B対照: 準備段階

2026-09-19。Qwen3.5の主評価・retrieval・効率・C4096/C8192 frontierと表記感度解析が完了したため、当初計画にあるQwen3-4B対照へ進む。

## 目的と限界

- 原論文で使われた`Qwen/Qwen3-4B`のofficial checkpointを固定し、同じprompt保護/eviction規約を全Full Attention層へ適用できる別adapterを検証する。
- Qwen3.5では8 FA層のみ、Qwen3では全FA層が対象。モデル間の比較にはtraining、tokenizer、layer/head数、生成長、推奨sampling等の差がある。**DeltaNetの因果効果を単離した実験ではない。**
- 原論文の結果を引用するだけで、自前Qwen3.5との差をarchitectureの差とは断定しない。
- 元のQwen3 reproduction、frozen Qwen3.5の生成ソース/環境/既存dataは変更しない。新規scripts/control moduleと別artifact/results directoryを使う。

## 現在の状態

Checkpoint固定、独立adapter、数値/生成/EOS/採点gate完了。初回BF16 eager参照は不合格であり、FP32診断後の明示的v2 gateを未使用C1024/C2048 traceでも通過した。詳細は`qwen3-control-numerical-investigation.md`。元ソース/環境は変更していない。

次の精度pilotは`qwen3-control-pilot-protocol.md`でMATH500 outcome取得前に固定した50問×2runs×3conditions、32768 cap。まず固定2問の8192 operational smokeと再開同一性を確認する。探索的pilotであり、architectureの因果効果や2pp精度維持を確定する試験ではない。

## 準備開始時のgate計画

1. `scripts/qwen3_control/prepare_artifacts.py`でmodel revisionをdownload前に固定し、config・generation_config・各download file SHAを保存する。Qwen3.5と同じMATH500 revision/固定pilot IDsを再利用する。
2. 実config、model cardの推奨sampling、現在のTransformers公式Qwen3 forwardを確認する。新しいlibrary installや既存環境の更新は行わない。
3. 公式native/no-eviction adapterのlogits一致、prompt/recent/absolute positions、独立row/head RNG、post-eviction eager参照、B2・EOS・採点を検証する。公式forwardをなるべく保持し、Qwen3.5用output gate/partial RoPEを誤用しない。
4. 精度試行の前にpilot条件・sampling・対象・run数・解析の位置づけを固定する。まず動作/数値gateであり、まだQwen3の新規精度結果はない。

## 取得・実装確認（精度pilot前）

- Revision `1cfa9a7208912126459214e8b04321603b3df60c`、Apache-2.0。約8.06 GBのweights/tokenizer/config等を取得し全file SHAを保存。`qwen3-control-artifacts.json`参照。
- 実configは36 FA層、32 Q heads / 8 KV heads、head_dim128、full RoPE、attention output gateなし。`max_position_embeddings=40960`、YaRNなし。モデルカードは32k output＋通常promptの余裕としてこのconfig値を説明している。
- `generation_config.json`のThinking設定: temperature.6/top_p.95/top_k20、EOS=[151645,151643]。追加presence penaltyは0を明示する。カードは反復時に1.5を提案するが、このgateや後続pilotの出力を見て切り替えない。Qwen3.5のtemperature1/presence1.5とは異なり、モデル間の交絡として開示する。
- `kvcompress/qwen3_control`のinstance-local hooksでQwen3のq_normと**full RoPE**を使用。公式forwardは変更せず、既存の汎用eviction storage/selectionとB1/B2 samplerはread-only再利用する。歴史的名称`HybridEvictionCache`でも、このconfigでは36層すべてがFAでDeltaNet stateは存在しない。
- 小型Qwen3のCPU tests4件通過: native/noop/eviction前logits一致、Random/Recency/SnapKVのeviction後eager参照、prompt/recent/position/cache形状、独立したcontrol sampling設定。
- 初回に計画した実checkpoint gateはB2、C256/r64、512 forwards。native/noop/3 selectorsのeviction前一致、36層のpost-eviction eager参照（BF16 rtol=.03/atol=.03）、sampled replay、EOS/counter、MATH500 prompt適合性を調べる。これは性能/精度評価ではない。

GPUは引き続きRTX5090一枚。downloadはcheckpoint取得だけで、9B/別datasetを同時に走らせない。Qwen3.5 C8192はreview後でもnative比−2.0pp、97.5% CI [−3.8,−0.4]で2pp精度維持は未確立。この結論を今回の対照によって遡って変更しない。
