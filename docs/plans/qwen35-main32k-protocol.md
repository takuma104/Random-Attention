# MATH500主評価: 事前固定プロトコル

固定日: 2026-09-10、本評価開始前。生成実装はcommit `cc919ad`。
対象: `results/qwen35/main32k_b2_v3/`。B1 pilot/確認実験と混ぜない。

## 条件

- Qwen/Qwen3.5-4B、revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- BF16、textのみ、全checkpointをGPU上に保持。学習・量子化・offload・MTPなし。
- MATH500全500問。revision/hashは`qwen35-artifacts.json`。全promptがC1024に適合（最大857 tokens）。
- 全問×2反復（run0/run1）、同一promptをB=2で同時処理。
- 全条件最大**生成**32768 tokens。EOS union=[248044,248046]。
- 公式Thinking template＋既存のboxed指示。temperature1、top_p.95、top_k20、presence_penalty1.5（生成tokenのみ）、repetition_penalty1、min_p0。
- seedは`stable_seed(problem_id, run, stream)`。generationとevictionは別stream、evictionはrow/layerごとに独立。
- 前後のFull Attentionゲート/partial RoPE/DeltaNetは公式forwardのまま。
- FLA chunk/recurrentを使用、causal convは公式torch fallback、Full AttentionはSDPA。
- 終了rowはresidentに残す。回答は各row最初のEOSまでで、partnerの後続生成を含めない。

## 5セル（当初候補を維持）

1. native hybrid (evictionなし)
2. random_pp C1024
3. random_pp C2048
4. recency_pp C1024
5. snapkv_pp C1024

Cはrecent r=64を含む総保持数/KV head。全8 Full Attention層を対象に、prompt全体を保持。合計5000回答、2500 batches。

## 評価・統計

- 主指標: `</think>`後の最後のboxed回答が正しい割合。cap例を分母から除外しない。
- 副指標: EOSを要求した正答率、元論文互換completion全体の正答率、cap率、thinking未終了率、生成長、eviction exposure。
- 問題単位の対応付きcluster bootstrap（10000回、seed20260910）。同じ問題の2反復を同じclusterとしてresample。各問題・各runを等重みとする。
- **精度維持の主比較**: Random C2048 − native。事前目標は低下2 percentage points以内。95% CI下限が-0.02を上回る場合のみ、この設定で目標を支持したと記述する。非有意を同等とみなさない。
- **選択手法の主比較**: Random C1024 − Recency C1024、およびRandom C1024 − SnapKV C1024。問題単位でrun平均差のexact sign testを行い、この2比較でHolm補正。方向の一致した95% CIと補正p<.05を確認して優劣を述べる。
- その他の比較・層別解析は副次的/探索的。副次的な差から主比較の失敗を置き換えない。
- SnapKVはリポジトリのdecode adaptationを参考にしたFP32 scorerであり、元実装とのbit同一性は主張しない。TriAttentionとの比較はこの実験に含まない。
- 本評価にはpilotで見た50問も含むことを開示。モデル学習・手法/予算の精度に基づく変更はしていない。B1 pilotとB2 mainの結果は合算しない。

## 監査と停止条件

全runのID、prompt tokens、EOS、seed、生成長、論理cache長、物理長、eviction回数、採点エラー、batch回復journalを監査する。

- 不正解・長い推論・capの多さを理由に問題や手法を除外しない。
- NaN、OOM、cache不整合、採点エラー、データ/コード不一致など実装・実行の問題では停止して調査する。
- 途中集計は動作・実行時間の監視用。精度を見てサンプル数・予算・samplingを変更しない。
- 20問完了後に最初の運用確認を行い、その後継続。500問完了前の精度は最終結果としない。
- ソースを実行中に編集しない。各回答は原子的に保存し、設定・source hashを一致させて再開する。

## 時間・メモリ

B1確認からの単純外挿は約250 GPU時間。短い固定workloadのB2 throughputは約1.9倍。反復の長さの違いによるidle rowを考慮すると、**約5〜8日程度を当面の目安**とするが、20問の運用確認で更新する。精度条件やサンプル数は変更しない。

`elapsed_seconds`は各batch wall time/2を配賦したGPU時間。実レイテンシは`request_latency_seconds`、共有peak memoryはbatchの値として扱う。native/圧縮はallocatorが異なり生成長も違うので、この生成実験のtok/sから一般的な高速化倍率を主張しない。

## 実行・再開コマンド

```bash
.venv/bin/python scripts/qwen35/run_math_batch.py \
  --out results/qwen35/main32k_b2_v3 \
  --subset all --limit 500 --runs 2 --max-new-tokens 32768
```

同じコマンドで再開。完了後:

```bash
.venv/bin/python scripts/qwen35/analyze_math.py results/qwen35/main32k_b2_v3 \
  --out docs/plans/qwen35-main32k-analysis.json
```

Holm補正と主比較の判定はこのprotocolに従って解析側で追加し、生成ソースは変更しない。
