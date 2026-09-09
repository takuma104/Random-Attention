# Qwen3.5-4B: 8k pilot結果 (2026-09-10)

## 状態と条件

**探索用pilot完了。32k主評価の代用ではない。**

- Qwen3.5-4B BF16、Full Attention 8層のみeviction、DeltaNetは変更なし。
- 事前固定MATH500 50問、各1回、7条件、計350回答。
- Cはrecent64を含む圧縮後の総保持token数/KV head。prompt全体を保護。
- Thinking mode、temperature1.0、top_p.95、top_k20、presence_penalty1.5（生成tokenのみ）。生成とevictionのRNGを分離。
- 最大**生成**8192 tokens。EOSはcheckpointとchat tokenizerのunion（248044,248046）。
- 入力データ・model revisionは `qwen35-artifacts.json`。生成codeはcommit337db8d、run開始commit8a8d524。
- 実行: 00:33–07:09、wall約6時間36分、記録された生成時間合計6.58時間。
- 生出力: `results/qwen35/pilot_v2/` (Git対象外)。集計・監査: `qwen35-pilot-v2-analysis.json`。
- `smoke_v1`/`pilot_v1`はEOS問題修正前の診断であり、以下に含めない。

## 結果

| 条件 | final boxed正答率 | paper互換正答率 | cap率 | 平均生成tokens | eviction発動率 |
|---|---:|---:|---:|---:|---:|
| native hybrid | 76% | 76% | 24% | 4937 | — |
| Random C1024 | 64% | 68% | 36% | 5348 | 100% |
| Recency C1024 | 66% | 66% | 34% | 5258 | 100% |
| Random C2048 | 70% | 72% | 30% | 5084 | 90% |
| Recency C2048 | 70% | 70% | 30% | 5100 | 90% |
| Random C4096 | 74% | 74% | 26% | 5011 | 56% |
| Recency C4096 | 74% | 76% | 26% | 5059 | 56% |

final boxedは`</think>`後の最終boxedのみを採点。paper互換は元リポジトリextractor/graderでcompletion全体（未終了thinkingを含む）を採点。上限到達例を除外しない。

**全条件で、EOS終了例のfinal boxed誤答数は0。final boxedで不正解扱いとなった例は全てcap到達だった。** したがってこのpilotで観測した差には「答えが間違う」だけでなく「8kまでに答えを出し終えない」が強く関わる。32kで再確認しなければ、推論内容の精度低下と生成長増加を区別できない。

## 対応付き統計（探索的）

問題cluster単位のpercentile bootstrap 10000回、seed20260910。以下はpercentage points。

| 差 | 平均差 | 95% CI | 問題勝ち/負け | exact sign p |
|---|---:|---:|---:|---:|
| Random C1024 − native | -12 | [-22,-4] | 0/6 | .03125 |
| Random C2048 − native | -6 | [-14,0] | 0/3 | .25 |
| Random C4096 − native | -2 | [-6,0] | 0/1 | 1.0 |
| Random − Recency (C1024) | -2 | [-10,4] | 1/2 | 1.0 |
| Random − Recency (C2048) | 0 | [-8,8] | 2/2 | 1.0 |
| Random − Recency (C4096) | 0 | [-8,8] | 2/2 | 1.0 |

多重比較補正なし。非有意を同等と解釈しない。C4096の点推定がnativeから2ポイント差でも、CIは6ポイント低下まで含むため、事前目標「低下2ポイント以内」を統計的に確認できたわけではない。またC4096では44%の回答がeviction未発動である。

## 監査

- 全7セル50回答、欠損・重複・budget不適合・採点エラー0。
- 生成token数、EOSの初回停止、cap長、累積cache長、8層のeviction回数を全例検査。
- 各条件で同一問題/runのprompt token列、生成seed、eviction seed、sampling設定、gold、EOSが一致。
- cap時のthinking中boxedと、正常終了したfinal boxedを区別して保存。
- dataset hash、source hash、環境、GPU状態、各回答IDはmanifestに記録。

## メモリと速度の解釈

- GDN再帰状態48 MiB + conv1.5 MiB/系列。全条件共通。
- 圧縮Full Attention KV allocationは(C+64)×32 KiB/系列。C1024で34 MiB、C2048で66 MiB、C4096で130 MiB。位置indexと一時gather領域は別。
- 全checkpoint重みは約9.08 GB。batch1の短い系列では総GPUメモリの多くが重みなので、KVの大幅削減が同率の総メモリ削減にはならない。
- 約75–76 output tok/sは回答長が異なる生成workloadの診断値。native DynamicCacheと圧縮用preallocationのallocatorも異なるので、手法の速度優位を主張しない。

## 次の判断（本評価前）

1. SnapKVの実モデルno-op/eviction後参照テストを追加。
2. **事前固定pilot IDの先頭10問**を最大生成32768 tokensで確認。結果で問題を選ばない。
3. 条件は当初の主評価候補を維持: native、Random C1024/C2048、Recency C1024、SnapKV C1024。各1回。
4. 8kと32kのprefix/終了結果を比較し、capによる差と残る誤答・未終了を確認。
5. 問題数10では精度結論は出さず、cap率と時間見積もりからMATH500×2反復の実行条件を固定する。

GPQA、AIME、9B、Qwen3対照、遅延検索probeは後続。結果を良く見せるためにprompt/temperature/penaltyを途中で変えない。
