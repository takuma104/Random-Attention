# 固定decode workload: batch・cache実装を揃えた効率測定

2026-09-17。18 cells / 324 windows、約2時間11分。入力/計数/メモリ監査通過。evictionなしのDynamicCacheとpreallocated adapterは、全54対応window終端で全vocabulary logitsが完全一致。

## 32k contextでの結果

throughput modeの3連続windowsのmedian。単位tokens/sはbatch全体。peak allocatedはdecode windowのPyTorch確保量（GiB）。

| Batch | 条件 | tokens/s | peak GiB |
|---:|---|---:|---:|
| 1 | native DynamicCache | 83.46 | 9.61 |
| 1 | native preallocated | 82.14 | 9.56 |
| 1 | Random C1024 | 82.15 | 8.55 |
| 1 | Random C2048 | 81.92 | 8.59 |
| 1 | Recency C1024 | 82.12 | 8.55 |
| 1 | SnapKV C1024 | 78.70 | 8.57 |
| 2 | native DynamicCache | 148.06 | 10.73 |
| 2 | native preallocated | 158.54 | 10.65 |
| 2 | Random C1024 | 157.23 | 8.64 |
| 2 | Random C2048 | 157.07 | 8.71 |
| 2 | Recency C1024 | 156.44 | 8.64 |
| 2 | SnapKV C1024 | 150.72 | 8.68 |
| 8 | native DynamicCache | 285.11 | 17.52 |
| 8 | native preallocated | 487.81 | 17.15 |
| 8 | Random C1024 | 636.27 | 9.15 |
| 8 | Random C2048 | 639.93 | 9.45 |
| 8 | Recency C1024 | 637.02 | 9.15 |
| 8 | SnapKV C1024 | 610.09 | 9.35 |

### 対応windowでのRandom1024 throughput比（median）

| Batch | 対DynamicCache | 対preallocated |
|---:|---:|---:|
| 1 | 0.984x | 0.999x |
| 2 | 1.060x | 0.992x |
| 8 | 2.232x | 1.304x |

B1/2では同じpreallocated adapter対照に対して明確な速度改善は見られない。B8では約1.30倍。DynamicCacheだけを対照にして2.23倍と述べると、cache実装の差を圧縮自体の効果に混ぜてしまう。

B8・32kで各stepにbarrierを置くmodeのaggregate tokens/sは、Dynamic280.29、preallocated471.46、Random1024 595.97、Random2048 598.87、Recency597.92、SnapKV572.65。

## Contextによる違い（B8 throughput）

| 条件 | 8k | 16k | 32k |
|---|---:|---:|---:|
| DynamicCache | 525.8 | 411.8 | 285.1 |
| preallocated | 637.8 | 578.1 | 487.8 |
| Random1024 | 633.7 | 636.2 | 636.3 |
| Random2048 | 642.2 | 640.4 | 639.9 |
| Recency1024 | 636.8 | 638.1 | 637.0 |
| SnapKV1024 | 607.1 | 610.1 | 610.1 |

## 正しい範囲の解釈

- 入力token列・論理window・batchを揃え、実modelで履歴を作った。KVやDeltaNet状態を人工的に注入していない。
- model＋LM headのteacher-forced測定。sampling、EOS、ragged scheduling、通信、複数の異なるrequestは含まない。実サービスのQPS/p99や自由生成速度とは異なる。
- preallocated nativeは最大33536+64 slotsを最初から確保する。8k/16kでのbacking確保量も32k相当なので、短いcontextのlive KV量と混同しない。
- preallocated対照は絶対position記録等のadapter bookkeepingも持つ。Dynamicとの差を純粋にallocator単独の効果とは呼ばない。
- 固定method順、3隣接windows、単一のteacher-forced notebook。独立試行に基づく有意差や汎用的性能保証ではない。GPUの温度/clock/powerもraw cellに保存した。
- BF16、FLA有効、causal convolutionは公式Torch fallback。最適化済みserving stackの上限性能ではない。
- 同じ容量のMATH B2評価では精度低下があった（表記差レビュー後native94.7% vs Random1024 72.4%、Random2048 82.4%）。速度/メモリ改善は精度無料の改善ではない。B8の自由生成精度はこの測定では評価していない。

機械可読記録: `qwen35-efficiency-v1-analysis.json`。
Data SHA256: `ce1ec22d62490b3db78cf391a96a8ccf8baafab127886008420cc95b0ca03d6e`。
次はnative preallocated B8のdecode peak allocatedを予算に、圧縮のbatchをメモリだけで選ぶ別実験を行う。
