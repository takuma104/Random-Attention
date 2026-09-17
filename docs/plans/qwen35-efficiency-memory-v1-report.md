# Decode allocated-memory budgetを揃えたbatch比較

2026-09-17。4 cells / 24 windows、54分43秒。固定済みのメモリ校正・batch選定を再監査し、全入力hash、論理window、ソース/環境、計数監査が通過。

## 結果（32k context、throughput mode）

3隣接windowsのmedian。予算はnative preallocated B8のdecode peak allocated=17.1531 GiB。batchは事前の4kメモリ校正だけで選んだ。

| 条件 | Batch | decode peak GiB | 予算比 | aggregate tok/s | ms/step | 対native throughput |
|---|---:|---:|---:|---:|---:|---:|
| native preallocated | 8 | 17.153 | 100.00% | 487.8 | 16.40 | 1.00x |
| Random C1024 | 96 | 16.750 | 97.65% | 3643.9 | 26.35 | 7.47x |
| Random C2048 | 64 | 16.259 | **94.79%** | 2895.2 | 22.11 | 5.93x |
| Recency C1024 | 96 | 16.750 | 97.65% | 3645.4 | 26.33 | 7.47x |
| SnapKV C1024 | 72 | 16.445 | 95.87% | 3137.8 | 22.95 | 6.43x |

Random C2048は事前の95〜100%許容区間外。B72では予算を超えたため、固定ルールのunder-budget fallback B64を採用した。**この点を許容区間内のmemory-matched結果とは呼ばない。** 他3点は32k実測でも許容区間内。

Random1024の対応window throughput比は7.469（範囲7.452〜7.481）。これはbatchを12倍にした集約処理量の増加で、1系列のstepは約1.61倍遅くなった。Recencyはほぼ同じ速度。SnapKVの小さいbatchとscorerを含む実装での結果から、一般的なselector/servingの速度順位は断定しない。

各stepにCUDA barrierを置くmodeでは、native471.5、Random1024 3596.4、Random2048 2850.5、Recency3601.1、SnapKV3090.7 tok/s。こちらもsampling等は含まない。

## 一致させていないメモリ

| 条件 | decode peak reserved GiB | prefill/履歴込みrun peak allocated GiB |
|---|---:|---:|
| native B8 | 17.279 | 17.170 |
| Random1024 B96 | 17.553 | 16.796 |
| Random2048 B64 | 16.916 | 16.259 |
| Recency1024 B96 | 17.553 | 16.796 |
| SnapKV1024 B72 | 17.041 | 16.445 |

Random1024/Recencyはallocatedが少なくてもreservedはnativeより大きい。比較対象は**PyTorch decode peak allocated**であり、NVMLの総VRAM、最大serving batch、prefillを含む完全iso-memoryではない。測定後のGPU温度は71〜72°C、power snapshotは約492〜505W。開始/終了snapshotはエネルギー測定ではない。

## 範囲と精度との関係

- 同じteacher-forced notebookを各rowへ複製。入力hash・各系列の論理windowは同一だが、batchは異なる。全履歴を実modelで生成経路に通しており、人工KV/RNN状態は使っていない。
- model＋LM headのみ。sampling、EOS、異なるrequestの混在、scheduler、通信、prefill latencyは速度の分子/分母に含めない。実サービスQPSやp99ではない。
- 固定順の単一workload・3隣接windows。rangeはconfidence intervalではない。高batchの自由生成精度も未評価。
- 別のMATH B2評価では、表記レビュー後native94.7%、Random1024 72.4%、Random2048 82.4%、Recency69.3%、SnapKV69.4%。**同等精度を保つ高速化はまだ示していない。** 元の主評価のC2048精度維持目標は失敗したまま。
- 同batchの効率報告ではB1/2の速度改善はなく、B8 Random1024もpreallocated対照比1.30倍だった。ここでの大きな倍率の主な操作上の違いはbatch増加である。

証拠: `qwen35-efficiency-memory-v1-analysis.json`、校正/選定: `qwen35-efficiency-memory-selection-v1.json`、事前手順: `qwen35-efficiency-protocol.md`。各cellのraw SHAと対照SHAは機械可読reportに保存。

次は精度・メモリのfrontierを明らかにするため、Randomのより大きい容量を別の事前固定拡張として評価する。元の主比較を置き換えず、既知のnative結果と同じMATH500を再利用する拡張であることを明示する。
