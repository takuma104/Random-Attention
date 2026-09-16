# Qwen3.5-4B: MATH500主評価（legacy grader）

2026-09-16 09:25 JST完了。500問×2反復×5条件=5000回答。B2、最大生成32768 tokens。
生成実装`cc919ad`、事前固定protocol `fa2864a`。生成141.719 GPU時間、開始から約5日23時間。

## 完了監査

- 全5000回答、欠損・重複・budget除外・採点例外なし。未処理journalなし。
- EOS、論理/物理cache長、eviction回数、paired seed/prompt、batch時間配賦、frozen source hashの監査通過。
- 途中CUDA停止1回。同じseedで未保存batchを再実行し、既存2238回答のSHA256不変を確認。再開後の約81時間、新しいXidなし。原因は未確定。
- completed data SHA256: `b3ad6d1939371a128587508e329ab92c2b3d09fa7d84a74ba07b41d056d26a66`。

## 固定したlegacy graderによる結果

**以下は自動採点器のスコア。既知の表記差によるfalse negativeを含む。** 別versionの感度分析を準備中。主評価JSONを書き換えたり、途中で採点規則を変更したりしていない。

| 条件 | final boxed | 元論文互換 | 32k cap率 | 平均生成tokens |
|---|---:|---:|---:|---:|
| native | 93.3% | 93.4% | 4.9% | 9701 |
| Random C1024 | 71.4% | 71.5% | 27.8% | 14165 |
| Random C2048 | 81.3% | 81.4% | 17.5% | 11703 |
| Recency C1024 | 68.2% | 68.6% | 30.8% | 14789 |
| SnapKV C1024 | 68.4% | 68.6% | 30.4% | 14755 |

EOSを要求した正答率は順に93.3/71.2/81.1/68.0/68.4%。capでも正しいfinal boxがある6回答を主指標には含めている。

## 事前固定した主比較

問題単位の対応付きcluster bootstrap、10000回、95% CI。2反復を独立問題として数えない。

- **Random C2048 − native: −12.0pp、CI [−14.5, −9.6]pp。** CI下限が−2ppを上回るという精度維持目標を達成しない。このbudgetで非圧縮と同等とは結論できない。
- **Random C1024 − Recency C1024: +3.2pp、CI [+1.6, +4.8]pp。** 問題単位のexact sign test、2比較Holm補正p=4.98e-5。
- **Random C1024 − SnapKV C1024: +3.0pp、CI [+1.5, +4.6]pp。** Holm補正p=1.06e-4。
- selector比較は事前基準（CI方向一致、補正p<.05）を満たす。ただしこれはこのモデル・sampling・budget・32k制限・legacy graderにおける結果であり、全般的な優越性を意味しない。
- 副次的なRandom C1024 − native差は−21.9pp、CI [−25.4, −18.5]pp。

## 解釈と制限

- 全24 DeltaNet層を変更せず、8 Full Attention層だけをevictしても、C1024/C2048で精度低下とcap増加が観測された。hybridであればこの小予算が安全だという証拠にはならない。
- cap率増加が主要な差だが、自由生成の差だけで内部機構やDeltaNetへの因果帰属はできない。
- EOS不正解判定は全条件合計66回答。100問時点の13回答を調べたところ12例が表記差のfalse negative、1例は本当のモデル誤答だった。全件で同じ割合と仮定しない。最終の表記差レビューを別途行う。
- 原論文の完全再現やQwen3とのarchitecture因果比較ではない。SnapKVもdecode adaptationで、元実装とbit同一とは主張しない。
- B1 pilotのスコアと混ぜていない。pilotで見た問題が本評価500問に含まれる点は開示する。

## メモリ・時間（運用診断）

- FA KVの確保量/sequence: C1024で34 MiB、C2048で66 MiB。DeltaNet recurrent48 MiB＋conv1.5 MiB/sequenceは不変。
- B2共有peak allocated: native10.75 GiB、Random1024 8.84、Random2048 8.89、Recency8.84、SnapKV8.93 GiB。
- 有効aggregate速度は125–131 tokens/s、useful row-step比88–92%。早く終わったrowをresidentに残す損失を含む。
- 生成長とallocatorが異なるので、この速度を一般的な圧縮による高速化倍率として使わない。iso-workload/allocatorを揃えた測定は別実験。

機械可読統計: `qwen35-main32k-analysis.json`、全件監査: `qwen35-main32k-progress-500.json`。
次: [表記差感度分析](qwen35-grading-sensitivity-plan.md)、[遅延参照probe](qwen35-delayed-retrieval-protocol.md)、iso-workload効率測定。
