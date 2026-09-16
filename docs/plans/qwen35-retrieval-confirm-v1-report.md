# 遅延参照probe v1: held-out確認結果

2026-09-17 05:50 JST完了。32個の新規nonce case×2 eviction seeds×6条件×5 gap、960 batches/1920 row readouts。独立単位は32 cases。19.125 GPU時間、監査通過、実行中の新しいXidなし。

## 設計

任意key→value factは保護promptではなく生成領域へteacher-force。中立な算術文で遅延させた後に四択を読み出す。生成tokenのsamplingやEOS/32k capによる打ち切りは介在しない。

校正8 casesと確認32 casesは別nonce。全条件でtoken列、prompt、絶対position、case/seedを対応させた。原fact/value tokenの保持位置は最終query tokenのforward直前に測定した。

## 四択正答率

各セルの分母は64 readoutsだが、native等の同一2rowを独立問題とは数えない。

| 条件 | gap0 | 512 | 2048 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|
| native | 100% | 100% | 100% | 100% | 100% |
| Random C1024 | 100% | 100% | 79.69% | 25% | 25% |
| Random-shared C1024 | 100% | 100% | 75.00% | 25% | 25% |
| Recency C1024 | 100% | 100% | 28.13% | 25% | 25% |
| SnapKV C1024 | 100% | 100% | 100% | 100% | 100% |
| Random C2048 | 100% | 100% | 100% | 29.69% | 25% |

Nativeの平均正解条件付き確率は遠距離でも約0.986。SnapKVはgap2048/8192/16384で0.964/0.921/0.905。Random1024は0.548/0.259/0.253。label全体の確率massも別途記録している。

## 事前固定したhead多様性比較

Random1024−Random-shared1024の正答差を、case内で2 seedsと3遠距離gap（2048/8192/16384）にわたり平均。32 casesの対応付きbootstrap（10000回、seed20260916）:

- 差 **+1.5625pp**、95% CI **[−1.0417, +4.1667]pp**。
- case単位の勝ち/負け/同点: 5/2/25、exact sign p=0.453125。
- **このprobeでhead別Randomの優位を確認する事前基準は満たさなかった。** 非有意を同等性の証拠とも扱わない。
- gap2048だけを取り出して主比較を置き換えない。他selectorやgapごとの比較は探索的。

## 元valueのKV保持（探索的）

「どこかのFull Attention layer/KV headがvalue span全tokenを保持」のrow割合:

| 条件 | gap2048 | gap8192 | gap16384 |
|---|---:|---:|---:|
| Random1024 | 98.44% | 1.56% | 0% |
| Random-shared1024 | 81.25% | 1.56% | 0% |
| Recency1024 | 0% | 0% | 0% |
| SnapKV1024 | 100% | 100% | 100% |
| Random2048 | 100% | 45.31% | 1.56% |

Random1024のolder-generated head間Jaccardはgap2048で約0.361、遠距離で約0.323。一方sharedは常に1。選択多様性そのものは確認できるが、「いずれかのheadが保持」だけで正しい読み出しが保証されるわけではない。

## 解釈

- 小予算Randomは長距離の任意対応を安定保持できず、SnapKVはこの合成lookupでは優れる。MATHではRandomがSnapKVより良かったので、lookup能力と長い自由推論の成績を同一視できない。
- gap0/512はeviction前で全条件logitsがnativeと一致。単にtaskが難しすぎて全条件が失敗している状況ではない。
- 原factのKVが消えても後続tokenのKVやDeltaNet状態に情報が転写され得る。retention/readoutの関連から、DeltaNet固有の因果効果やMATHでのcap増加の原因を断定しない。
- 32 casesの限られた任意key/color-word課題。100%はこの標本での観測であり、あらゆる遅延参照の保証ではない。

機械可読記録: `qwen35-retrieval-confirm-v1-analysis.json`。
Data SHA256: `369b1d5ecbd19307ff6f3435ab7fd7c462a2b353ead95c4aa61f7df07ee99c55`。
