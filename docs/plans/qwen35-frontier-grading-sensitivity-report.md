# Frontierの表記grader感度レビュー

2026-09-19。Legacy主解析とは別のcurated overlay。新規2000回答と旧main5000回答はbyte単位で不変。

## 手順

- 新規全32 EOS-incorrect、その他incorrect nonnull box、cap-correct、固定hashで選んだ100 acceptedを対象とする既存selectorを同じ条件で適用。合計133 records / 108 packets。
- condition/runを隠したquestion/gold/boxedをレビュー。73 packetsはgold文字列と完全一致。35非exact packetsを確認し、units/base表記の8 final-text variantsも読んだ。
- 108判断をcommit `54ee93e`で保存・pushした**後**にprivate mappingを読み、条件別に集計した。
- 旧mainのreviewと新規条件の集約結果は既知。再登場する問題は既知の場合があり、独立したblind human審査ではない。**AI-assisted** review。
- nativeは旧レビューの判断をそのまま再利用し、14 correctionsを変更していない。新規条件の結果に合わせて再判定していない。

## 結果

| 条件 | Legacy final | Reviewed final | Reviewed EOS-required | 新規/既存upgrades | Unresolved |
|---|---:|---:|---:|---:|---:|
| native（固定） | 93.3% | 94.7% | 94.7% | 14（既存） | 3 |
| Random C4096 | 87.1% | 88.5% | 88.4% | 14 | 2 |
| Random C8192 | 91.4% | 92.7% | 92.7% | 13 | 1 |

新規**27 false negatives**を補正: units8、明示的な根5、interval4、ordinal4、text wrapper4、文脈で確認した進数suffix2。固定100 accepted sampleでfalse positiveは見つからなかったが、全件のgrader正しさを保証しない。

新規EOS-wrongの残り5回答は、逆cotangentのbranchに関する未解決3件と、欠落/空のfinal box2件。未解決はlegacyを保持し、boxがない回答を本文から救済しない。

### 固定したfrontier基準を感度解析にも適用

| Reviewed比較 | 差 | 通常95% CI | 個別97.5% CI（新規2比較Bonferroni） | 下限>−2pp |
|---|---:|---|---|---|
| C4096 − native | −6.2pp | [−8.3, −4.3] | [−8.6, −4.0] | no |
| C8192 − native | −2.0pp | [−3.5, −0.6] | [−3.8, −0.4] | no |

**Legacyの結論は変わらない。C8192でも2pp以内の精度維持は確立していない。** nativeを含め全unresolvedを正解扱いする参考シナリオでもnative95.0%、C4096 88.7%、C8192 92.8%、差−6.3pp/−2.2pp（97.5% CI [−8.7,−4.1]/[−4.0,−0.5]）で同じ判定。

## 再現性と範囲

- `prepare_frontier_review.py`は元packet selectorを再利用し、今回のprior exposureを明記する。
- `record_frontier_review_decisions.py`はこの特定の108 judgmentsを再現するだけで、汎用graderではない。
- `analyze_frontier_review.py`は元answer SHA、packet/mapping/decision hash、gold/question、文脈text、固定native overlayを監査し、in-memory scoreのみを変更する。CPU tests計8件通過。
- `qwen35-frontier-grading-decisions-v1.json`、`qwen35-frontier-grading-sensitivity.json`に判断・根拠・各record SHA・旧新scoreを保存。
- 同じMATH500・既知native対照を再利用する後続実験。元C2048主比較の失敗を置き換えない。表記レビュー後の数値だけを選んで元の主結果とは呼ばない。
