# 表記差の感度分析 v1（事後的・AI-assisted review）

2026-09-16。これは事前登録した主スコアの置換ではない。生成・legacy grader・5000回答JSONを変更せず、in-memory overlayとSHA256付きsidecarで集計した。

## 方法・範囲

- 全5000回答から、EOS不正解全66例、不正解の非null final box、cap正答全6例、および固定hashで抽出した自動正答100例を選択。
- 重複を除く177回答、同一問題/boxed文字列でまとめた122 packets。全てのcondition/run名をpacketから隠し、decisionを記録してからmappingへ戻した。
- 最初の100問で既に見た例は非blindであったことを開示。レビュー主体はcoding assistantであり、独立した人間の採点者ではない。
- 81 packetsはgoldとの文字列完全一致を機械的に確認。残り41 packetsは問題・gold・boxedを照合し、units/base/出力形式の文脈が必要な例では異なるfinal textも確認した。
- 元のboxed抽出規則は変更せず、missing/empty boxを推論本文から救済しない。通常のordered tupleを勝手に集合化しない。
- 全件で発見したinterval/units/ordinalの同値表記も事後レビューに含めた。これらを事前登録した規則とは称さない。

## 補正

56回答を表記差のfalse negativeとして補正。内訳:

| 種類 | 回答数 |
|---|---:|
| 文脈で明らかな単位の省略 | 19 |
| intervalとinequality/集合所属の同値表記 | 14 |
| 12と12th grade等のordinal | 9 |
| eastとLaTeX text wrapper | 9 |
| plus/minusと明示的な2根 | 2 |
| option labelのbold/text差 | 2 |
| 要求された八進数のsuffix省略 | 1 |

自動正答100例のsampleからfalse positiveは見つからなかった。ただし全正答の無誤判定を証明するものではなく、gold自体の正しさを全問で再証明したわけでもない。

## 結果

| 条件 | Legacy final | 表記差レビュー後 | 増加回答数 |
|---|---:|---:|---:|
| native | 93.3% | 94.7% | 14 |
| Random C1024 | 71.4% | 72.4% | 10 |
| Random C2048 | 81.3% | 82.4% | 11 |
| Recency C1024 | 68.2% | 69.3% | 11 |
| SnapKV C1024 | 68.4% | 69.4% | 10 |

- Random2048−native: **−12.3pp、95% CI [−14.8,−9.8]pp**。
- Random1024−Recency1024: **+3.1pp、CI [1.5,4.8]pp**。
- Random1024−SnapKV1024: **+3.0pp、CI [1.5,4.6]pp**。
- 後2比較へ同じHolm手順を適用したpは両方1.23e-4。これらは事後的な感度分析の統計であり、新たな事前登録主結果ではない。
- 表記差を補正しても、Randomの同容量selector比較での優位と、C2048で精度低下2pp以内を達成しないという結論は変わらない。
- Cap率・生成長・メモリは不変。元論文互換の別extractorのスコアはここでは再採点していない。

## 未解決・残るEOS不正解

補正後もEOS不正解扱いは10回答:

- **5回答は解釈の曖昧さを残してlegacy判定を維持**。3例は逆余接のbranchによって負の解の扱いが変わる問題、2例は問題文の「region」とscalar volumeのgoldとのずれ。
- 3回答は非emptyなfinal boxがない。抽出契約を維持して不正解のまま。
- 2回答は確認できる代数/算術誤り（定数1の欠落、hyperbolaの中心等からの合計16を18とした例）。

未解決5回答を全て受理する上側の感度値はnative95.0%、Random2048 82.6%、他は不変。根拠なくこれを正式正答率にはしない。

生成lossの中心は依然として32k capだが、これだけで内部機構・DeltaNetへの因果帰属はできない。

## 再現・不変性

- `prepare_grading_review.py`: condition-masked packetとprivate mappingを生成。
- `qwen35-grading-decisions-v1.json`: mappingを戻す前に固定した判断と理由。
- `analyze_grading_review.py`: SHAを照合し、元ファイルを変更せず全条件を再集計。
- `qwen35-main32k-grading-sensitivity.json`: 統計、全177件のsidecar、変更56件の追跡情報。
- 解析後も全回答のaggregate SHA256は主評価と同一: `b3ad6d1939371a128587508e329ab92c2b3d09fa7d84a74ba07b41d056d26a66`。
- Missing-box救済禁止・unresolved維持等のCPU testsを追加。統計testと合わせ6 tests passed。
