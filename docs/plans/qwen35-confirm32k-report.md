# 32k確認とbatch=2への移行判断 (2026-09-10)

## 32k確認（B1）の結果

事前に固定したpilot IDの先頭10問×5条件×1回、最大生成32768 tokens。
07:16–09:46、約2.50 GPU時間。欠損・重複・採点エラー・cache counter不整合なし。

| 条件 | 正答/10 | 32k cap率 | 平均生成tokens |
|---|---:|---:|---:|
| native | 10 | 0% | 7157 |
| Random C1024 | 8 | 20% | 12564 |
| Random C2048 | 9 | 10% | 10823 |
| Recency C1024 | 7 | 30% | 17838 |
| SnapKV C1024 | 7 | 30% | 16707 |

- 全条件でEOS終了した回答は全てfinal boxed正解。残る不正解扱いは全てcap。
- 10問だけなので手法の優劣・非劣性は未確定。
- 対応する8k実行がある4条件×10問、**全40回答でtoken列のprefixが完全一致**。EOS済み回答は全token一致、cap回答は先頭8192 tokensが一致。
- 同じ10問の8k→32k cap数: native3→0、Random1024 5→2、Random2048 4→1、Recency1024 6→3。
- 生成上限延長で回復する一方、圧縮で非常に長い反復推論・未完了が残る。主評価は予定通り32kで行い、正答率と生成長/cap率を必ず併記する。
- 機械可読記録: `qwen35-confirm32k-analysis.json`, `qwen35-8k-32k-prefix-audit.json`。

## 本評価の計算量

この10問の単純外挿では、500問×2反復×5条件をbatch1で約250 GPU時間（10.4日）。選んだ10問は少数であり時間予測には幅がある。

精度条件を変えず、当初計画の同一prompt反復batch化を先に検証する。B=2で各問題のrun0/run1を同時実行する。全手法でbatchを揃え、B1 pilotとB2本評価のスコアは混ぜない。

## Batch数値検証

- 小型hybrid modelのB2 vs 独立B1、row別eviction RNG、B1 sampler互換性のテスト通過。
- 実4B BF16のbatch形状変更ではpointwise tolerance (.15 absolute, .03 relative)に一部が不適合。単に無視せずFP32参照を追加。
- native FP32＋公式torch-only DeltaNetでB1/B2差は最大2.86e-5 logits、最大KL4.88e-8。
- BF16＋FLAでB1/B2差は最大.5625、平均.0424 logits、8 step最大KL.001594。
- BF16 B1/FP32差は最大.336、BF16 B2/FP32差は最大.285。8 step argmax一致率は各比較100%。
- この測定はBF16のbatch形状による丸め差と整合的。新基準は最大絶対差<1.0、平均絶対差<.1、token KL<.005。実32 step確認で最大KL.001727、通過。
- これは長い自由生成がB1/B2で一致するという主張ではない。全比較条件をB2で固定する。

## 固定1024-output benchmark（短い実装診断）

| 条件 | B1 aggregate tok/s | B2 aggregate tok/s | 倍率 |
|---|---:|---:|---:|
| native | 75.79 | 147.08 | 1.94 |
| Random C1024 | 75.77 | 145.41 | 1.92 |
| SnapKV C1024 | 72.82 | 139.85 | 1.92 |

長い系列のserving性能ではない。早く終了したrowをresidentに残すため、長さが異なる反復では約1.9倍をそのまま得られない。

## B2実装・評価規則

- row別の独立generation RNGとeviction RNG。layerごとも独立。
- 終了rowは同じbatch形状を保つためresidentに残し、追加出力は回答に含めない。他rowへ情報は混ざらない。
- 各回答のcache長・eviction回数・終了理由はその回答自身のEOS/cap時に保存。
- `request_latency_seconds`は実レイテンシ。`elapsed_seconds`はbatch wall time/2のGPU時間配賦で、全回答の和が総batch時間になる。共有peak memoryを個々の回答固有の消費と解釈しない。
- 2回答のraw結果を一つのjournalへ原子的に保存してから個々を採点。部分採点後の再開でも生成を繰り返さない。
- 本評価前に2問×3条件×2反復、8k上限のB2 smokeを行う。単体テストでは片方が先にEOSとなる場合と部分採点からの復旧を検証する。
