# Qwen3 BF16 eager-reference gateの調査

2026-09-19。実checkpoint gate (`9bc057c`) が19秒で停止したため、精度pilotは開始していない。

- job: `qwen3-control-validation-55d7`。
- `rtol=.03, atol=.03`のSDPA対eager比較で、停止したtensorの1/5120要素が不一致。
- 該当要素の絶対差.041015625、相対差.419921875、index=(1,0,312)。これは全検証tensorを通した最大誤差/不一致率ではない。
- 最後の進捗表示はlogical288。失敗時のlayer/policy/logical位置は別diagnosticで特定する。
- CUDA例外ではなく数値assertion。許容差をそのまま広げたり、失敗を除外してpilotを始めたりしない。

## 切り分け（実行前に記録）

別script `diagnose_attention_numerics.py` で同じforced trace、C256/r64/B2、512 forwardsを調べる。

1. 実際のo_proj入力をcaptureし、同じQ/K/V・maskで再計算したSDPAと完全一致するか確認。
2. 同じQ/K/VをFP32へcastした独立eager参照を作り、BF16 SDPA/eagerのattention出力とprojected出力をそれぞれ比較する。FP32 SDPA対FP32 eagerも照合する。
3. 元のBF16 toleranceを超えた座標におけるSDPA/eager/FP32 oracle値を保存し、低精度softmaxやprojectionによる打ち消しの影響を分ける。
4. モデル全体をFP32へcastした別passを実施。native/noop/eviction前一致と、eviction後FP32 output対eager oracle（rtol=1e-4/atol=1e-4）を検証する。

TF32は無効を要求。新しい数値gateを決めるなら結果・根拠・変更を明記して改めて検証する。元の不合格を合格として書き換えない。Qwen3.5のfrozen source/既存結果は変更しない。
