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

TF32は無効を要求。元の不合格を合格として書き換えない。Qwen3.5のfrozen source/既存結果は変更しない。

## 診断結果

`qwen3-control-attention-diagnostic-1948`は70秒で完了。BF16とFP32各540 snapshots（36層×3 policies×5時点）、各2,764,800 projected要素を検証。集約とraw SHAは`qwen3-control-attention-numerics.json`。

- native/noop、および3 selectorsのeviction前full-vocabulary logitsは両dtypeで完全一致。captureしたattentionと同じQ/K/VによるSDPA再計算も全snapshot完全一致。
- 元のBF16 gateは61 snapshots、計2609要素で不合格。SDPA対BF16 eagerの最大絶対差は2.0、最大相対L2は3.972%。最初に止まった1要素だけの問題ではない。
- 最初の不合格はRandom、layer24、logical321。値はSDPA **−.09765625**、BF16 eager **−.056640625**、同じ入力のFP32 oracle **−.09651950**。この点ではSDPAの方がoracleに近い。
- 同じBF16入力のFP32 oracleに対するprojected出力の最大相対L2（snapshot全体）はSDPA **.3040%**、BF16 eager **4.1358%**。projection前もSDPA **.2067%**、eager **3.0617%**。SDPA/oracleのprojected最大絶対差は.88257で、絶対差だけでは層ごとのスケールを反映できない。
- 全モデルFP32 passのSDPA/eager projected最大絶対差は**6.1035e−5**、最大相対L2 **1.1137e−6**。固定したrtol=1e−4/atol=1e−4を全件通過。

これらはBF16 eager側の中間丸めを含む参照差という説明を支持する。公式forwardやcacheの修正はしない。BF16とFP32の全モデル生成が同一であるという主張ではない。

## 改訂gate v2（追加trace実行前に固定）

元のelementwise BF16 eager gateは不合格のまま保存し、新gateは**同じQ/K/VとweightのFP32 eager oracle**を参照する。

- native/noop/eviction前full logitsと同じ入力のSDPA replayは引き続き完全一致を要求。
- projection前は**各batch row・query headごと**、projection後は**各batch rowごと**に `||error||₂/||reference||₂ ≤ .01` および `||error||∞/||reference||∞ ≤ .01` を要求。非有限値は拒否する。
- 1%はBF16の相対spacing（最大約.78%、rounding unit約.39%）を踏まえた工学的許容値で、診断後に採用したことを明記する。要素ごとのゼロ付近の打ち消しを相対誤差の分母に使わず、別rowや強いheadによる隠蔽も避ける。誤差の一般的な数学的保証ではない。
- 元のgate不一致件数/最大絶対差も並行記録し、消さない。
- 同じcalibration C256を再検証した後、**未使用のforced trace・prompt・eviction seeds (17,23)** でC1024/C2048をそれぞれ2回のevictionまで検証する。この追加traceの結果で閾値を調整しない。
- sampled replay/EOS/counterと全500 prompt適合性も検証する。accuracy pilotは全gateの通過後にprotocolを固定してから開始する。

`test_numerics.py`はBF16 rounding、ゼロ一致、小さな異常rowが大きなrowに隠れないこと、非有限値拒否を検証。adapter testsと合わせ7件通過。
