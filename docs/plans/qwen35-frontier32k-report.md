# Random C4096/C8192: 主評価後の容量frontier拡張

2026-09-19。500問×2反復×2条件＝2000回答、1000 B2 batches、46.392 GPU生成時間で完了。既知のnative1000回答を再利用する事前固定の後続実験であり、独立した追試や元主比較の置き換えではない。

## Legacy主指標

| 条件 | Final | Paper | EOS-required | Cap率 | 平均生成tokens | Eviction exposure |
|---|---:|---:|---:|---:|---:|---:|
| native（既存） | 93.3% | 93.4% | 93.3% | 4.9% | 9701 | 0% |
| Random C4096 | 87.1% | 87.1% | 87.0% | 11.3% | 10661 | 66.3% |
| Random C8192 | 91.4% | 91.5% | 91.4% | 7.1% | 10047 | 39.1% |

| 新規比較 | 差 | 通常95% CI | 個別97.5% CI（2比較Bonferroni） | 2pp低下以内の目標 |
|---|---:|---|---|---|
| C4096 − native | −6.2pp | [−8.3, −4.3] | [−8.5, −4.0] | 支持されず |
| C8192 − native | −1.9pp | [−3.3, −0.6] | [−3.5, −0.4] | 支持されず |

C8192の点推定は2pp以内だが、事前基準はCI下限が−2ppを上回ること。**精度維持が確立したとは言えない**。これは「損失が必ず2ppを超える」と証明した意味でもない。旧C2048の主比較の失敗も変更しない。

全cap例を分母に含め、500問題clusterの両runを保持する10000 bootstrap（seed20260917）。Bonferroniはこの新規2比較だけで、研究全体の適応的な検討を補正したものではない。bootstrap coverageは近似。

## 探索的な長さ別観察

native生成長で層別した記述集計では、≥16384 tokensの202回答におけるfinal accuracyはnative72.77%、C4096 44.55%、C8192 63.37%。C8192の総差はこの層の差と一致するが、他層で個々の回答が一切変わらなかった意味ではなく、因果的な機構の証明でもない。

平均native生成長は9701 tokens。32k上限/Cを典型的traceの圧縮率とは呼ばない。高容量ではevictionが作動しない回答も多い。

## 監査・範囲

- 2000回答の欠落/重複/採点例外なし、pending journalなし。source/model/data/environment/sampling/paired prompt/seed整合性を確認。
- 全2000回答のeviction前prefix（計9161707 tokens）がnativeと完全一致。旧main5000回答と途中400問の1600回答のSHAは不変。
- 20/100/250/400問の運用監査で条件変更や中間仮説検定なし。全期間kernel logに新しいNVRM/Xid記録なし。
- B2共有peak allocatedはC4096 9.005 GiB、C8192 9.224 GiB。自由生成の長さが異なるので、このrunの時間やメモリを固定workloadの速度倍率としない。
- 表記graderの別感度レビューも完了。新規27 false negativesを補正し、native94.7%（旧判断固定）、C4096 88.5%、C8192 92.7%。Reviewed C8192−nativeは−2.0pp、97.5% CI [−3.8,−0.4]で目標判定は変わらない。詳細は`qwen35-frontier-grading-sensitivity-report.md`。raw scoreは不変。

Protocol: `qwen35-capacity-frontier-protocol.md`。機械可読結果: `qwen35-frontier32k-analysis.json`。
新規raw SHA256: `993523c1f7287262f4ade9464c29ecc39459fe42b6a3f04fb47b91f6014f479e`。
