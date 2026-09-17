# Random大容量frontier: 主評価後の事前固定拡張

固定: 2026-09-17、新規C4096/C8192のB2精度結果を見る前。

## 位置づけ

既存C1024/C2048はMATH500で精度低下を示した。一方、固定decode測定ではメモリ削減によってbatchを増やせた。**同等精度を保つ実用点はまだ見つかっていない。** このため次にRandomの容量を増やし、精度低下・eviction exposure・cache容量のfrontierを調べる。

- 元のC2048主比較の失敗を置き換えない。元結果を見て選んだ後続実験であり、最初から登録された主評価や独立した追試とは呼ばない。
- 同じMATH500・既知のnative対照を再利用する。B1の8k pilotではC4096の結果を見ているが、B2/32kの新しい2条件の結果は未観測。
- 原論文の主budgetは典型的traceの約4倍圧縮。Qwen3.5のnative平均生成長は約9701なので、32k上限をCで割った数値を典型的traceの圧縮率とは呼ばない。

## 条件・固定データ

- **Random prompt-protected C4096、C8192**、各500問×run0/run1＝計2000回答、1000 B2 batches。
- MATH500全問、同じ順序、同じproblem/run別generation/eviction seed。サンプル数・budget・samplingは途中結果で変更しない。
- 既存のfrozen `run_math_batch.py`、BF16 Qwen3.5-4B、同じrevision/環境、B2固定residency、max_new_tokens32768、r64、全prompt保護、temperature1/top_p.95/top_k20/presence_penalty1.5など元主評価設定をそのまま使う。
- Full Attention 8層のKVのみを圧縮し、DeltaNetは変更しない。新しい学習・量子化・kernel最適化をしない。
- nativeは完了済み`results/qwen35/main32k_b2_v3`の1000回答を再利用し、再生成しない。ソース/モデル/データ/環境/sampling/全問題IDとpaired prompt/seedを照合。旧raw全体SHAが既存reportと一致することも確認する。
- 出力: `results/qwen35/frontier32k_b2_v1`。旧結果やscoreは書き換えない。

## 数値gateと監査

- 新しい別scriptでC4096/C8192・B2の実checkpoint teacher-forcingを実施し、両capacityで2回のevictionまで動かす。
- eviction前の複数logical checkpointsでnativeとの全vocabulary logits完全一致を要求。eviction後は既存と同じ独立eager attention参照（BF16 rtol=.03/atol=.03）で照合。
- main生成ソースは変更しない。数値gateに失敗したら精度実験を開始せず調査する。
- 完了回答のEOS/length/position/cache/event/batch/採点/journalの既存監査を使う。nativeと比較可能なeviction前の生成prefixも照合する。
- 最初の20問で運用監査、その後は必要に応じて完了prefixをCPU監査する。途中の有意差・目標達成判定は行わない。NaN/CUDA/採点エラー等は停止調査し、既存回答を再生成しない。

## 最終解析

- 主指標は元主評価と同じlegacy `final_correct`。全cap例を分母に残す。
- 新規比較2つ: **Random4096−native、Random8192−native**。
- 500問題clusterをresampleして同一問題の両runを保持。10000 bootstrap、seed20260917。
- この新規2比較familyについてBonferroni方式の**個別97.5%両側percentile CI**（quantiles .0125/.9875）を用い、同時95%のnominal coverageを意図する。bootstrapの近似性や後続実験である点を明示する。
- 2pp低下以内の目標を支持したとするのは、その条件の下限が**-0.02より大きい**場合だけ。非有意を同等としない。旧条件を含む研究全体の多重性を補正したとは主張しない。
- 参考として通常95% CI、EOS-required/paper metric、cap率、生成長、eviction率、native長による層別を報告。native生成長の区間は[0,4096)、[4096,8192)、[8192,16384)、[16384,32769)。層別はrow単位の記述集計で有意差検定しない。C4096−C8192等は探索的。
- 既知のgrader表記問題があるため、旧レビュー方針に沿った別のmasked sensitivity reviewも行う。既存nativeの判断を恣意的に変更せず、新規条件にも等しく適用し、全overlayをSHA-linked sidecarとして保存する。未解決ambiguityはlegacyのまま、欠落boxは救済しない。AI-assisted reviewであり独立human審査とは呼ばない。
- legacyと表記レビュー感度の結論が異なれば両方示し、頑健な精度維持とは称さない。新しいレビューを元主評価の事前登録に遡って組み込まない。
- この生成runの速度は生成長が異なるため、固定workload性能として扱わない。高容量での効率再測定は精度結果を見て別手順で固定する。

## 実行

```bash
.venv/bin/python scripts/qwen35/run_math_batch.py \
  --out results/qwen35/frontier32k_b2_v1 --subset all --limit 500 \
  --runs 2 --max-new-tokens 32768 --cells random_pp:4096,random_pp:8192
```

所要時間は約2日を暫定目安とし、完了20問の実測で更新。精度に基づく早期停止や条件追加はしない。Qwen3対照・9B・別datasetはこのfrontierの後に検討し、単一GPUを共有して同時には動かさない。
