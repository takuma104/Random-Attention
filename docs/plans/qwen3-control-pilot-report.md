# Qwen3-4B control pilot: completed exploratory results

2026-09-19。**同じ50問×2runs、native/Random1024/Random2048の300 answersが完了**。Qwen3でもこのpilotでnear-losslessな圧縮は確認できていない。ただし小標本の探索的結果であり、2pp精度維持の確証試験でも、DeltaNetの因果効果の検証でもない。

## 固定条件・完全性

- Protocol `13802cd` (`qwen3-control-pilot-protocol.md`) は最初のQwen3 MATH smoke outcomeより前に固定。公式Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`、BF16/SDPA、36 FA層、full RoPE/no output gate。
- 同じ歴史的50 pilot IDs、runs0/1、same-prompt B2、output cap32768、thinking、temperature.6/top_p.95/top_k20/presence0、EOS=[151645,151643]。Cは全protected promptとrecent64を含み、C+64でattention後にCへcompact。
- 原Qwen3 reproductionおよびfrozen Qwen3.5のsource/環境は変更していない。新しいQwen3 hookとread-onlyの汎用cache/sampler/grade recoveryを使用。
- job `qwen3-control-pilot32k-9025`、09-19 10:04:35–13:04:04 JST、exit0。**2.969622 GPU generation-accounting hours**、150 batches。欠損・余分なanswer・pending journal・grading例外なし。
- Source/model/data/36-layer memory/EOS/pairing監査通過。**200 native-versus-compressed pre-eviction prefixes、296,609 tokensが完全一致**。
- 8k smokeとの12 overlapping trajectories、**69,929 tokens完全一致**。すでにEOSだった9 answersの終了も不変。Smokeはpilot accuracyに混ぜていない。
- 10問・25問operational snapshotsのraw SHAは最終時点でも不変。途中の検定、sample-size/parameter adaptation、長文/不正解/cap answerの除外なし。実行時間帯のkernel journal照会ではNVRM/Xidに一致するentryなし。

## Accuracyと生成長

各condition100 answersだが、独立な解析単位は**50 problem clusters**。

| Condition | Legacy final / paper / EOS-required | Cap | EOS-wrong (legacy) | Mean generated tokens | Eviction exposure |
|---|---:|---:|---:|---:|---:|
|native|98 /98 /98%|0%|2|4779|0%|
|Random1024|91 /91 /91%|8%|1|6066|100%|
|Random2048|93 /93 /93%|6%|1|5557|77%|

圧縮条件の14 capped answersは全てlegacy-incorrect。EOS-wrongの計4 recordsはこの報告では追加adjudicationしていないため、真のmodel errorと表記/解釈問題を同一視しない。Primary graderとraw outcomesは保持し、Qwen3.5で行ったreview correctionを自動転用していない。

固定したproblem-cluster bootstrap（各problemの2 runsを保持、10,000 draws、seed20260919）の結果:

| Contrast | Difference | Descriptive95% percentile CI | Problem wins / losses / ties |
|---|---:|---:|---:|
|Random1024−native|−7.0pp|[−13.0,−2.0]pp|0 /6 /44|
|Random2048−native|−5.0pp|[−11.0,0.0]pp|0 /3 /47|

これらは事前に**探索的・非multiplicity-adjusted**と定めたintervalであり、confirmatory significance claimや≤2pp preservation判定に使わない。C2048のintervalが0を含んでも同等性は示さない。新しい独立dataset上の再現でもない。

## Memory・時間の解釈

Shared peak allocatedはnative **14.203GiB**、Random1024 **7.867GiB**、Random2048 **8.141GiB**。Qwen3のFA KVは144KiB/token/sequence、DeltaNet stateは存在しない。

Useful aggregate tok/sは151.7/151.9/156.6、useful-row fractionは89.7/83.2/85.8%。これは**異なる生成長・終了時刻を持つ実験全体の記述**で、固定workload speedupやserving throughputの証拠ではない。`elapsed_seconds=batch_wall/2`はaccounting、peakはbatch shared。C2048の23%はevictionなしで終了しており、32768/Cを典型的圧縮率と呼ばない。

## 同じ50問に限定したQwen3.5とのpost-hoc記述比較

Qwen3の50問pilotとQwen3.5の全500問mainを直接比較すると問題構成まで異なるため、別scriptで**同じ50 IDs・2runs・B2・32768 cap**のlegacy scoresを抽出した。Qwen3.5のreviewed scoresとは混ぜない。

| Model, same50 questions | Native | Random1024 | Random2048 | Native mean tokens |
|---|---:|---:|---:|---:|
|Qwen3-4B|98%|91% (−7pp)|93% (−5pp)|4779|
|Qwen3.5-4B|97%|77% (−20pp)|91% (−6pp)|7751|

**このmatched subsetではC2048の点推定のlossは−5pp対−6ppで近い**。Qwen3.5全500問での−12ppと今回の−5ppだけを並べ、architecture差と解釈してはいけない。C1024の点推定差も、training/tokenizer/36対8 FA層/head配置/output gate/生成長や、Qwen3 .6/presence0対Qwen3.5 1/presence1.5というsampling差に交絡する。Cross-model hypothesis testやDeltaNet causal claimは行っていない。

## 数値gateの透明性

初回BF16 eager参照gateは**不合格のまま保存**。FP32診断はBF16 eagerの中間丸めを含む参照差という説明を支持し、native/noop/preevictionは両dtypeで完全一致、full-FP32 post-eviction参照最大絶対差6.1035e−5だった。

診断後、same-input FP32 eager oracleに対する各row/head（projection前）・各row（後）のrelative L2/Linf≤1%を新gateとして明示し、未使用C1024/C2048 traceで検証してからMATHを開始した。初回の失敗を消したり、公式forwardを都合よく変更したりしていない。詳細は`qwen3-control-numerical-investigation.md`。

## Artifacts・次段階

- `qwen3-control-pilot-analysis.json`: 固定解析。
- `qwen3-control-pilot-integrity.json`: snapshot不変性・完全性。
- `qwen3-control-8k-32k-prefix-audit.json`: output-cap extension一致。
- `qwen3-qwen35-matched-pilot-descriptive.json`: 同じ質問に限定したpost-hoc比較、両subset SHA。再生成: `scripts/qwen3_control/compare_matched_pilot.py`。
- Pilot data SHA: `5cba2925451fe7d1a8072bcf0ab1ec24695b267e8368d37cc66d0dd714a029ff`。
- Manifest SHA: `b6f4de7349587205b14ad04980572567f3b9fec314fc69aa6d26c996a2aaa0aa`。

Qwen3全500問の確認実験は**まだ開始していない**。進める場合は未実行450問をprospective primaryとし、既知pilot50問を含む500問集約は別に位置づける設計が考えられるが、新protocolが必要。今回のpilotだけで近似同等性を宣言せず、Qwen3.5 main/C4096/C8192 frontierの既存結論も変更しない。
