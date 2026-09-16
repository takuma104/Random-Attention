# 固定decode workload / allocator対照の効率測定 v1

2026-09-17、測定前に固定。MATH生成長の差を速度差とみなさないための別実験。

## Phase 1: 同じbatch・同じtoken列・同じ論理context

- BF16、同じ4B checkpoint、FLA＋公式torch conv、SDPA、1 RTX5090。
- B=1/2/8。同じpromptとteacher-forced token列を各rowに複製し、全条件へ同一入力を与える。これは多様なリクエストのserving QPSではない。
- 既存probeのcalibration-000用data helperから、任意factと中立な算術文を固定生成。query/回答やsamplingは使わない。KV/RNN状態を人工的に注入せず、実modelの全forwardで履歴を作る。
- 論理cache長8192/16384/32768から測定開始。各pointで128 steps×2 modes×3反復の連続windows。反復内の順序はthroughput→synchronized。
- 比較対象の各windowは同じ開始/終了logical length、同じ次token列。反復は同一履歴上の隣接区間であり独立した問題ではない。
- 各windowはr=64の2 eviction周期を含む。cold load/JIT/prefill/履歴作成はdecode時間から除外。

### 条件

1. native_dynamic: 公式DynamicCache、evictionなし。
2. native_preallocated: 既存adapterを大容量C=最終logical lengthで使い、evictionが一度も発生しない対照。
3. Random C1024
4. Random C2048
5. Recency C1024
6. SnapKV C1024

preallocated nativeは最大長33536+64 slotsを初めから確保する。したがって短いcontext時の確保量はDynamicCacheより大きい。viewのlive KVとbacking storageを区別し、32k時のメモリ比較と各contextでの時間比較を混同しない。

### 測定mode

- **throughput:** windowの前後でCUDA synchronize。model＋LM headの処理速度。GPU上に用意した次tokenを使い、sampling/CPU EOS判定は含めない。
- **synchronized:** 上記に加え各forward後にもCUDA synchronize。モデル1 stepのbarrier込み時間。こちらもsamplingを含む実サービスのレイテンシとは異なる。
- 3連続windowsのmedian/min/maxを記録。少数かつ非独立なので、有意差やserving p99を主張しない。
- methodの実行順は上の順序で固定。各cellは最大contextまで十分にwarmし、GPU温度/clock/powerを記録するが、完全な熱制御やランダム化試行とは称さない。

### 正当性・メモリ

- 対応する各window終端でnative_dynamicとnative_preallocatedの**全vocabulary logitsの完全一致**を検証。
- 圧縮条件もeviction前なら同じ照合を行う。入力hash、logical長、eviction回数、全window数を監査。
- peak allocated/reserved、KV backing bytes、DeltaNet recurrent/conv bytes、prefill/履歴を含むrun全体のpeakも保存。
- 速度の主要な対照はnative_preallocated対圧縮。native_dynamicとの差も示す。preallocated対照は絶対positionの記録などadapter共通のbookkeepingも含むため、Dynamicとの差を純粋にallocator単独の効果とは呼ばない。
- 小さなcontextの全条件・B1/2/8 smokeを先に実行。モデル精度ではなく実装・数値・計数のgate。
- 本matrixの概算は2〜3 GPU時間。生成ソースは主評価/probeと分離。

## Phase 2候補: decode memoryを揃えたbatch

Phase 1のnative B8 peakを参考に、圧縮のbatchを増やしたdecode-memory matched点を追加する案。batchの選定はメモリ制約に基づき、精度で選ばない。Phase 1の実測を見て手順・許容差を固定してから実行する。

これは同じ精度を保証する比較ではない。主評価でC1024/C2048の精度低下が確認されているので、速度・メモリと必ず並記する。最大serving同時接続数やprefill込みの完全iso-memoryを未測定のまま主張しない。

## Phase 2の固定 (2026-09-17、メモリ校正前)

- 対照: Phase 1のnative_preallocated B8、context32768、decode-window peak allocated **17.15313148498535 GiB**。
- 対象: Random1024、Random2048、Recency1024、SnapKV1024。batchは8の倍数。
- 同じtoken列のprefix、context4096、128steps×2modes×1反復で先にメモリを校正する。両容量ともevictionが作動し、KV/query/recurrentのshapeはboundedな定常状態となる。
- 初期候補はPhase 1のB1/B8 peakの線形外挿から、Random1024 B96、Random2048 B64、Recency1024 B96、SnapKV1024 B72。
- 校正の最大decode allocatedが対照の**95〜100%**になることを要求する。予算超過ならBを8減らし、95%未満なら8増やして再測定。適合する8刻みがない場合は予算以下の近い点を選び、非一致率を明示する。測定速度や精度でbatchを選ばない。
- 選定後はcontext32768、128steps×2modes×3反復で本測定。全ての履歴を実modelでteacher-forceする。32kで再度予算を監査し、外れた点をiso-memoryと表示しない。
- native対照とは同じ入力hash、window位置、samplingなしの条件を照合する。異なるbatchのaggregate throughput比とms/stepの両方を示す。
- **揃えるのはPyTorchのdecode peak allocatedだけ**。reserved/NVML使用量、prefillを含むrun全体のpeak、実サービス同時接続数は同じとは限らず、別途表示する。
- 高batchのモデル精度は未評価。主評価B2での精度低下を必ず併記する。成功しても同等精度の高速化・最適な最大batchとは主張しない。

### 校正結果と本測定batchの固定（32k測定前）

| 条件 | 選定B | 4k decode peak GiB | 予算比 | 95〜100%内 |
|---|---:|---:|---:|---|
| Random1024 | 96 | 16.7500 | 97.650% | yes |
| Random2048 | 64 | 16.2590 | 94.787% | **no** |
| Recency1024 | 96 | 16.7496 | 97.648% | yes |
| SnapKV1024 | 72 | 16.4452 | 95.873% | yes |

Random2048 B64は95%未満だったため、固定ルール通りB72も測定したが17.2495 GiB（100.562%）で予算超過。8刻みで区間に入らず、B64をunder-budget fallbackとして使う。**許容差を事後に緩めず、Random2048を厳密な許容区間内のmemory-matched点とは呼ばない。**

全5校正cells（各2 windows）の監査通過。CPU選定コードは速度を参照せず、メモリとbatchのみで選定する。証拠hash・全試行・選定理由は`qwen35-efficiency-memory-selection-v1.json`に記録。本測定でも実測メモリを再監査する。
