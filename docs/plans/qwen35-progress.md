# Qwen3.5追試 進捗

## 2026-09-09 開始

- ユーザー承認: 計画保存と自律実行（詰まった場合に相談）。
- 計画: [qwen35-full-attention-random-eviction.md](qwen35-full-attention-random-eviction.md)
- GPU: RTX5090 32607 MiB、driver 610.57.04。開始時621 MiBはdesktop使用。
- Host RAM 61 GiB、disk空き2.8 TiB。
- 元の実装は変更せず専用実験系を追加する方針。
- Python 3.12 / torch 2.11.0+cu128 / transformers 5.16.1 / fla-core 0.5.2で専用環境構築完了。

## Phase 0: 実装・正当性

- `kvcompress/qwen35/`に実験系を分離。公式Qwen3.5 attention forwardを保持し、post-hookでFull Attention KVのみを物理compactionする。全8層対象。
- モデルrevision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- MATH500 revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。50問のIDをhash順で事前固定（`work/qwen35/artifacts.json`）。
- FLAはroot importだけでは不十分で、`fla.ops.gated_delta_rule`までTransformersより先にimportしないと、循環importによりtorch fallbackになることを確認。実行前に関数のprovenanceを検査してfail-fastする。
- DeltaNet chunk/recurrentはFLA使用。causal convは現段階では公式torch fallback。最適化速度の主張はまだしない。
- 初期実モデル検証のtokenizer戻り値不一致（Transformers5.16のBatchEncoding）を修正。
- 4B BF16全checkpoint読込成功、missing/unexpected keysなし、parameter bytes=9,078,531,072。
- 実モデル179 decode stepsの検証: no-op adapter/native logits最大差0.0、eviction発動前最大差0.0。
- eviction後432箇所のattentionを独立eager参照計算と比較。BF16のrtol=.03, atol=.03内で全通過（最大絶対差0.0625）。これは相対誤差を含む基準で、全て絶対差.03以下という意味ではない。
- 上記検証で32 layer-eviction events、約22秒、peak allocated=9,360,887,808 bytes。
- CPU/GPU cache不変条件、RNG分離、Transformersとtop-k/top-p分布一致、boxedとthinking境界の採点など10テスト通過。
- 採点は元リポジトリの数式graderを利用。CUDA processからforkしないよう、別CPU processへ分離。final-boxed metricとpaper互換の全completion採点を両方記録する。
- 各回答を原子的に保存し、IDベースで再開。生成済み未採点も再生成しない。設定・source hash不一致のresumeは禁止。
- 実装と検証記録はcommit `9e4bccc`としてorigin/experimentsへpush済み。

## 2026-09-10 00:13: smoke完了

- `results/qwen35/smoke_v1`: 事前固定pilotの先頭2問×native/random_pp C1024/recency_pp C1024、最大生成2048 tokens。
- 6回答すべて正常に保存・採点。budget不適合0、grading error0。
- 全回答が2048-token cap到達、thinking未終了。final/paper正答率0だが、短すぎる出力上限であり精度の結論は出さない。出力をspot checkし、数学的な推論が進んでいることを確認。
- 各圧縮回答で136–144 layer-eviction events（8層合計）。nativeでは0。
- nativeと圧縮の生成token列は初回eviction以前に一致し、相違はeviction後にのみ発生。
- native約75.6 tok/s、random約74.8、recency約74.9（batch1、この短いworkloadの診断値。速度優位は主張しない）。
- Full Attention KV実allocation: 圧縮35,651,584 bytes（C+r=1088）、native約71–72 MB。DeltaNet再帰状態50,331,648 bytes、conv1,572,864 bytesは同じ。
- 50問×7条件×8192-token capのpilotに進む。上限まで全回答が生成した場合、75 tok/s基準で約10.6 GPU時間。実際のEOSにより短縮する可能性あり。
- 次: pilotのEOS、cap率、eviction発動率と精度・長さを確認し、32k確認と主評価の条件・時間見積もりを決定する。

## 2026-09-10 00:28: pilot_v1停止・EOS修正

- 最初のEOS回答（recency C4096、8047 tokens）をspot checkし、正しい最終boxedとthinking閉鎖を確認。
- 同時にcheckpoint text config EOS=248044 (`<|endoftext|>`)とtokenizer EOS=248046 (`<|im_end|>`)の不一致を発見。checkpointにgeneration_config.jsonはなく、前者だけで終了判定していた。
- 回答は`<|im_end|>`の後に改行・`<|endoftext|>`まで生成していた。chatターン境界後の継続を避けるためpilot_v1を停止し、両EOSのunionで止めるよう修正。旧smoke/pilot_v1は実装診断のみで、確認評価には混ぜない。
- tokenizer EOSを含める回帰テストを追加（11 tests passed）。修正commit `337db8d`をpush。
- 00:32に実モデルEOS回帰テスト完了。既知のrecency C4096回答は8045 tokensで`<|im_end|>`終了し、thinking閉鎖・正しいfinal boxedを確認。旧8047-token列の先頭8045 tokensと完全一致。生成分布は変えず正しい位置で停止したことを確認。
- 新規`results/qwen35/pilot_v2`で50問×7条件、最大8192 tokensのpilotを開始。v1は混ぜず全例を新設定で再生成。

## 2026-09-10 07:09: pilot_v2完了

- 350回答完了、wall約6時間36分。全セルの欠損/重複/budget不適合/grading errorsなし。終了token、cache長、eviction回数、paired条件の監査通過。
- 詳細: [qwen35-pilot-v2-report.md](qwen35-pilot-v2-report.md)、機械可読統計 `qwen35-pilot-v2-analysis.json`。
- final正答率: native76%、Random/Recency C1024=64%/66%、C2048=70%/70%、C4096=74%/74%。
- ただし全条件でEOS終了例の誤答0。native24%、圧縮26–36%が8k capで未完了。差の解釈には32k確認が不可欠。
- Random vs Recencyの95% CIは広く、同等性は未確認。C4096でもnativeからの低下2ポイント以内という事前目標をCIは保証しない。
- 次の32k確認は事前固定pilot先頭10問、当初の5条件（native、Random1024/2048、Recency1024、SnapKV1024）、各1回。SnapKV実モデル検証通過後に実行する。結果に基づく問題選別はしない。
- MATH500全500問のprompt長を事前検査: 最大857 tokens、C1024のprompt保持枠960未満で全例適合。

## 2026-09-10 07:14: SnapKV検証・32k確認へ

- SnapKV＋promptの実モデル179 decode-stepテスト通過。no-op/nativeと発動前logits最大差0、eviction後432箇所の参照attention比較通過、32 layer-events。
- 記録: `qwen35-snapkv-validation.json`。モデルのゲート・partial RoPE・DeltaNetは公式実装を維持。
- このSnapKVはリポジトリのdecode adaptation（直近query平均、幅5平均pool、GQA group平均）を移植。スコア算出はFP32で、元実装のBF16演算と数値的に完全同一とは主張しない。
- `results/qwen35/confirm32k_v2`に先頭10問×5条件×1回、max_new_tokens32768を実行する。既存8kと同一seed・prompt・samplingで、出力上限のみを延長。SnapKVは追加比較。
- 50回答全て32kに到達した場合の約6時間は上限長に基づく概算。EOS終了すれば短縮される。完了後に本評価500問×2回×5条件の見積もりを更新する。

## 2026-09-10 09:46: 32k確認完了

- 50回答、2時間30分、監査通過。native10/10、Random1024 8/10、Random2048 9/10、Recency1024とSnapKV1024は7/10。
- ここでも全EOS回答は正解、残る不正解は32k cap。重なる8k実行との全40 prefixが完全一致。
- 詳細: [qwen35-confirm32k-report.md](qwen35-confirm32k-report.md)。条件は変更せず、32k本評価へ進む。
- B1で500×2×5へ単純外挿すると約10.4日。予定していた同一promptの反復B2を先に検証し計算時間を削減する。

## 2026-09-10 09:56: B2検証

- FP32参照でB1/B2 logits差2.86e-5、BF16ではbatch形状由来と整合的な丸め差を観測（最大.5625 logits、32step最大KL.001727）。B1/B2の自由生成スコアを混ぜず、本評価全手法をB2に統一する。
- 固定1024-output実測: native75.8→147.1 tok/s、Random75.8→145.4、SnapKV72.8→139.8。短い固定workloadで約1.9倍。長い反復の長さの違いにより実効倍率は下がり得る。
- B2実装、row別乱数、EOS後の片方のみ継続、部分採点からの原子的再開など15 tests passed。
- 次: `run_math_batch.py`で2問×3条件×2反復、8k smoke。実データ監査・再開テスト通過後、500問×2反復×5条件、32k本評価を開始する。

## 2026-09-10 10:15: B2 smoke完了、本評価固定

- `smoke_b2_v3`: 12回答、11分21秒、全監査通過。nativeの2回答がEOS（6828/5309 tokens）で両方正解、残り10回答は8k cap。2問だけの動作確認であり精度結論には使わない。
- 同一batchで先に終了したrowのcache長・レイテンシが適切に保存され、partnerは最後まで継続。
- 同一コマンドで再開したところ全12回答をskip。再監査のcompleted data SHA256は`b921baccc722eb35fa5386625094b3a2f5c808eb64dedbc8feb3cc90f49569fc`で変更なし。
- [本評価プロトコル](qwen35-main32k-protocol.md)を開始前に固定。500問×2反復×5条件、B2、32k。主比較・CI・2比較のHolm補正、途中精度で条件変更しない規則を記載。
- 所要時間は暫定5〜8日。最初の20問完了で運用・時間の確認を行う。生成ソースは以降変更しない。

## 2026-09-10 15:58: 主評価20/500問（200/5000回答）完了

- 主評価は10:20:30開始、job `qwen35-main32k-b2-34f9`。開始commit `fa2864a`、実装`cc919ad`。
- 最初の20問をCPU-onlyの派生snapshotで監査。元manifest/回答は変更せず、EOS・counter・batch・ペア対応・source hash全て通過。
- 200回答の生成は5.610時間。単純外挿は全体5.84日（約140時間）、完了目安は9月16日午前。最初の20問はデータセット順の複数分野で、無作為な追加標本ではないため暫定推定。
- EOS終了した161回答は全て正解、残り39回答が32k cap。途中の手法別精度から条件やsample数は変えず継続する。
- 実効aggregate速度123–132 tokens/s、useful row-step比86–93%。早期終了rowを残す損失を含む。共有peak allocatedはnative10.70 GiB、圧縮8.72–8.78 GiB。
- 運用確認時: GPU55℃、約257W、使用約10GiB、disk空き2.8TiB。異常なし。
- 記録: `qwen35-main32k-progress-020.json`。次の監視は100問完了時。
- 生成ソースには触れず、CPU-onlyの`audit_progress.py`と事前固定統計用`analyze_main.py`を追加。後者のHolm/strict margin/cluster単位は3 CPU tests passed。
- 次段階の[遅延参照probe案](qwen35-delayed-retrieval-protocol.md)を保存。factを保護promptではなく生成領域に置く点と、post-attention evictionによる測定時点のずれを明記。

## 2026-09-11 10:27: 主評価100/500問（1000/5000回答）完了

- 保存・EOS・cache counter・batch・ペア対応・frozen source hashの監査通過。生成ソースは開始時のまま。
- 生成24.017時間、全体の単純外挿は約120時間（5.00日）。現在の完了目安は9月15日午前へ更新。分野/難易度による時間変動は残る。
- GPU55℃、約254W、使用約10GiB、disk空き2.8TiB。異常なし。次の確認は250問完了時。
- 記録: `qwen35-main32k-progress-100.json`。

### 採点器の表記差によるfalse negativeを発見

- 1000回答のうち、163回答が32k cap、837回答がEOS。自動採点ではEOS不正解13例。
- 全13例を確認すると、**12例は数学的には正しい表記の違い**、1例は実際のモデル誤答（5乗根の恒等式で定数1を落としたRecency回答）。
- 内訳: `east`と`\text{east}`の差9例、問題の要求通りのcomma-separated rootsとgoldの`\pm`表記の差2例、八進数を明示する文脈でboxed内のbase suffixを省略した例1例。根の代入はSymPy、進数と回転は整数演算で独立確認。
- さらにcapでも最後の`</think>`後に正しいboxがある3例を確認。これは事前固定した主指標では正答だが、EOSを要求する副指標では不正解となる。
- 記録: `qwen35-main32k-grading-review-100.json`。採点例外・出力破損ではなく、継承したgraderの意味的な限界。生成に影響しないため実行は継続し、途中でgraderやraw scoreを書き換えない。
- 最終報告では「固定したlegacy graderのスコア」と、全条件に同じ規則で行う別versionの表記差に関する感度分析を区別する。途中で発見した補正を事前登録済みと称したり、都合のよい条件だけ修正したりしない。

## 2026-09-13 00:23: CUDA停止から再開、224/500問完了

- 00:11:48、224番目のSnapKV batchでCUDA launch failure、kernel Xid13を記録。原因は未確定。保存済み2238回答は維持され、実行中の2回答のみ未保存。
- fresh processでGPU tests15件、完全な223問の監査、frozen source hash照合が全て通過。reset/rebootや環境変更は行わなかった。
- 00:14:35、同じコマンド・設定・seedで再開（job `qwen35-main32k-b2-resume1-d641`）。00:22:49、失敗していたbatchの両rowが32kまで完了し、次の問題へ進んだ。
- 旧2238回答のSHA256が全て不変、224問2240回答の監査も通過。sourceや採点を変更せず、失敗batchを除外せずに継続する。
- 生成61.610時間からの暫定外挿は5.73日。完了目安は9月16日未明へ更新。これは入力/生成長の変動を含む暫定値で、停止・診断時間は生成時間とは別に扱う。
- 詳細: [CUDA incident](qwen35-cuda-incident-20260913.md)、`qwen35-cuda-recovery-20260913.json`、`qwen35-main32k-progress-223.json` / `224.json`。
- 再開後のGPUは53℃、約254W、使用約10GiB、新しいXidなし。再発時は再試行を繰り返さず独立診断。次の通常監視は250問完了時。

## 2026-09-13 09:02: 主評価250/500問（2500/5000回答）完了

- 中間点へ到達。CPU-onlyの完全prefix監査が通過し、frozen sourceも一致。主評価の条件・sample数・採点ソースは変更していない。
- 復旧から約8時間47分、kernel logに新しいNVRM/Xidなし。GPU52℃、約252W、使用約10GiB、disk空き2.8TiB。
- 生成70.193時間、全体の単純外挿は140.387時間（5.85日）。完了目安は9月16日午前。入力/生成長の変動と運用時間を含むため引き続き暫定値。
- 2500回答中565回答が32k cap。legacy graderのEOS不正解は18回答で、既知の表記差問題も含む。この途中結果で条件やsample数を変えず、仮説検定は完了後に行う。
- 記録: `qwen35-main32k-progress-250.json`。次は400問完了時、またはjob異常終了時に確認する。

## 2026-09-15 02:50: 主評価400/500問（4000/5000回答）完了

- 80%まで完了。CPU-onlyの保存・EOS・cache・batch・ペア対応監査が通過し、frozen source hashも開始時のまま。
- 再開から約50時間35分、新しいNVRM/Xidなし。GPU54℃、約254W、使用約10GiB、disk空き2.8TiB。
- 生成111.539時間、全体外挿139.424時間（5.81日）。残り約28生成時間、完了は暫定で9月16日午前の見込み。
- 4000回答中886回答が32k cap、legacy graderのEOS不正解は56例（既知の表記差による誤判定を含む）。途中の成績による条件変更や仮説検定は行わない。
- 記録: `qwen35-main32k-progress-400.json`。このまま全500問完了まで待ち、最終監査・事前固定統計・別途のgrading感度分析へ進む。異常終了時は直ちに診断する。

## 2026-09-16 09:25: 主評価5000回答完了

- 全500問×2反復×5条件を完了。生成141.719 GPU時間、全件監査通過、未処理journalなし。再開後に新しいXidなし。
- Legacy graderのfinal score: native93.3%、Random1024 71.4%、Random2048 81.3%、Recency1024 68.2%、SnapKV1024 68.4%。
- 主比較Random2048−nativeは−12.0pp、95% CI [−14.5,−9.6]pp。低下2pp以内という目標を達成しない。
- Random1024はRecency比+3.2pp / SnapKV比+3.0pp。対応付きCIはそれぞれ[1.6,4.8]/[1.5,4.6]pp、事前固定した2比較Holm補正sign testも通過。
- 詳細: [主評価報告](qwen35-main32k-report.md)、`qwen35-main32k-analysis.json`、`qwen35-main32k-progress-500.json`。
- この数値には既知のgrader表記差問題がある。元スコアを保ち、全条件に同じ規則を適用する別versionの感度分析を次に実施する。

## 2026-09-16: 採点感度分析v1完了

- 全EOS不正解等と固定hashの自動正答100例、合計177回答を122個のcondition-masked packetsへまとめ、判断を保存してからmappingを戻した。過去の100問で見た例の非blind性と、AI-assisted reviewである点を開示。
- 表記差のfalse negative56回答をsidecarで補正。自動正答sample100例にfalse positiveは見つからなかった。元の5000回答・主スコアは不変、aggregate SHAも一致。
- 補正後: native94.7%、Random1024 72.4%、Random2048 82.4%、Recency69.3%、SnapKV69.4%。Random2048−nativeは−12.3pp、95% CI [−14.8,−9.8]pp。C1024 selector差は+3.1pp/+3.0ppで、主結論は変わらない。
- 曖昧なinverse-cotangent branchとregion/volumeの5回答は無理に補正せずlegacy判定を維持。残るEOS不正解10回答は、この5例＋final boxなし3例＋実際の計算誤り2例。
- 詳細: [感度分析報告](qwen35-grading-sensitivity-report.md)、決定JSON、統計/sidecar JSON。CPU testsは統計と合わせ6件通過。
- 次は遅延参照probeを別実装・別データで校正し、KV-head選択多様性と古い生成情報の保持を調べる。

## 2026-09-16: 遅延参照probe v1の校正へ

- 別script `run_retrieval_probe.py`と独立データ生成を実装。protected prompt65 tokensの後、生成領域[65,85)へ任意のkey→value factをteacher-forceする。
- 1問gap0の全6条件で、eviction前の4択label logitsがnativeと完全一致（最大差0.0）。正解条件付き確率0.99293。記録`qwen35-retrieval-control-v1.json`。
- fact/value span、retentionの測定位置、head共有選択とrow別RNGをCPU testで確認。元MATH生成ソースは変更していない。
- 校正gateを実行前に固定: calibration用8問、native、gap0/8192/16384でそれぞれ7/8以上の正答。B2の同一trace2rowを独立問題として数えない。
- 次に長距離native校正を実行し、通過した場合のみ別nonceの確認gridを最終固定する。

## 2026-09-16 10:30: 遅延参照native校正通過

- native、8校正case×3gapの24 batches、43分44秒。gap0/8192/16384の全てで8/8正答、事前gateを通過。
- 平均正解条件付き確率0.9946/0.9878/0.9874、label massは約0.999。重複rowを独立標本として数えず8case単位で判定。
- strict CPU auditorを追加し、trace hash、生成領域のfact span、最終query直前のsnapshot/counter、全head保持、paired inputsを監査。off-by-one/改変traceの拒否を2 CPU testsで確認。
- 記録: `qwen35-retrieval-calibration-v1-analysis.json`、`qwen35-retrieval-control-v1-analysis.json`。
- 確認gridは32case×2seeds×6条件×5gapで固定（約20時間）。主比較は遠距離3gapをcase内平均したRandom1024−Random-shared1024。
- まず固定した2校正case、gap2048、全6条件のpost-eviction実装smokeを行う。圧縮の精度を見てgridを変えない。

## 2026-09-16 10:41: Post-eviction smoke通過、確認実験へ

- 固定した2校正case×6条件、gap2048の12 batchesが5分32秒で完了。trace/counter/retention/paired inputsの監査通過。
- Older-generated KV位置のhead間JaccardはRandom1024で0.361、Random-sharedで1.000。Recencyでは元fact tokenが全て消え、nativeでは全保持。設計した選択・測定の挙動を確認した。
- 2caseだけの精度から優劣は判断せず、事前固定した確認gridを変更しない。
- 記録: `qwen35-retrieval-posteviction-smoke-v1-analysis.json`。
- 次: `retrieval_confirm_v1`、別nonce32case×2seeds×6条件×5gap、960 batches。約20時間を見込み、生成ソースを固定して実行する。
