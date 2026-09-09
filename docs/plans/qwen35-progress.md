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
