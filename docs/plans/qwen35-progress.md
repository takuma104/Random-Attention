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
- 次: 2問×3条件、最大生成2048 tokensのsmoke、その後50問×7条件、最大8192 tokensのpilot。短いsmokeの精度から結論は出さない。
