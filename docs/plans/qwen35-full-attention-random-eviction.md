# Qwen3.5 Full Attention-only Random Attention追試

作成: 2026-09-09。状態: 開始承認済み。実行ログは `docs/plans/qwen35-progress.md`。

## 目的・仮説

RTX 5090 32 GiB ×1で、Qwen3.5のGated DeltaNetは通常通り動かし、Full Attention層のdecode時KVだけを物理的にevictionする。学習・重み量子化・画像入力・MTP/speculative decodingは初期実験の対象外。

- H1: DeltaNetの状態が履歴情報を補い、Full AttentionのKV圧縮に耐える。
- H2: 少数のFull Attentionが正確な長期参照を担うため、逆に圧縮に弱い。
- RandomとRecency・スコア型の差、非圧縮からの精度差、実メモリ・速度を別々に検証する。
- Qwen3との比較はモデル間比較であり、学習や構造の差があるためDeltaNetの因果効果とは断定しない。

## モデル・環境

主対象 `Qwen/Qwen3.5-4B` BF16。32層中24層がDeltaNet、8層がFull Attention（0始まり3,7,...,31）。Full Attentionは16 query heads、4 KV heads、head_dim=256、rotary_dim=64、出力ゲートあり。モデル・データrevision、ライブラリ、GPU環境を保存する。

- 専用Python 3.12環境を作り、Blackwell対応PyTorchとQwen3.5対応Transformersを固定。
- 元のtorch 2.4/cu121前提のsetup.shは使用しない。
- 正当性のためまず公式forwardとSDPAを活用し、DeltaNetの高速kernelも確認。
- 主評価は4B。Qwen3-4Bの対照確認、9B BF16の代表セルは後続。
- 重み以外のキャッシュ: Full Attention KVは32 KiB/token/系列、32kで1 GiB。DeltaNet再帰状態はFP32なら約48 MiB/系列＋conv等。

## 予算と方法

論文のK表記とコードのtoken_budgetにはbufferを含むかの違いがある。ここではCを圧縮直後の総保持tokens/KV headとする。

- recent buffer r=64。
- prompt長Pはchat templateとgeneration openerを含むprefill全体。
- 保護型ではC-r個の候補保持枠のうちP個をprompt、C-r-P個を生成履歴に使う。
- C=1024,2048,4096を初期sweep、512は後続。
- C+rまで伸ばし、attention計算後にCへ戻す。候補は直近rを除く。以後64 decode tokensごとにeviction。
- promptが収まらない(P>=C-r)場合は黙って保護範囲を切らず、予算不適合として記録。全手法共通の対象集合で比較する。
- 選択は層・KV head・系列ごとに独立。保持indexは時系列順にsortし、post-RoPE KとVをgatherする。破棄KVは復活させない。
- 論理位置は元の累積token位置。物理cache長とは分離する。
- 各方法ともDeltaNetを直接圧縮・resetしない。ただし上流evictionの影響が通常forwardを通じてDeltaNetに伝わることは許容する。

方法:
1. native: 公式ハイブリッドモデル、evictionなし。
2. random_pp: prompt保護＋独立random。
3. recency_pp: prompt保護＋直近履歴。
4. snapkv_pp: prompt保護＋直近queryベースのスコア（実装定義を明記）。
5. random: prompt保護なし（ablation）。
6. random_shared_pp: 各層でhead間の選択を共有（ablation）。

TriAttentionはpartial RoPE対応と再校正が必要なため後続。SnapKVのみとの比較で最強selectorとの同等性は主張しない。

## Phase 0: 正当性テスト

- 公式nativeとadapter eviction無効のlogits・greedy出力一致（dtype/kernel誤差を考慮）。
- eviction発動前一致、eviction後に同一保持KVを使う参照attentionとの一致。
- prompt/recent保護、保持数、絶対位置、head独立性、再現性、物理メモリ上限。
- eviction操作直前/直後のDeltaNet state不変。同一系列の過去状態を保存し続けない。
- prefillと逐次decodeの整合、長い論理位置でのmaskサイズ確認。
- 最初batch=1、次に同一promptの複数反復。終了系列を除くとき全stateを同じindexで選ぶ。
- 生成用とeviction用のRNGを分離する。

## Phase 1: パイロット

MATH500の50問を固定seedで抽出しIDを保存。各1回。nativeとrandom_pp/recency_pp × C={1024,2048,4096}の7セル。

- 最初8k max_new_tokensで動作・生成長・速度・VRAMを測る。
- 一部32kで確認。8kの結果のみで長い推論への結論は出さない。
- 全例について正答、生成長、終了理由、eviction回数、発動率、peak memory、実行時間を記録。
- native長を用いた層別解析も行い、圧縮後の長さだけで条件付けしない。
- パイロットは探索用として明示し、予算変更等を結果とともに記録。確認用は別seed/集合を明示（MATH500全体集計時にはpilotを含むことを開示）。

## Phase 2: 主評価

最初のMATH500は500問×2反復×5セル:
- native
- random_pp C=1024,2048
- recency_pp C=1024
- snapkv_pp C=1024

代表予算が不適切ならpilot後・本評価前に変更理由と最終条件を固定。max_new_tokens=32768。失敗・上限到達を除外せず分母に残す。

追加:
- GPQA-Diamond 198問×2回（必要なら4回、データ利用許諾が前提）
- AIME2025+2026 60問×4回（探索的。小差判定には不足）
- Qwen3-4B対照、9B BF16代表セル
- LiveCodeBenchは長prompt対応とsandbox採点整備後

## Phase 3: 機構分析

短いprefillの後、decode経路で履歴をteacher-forceし、ランダムな変数・値を途中に一度だけ挿入。1k/4k/8k離したqueryでexact matchと正答logprobを測る。再提示あり/なし、独立random/head共有random/recency/nativeを比較する。

needleを保護promptに入れない。値をKVから消してもDeltaNetや後続tokenに情報が残り得るので、全KV削除を情報の完全消去と解釈しない。

## 生成・統計

Qwen3.5公式template、enable_thinking=True。モデルカードの一般Thinking設定を初期採用: temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1.0。適用実装を検証し各runに保存。

- max_length（入力込み）ではなくmax_new_tokensを使う。
- 問題ID・run IDに基づくseedを使いresumeや実行順に依存しない。
- 生成RNGとeviction RNGを分離。
- 正答率、nativeとの差、終了率、cap率、長さ、eviction率を報告。
- 問題単位で対応付きcluster bootstrap 95% CI。非有意を同等とみなさない。
- 実用上の目標としてnativeからの低下2 percentage points以内を事前設定するが、CIが支持しなければ未確定とする。
- graderのboxed抽出/数式判定を人工例と手動spot checkで監査。thinking内部と最終回答を区別する。

## Phase 4: 効率

正当性と精度を確認後、デバッグlog無効、単一GPU job、同一kernel経路で測る。

1. 同じbatch=1/2/4、固定promptと8k/32k出力でdecode tok/s、TTFT、全時間、eviction時間、peak allocated/reserved memory。
2. 同じ安全VRAM上限で各手法の最大batchと集約tok/s。精度を維持できる予算を使う。

重み、KV、DeltaNet、index/一時領域を区別。KV削減率と総VRAM削減率を混同しない。精度runの長さ変化を固定workload速度と混ぜない。vLLMは初期対象外。異なるallocator/未最適化selectorの比較から一般的な速度優位を主張しない。

## 計算量・自律実行

MATH500の5セルは5000回答、平均5k–10kなら25M–50M tokens。仮に100 aggregate tok/sなら生成だけで69–139時間（実測予測ではない）。pilotで実測し見積もりを更新する。

ユーザー承認により、環境構築・モデル/公開データ取得・実装・テスト・pilot・主評価を自律的に進める。既存環境/他processを壊さず、GPU生成jobは原則1本。OOMはbatch縮小、数値不一致は本評価を停止して修正。gatedデータ許諾・解消不能な互換性問題などがあればユーザーへ相談。結果と設定のmanifestを保存し完了例を再生成しない。

## 参照

- `docs/paper/2609.03430.md` (§2–6, Appendix D/F/G)
- `kvcompress/engine/cache_utils.py`, `modify_forward.py`
- `kvcompress/harness/generation_utils.py`, `eval_hf.py`
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5
