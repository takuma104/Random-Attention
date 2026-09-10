# 遅延参照probe: 準備案（主評価と独立、2026-09-10）

GPU実行は主評価完了後。主評価の生成ソースは変更しない。

## 問うこと

Full Attentionを8層だけ持つhybridで、古い生成情報の保持と読み出しはどう変わるか。
RandomのKV-head別選択の多様性が助けになるか。Math正答率と同じタスクではなく、機構を絞ったteacher-forced probeとして扱う。

**重要:** `*_pp`はprompt全体を保護する。古いfactをpromptに置くだけでは、evictionによる忘却の検査にならない。factと遅延部分は必ず初期prefillの後、**生成領域へ1 tokenずつteacher-force**する。

## Trace

1. 共通の短い初期指示だけをprefill（promptとして保護）。
2. 任意のnonce key→value対応を、assistantの作業メモ形式で生成領域に注入。
3. target key/valueを含まない固定seedの中立な文章・簡単な計算を挿入。
4. keyのvalueを選ぶ四択と回答prefixを生成領域に注入。正解ラベルは均等に割り付ける。
5. A/B/C/Dのnext-token logitsから条件付き正答確率、margin、argmax正答を測定。自由生成の長さ・EOSを介在させない。

factのtoken span、queryの長さ、fact終端からquery開始までの距離を実token単位で保存。選択肢は同じvalue集合の順列で、ラベル/文字列priorから正解できないようにする。ラベルが単一tokenであることをtokenizerで検証する。

## 候補grid

- native、Random C1024、Random-shared C1024、Recency C1024、SnapKV C1024、Random C2048。
- Random-sharedは既存の`random_shared_pp`: 同じlayer内でKV headsが同じ選択を共有する。layer間とrow間のRNGは依然独立。
- 距離: 0/512/2048/8192/16384 tokens。
- 別nonceの8問で課題理解を校正。その後、固定した32問×2 eviction seedsで確認する案。
- samplingなし。初期promptと全teacher-forced token列を手法間で同一にする。B2では同じpromptと同じtraceの2 seedを組にできる。
- nativeの校正成績が低い場合は課題設計を見直して別versionとし、確認用nonceを見て修正しない。gridとnative校正基準の最終固定はGPU校正前に行う。

## Retentionの測定時点

- layer/head別にfact spanの保持数、どれか1 tokenの保持、value span全体の保持、head間の選択重複率を記録する。
- adapterはattention計算**後**にevictする。最後のquery tokenを処理した後のpositionsだけを見ると、そのlogitに実際に使われたKVとずれる可能性がある。
- 最後のquery tokenのforward直前にresident positionsを保存する。新規tokenはfactではないため、これが最終logit計算に利用可能だったfactの保持状況となる。
- KVからfactが消えてもDeltaNet状態には情報が残り得る。保持と正答の関連だけでDeltaNetへの因果帰属はしない。

## 対照と限界

- 短い距離・eviction前のnative/no-op一致を確認し、teacher-force adapterのバグと忘却を区別する。
- Random-shared比較でhead選択多様性の寄与を調べるが、Qwen3との比較なしにhybrid固有とは結論しない。
- 生成領域への注入はモデル自身の自然な推論生成ではない。このprobeだけからMATHでのcap増加の原因を断定しない。
- 事前分布に対する4択chanceは25%。独立単位はnonce問題であり、head数や同一問題のseed数を独立sample数として数えない。
- 推定計算量は約1000万teacher-forced token slots、B2で概ね1日前後。校正で実測後に更新する。
