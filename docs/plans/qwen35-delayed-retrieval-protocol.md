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
- 元factのKVが消えても、DeltaNet状態や後続tokenのKVへ情報が転写されている可能性がある。保持と正答の関連だけでDeltaNetへの因果帰属はしない。

## 対照と限界

- 短い距離・eviction前のnative/no-op一致を確認し、teacher-force adapterのバグと忘却を区別する。
- Random-shared比較でhead選択多様性の寄与を調べるが、Qwen3との比較なしにhybrid固有とは結論しない。
- 生成領域への注入はモデル自身の自然な推論生成ではない。このprobeだけからMATHでのcap増加の原因を断定しない。
- 事前分布に対する4択chanceは25%。独立単位はnonce問題であり、head数や同一問題のseed数を独立sample数として数えない。
- 推定計算量は約1000万teacher-forced token slots、B2で概ね1日前後。校正で実測後に更新する。

## 校正v1の固定 (2026-09-16、GPU実行前)

- 別script `scripts/qwen35/run_retrieval_probe.py`。MATH主評価の生成ソースは変更しない。
- `calibration-000`〜`007`の8問。nonce/data seedはcase IDから固定。確認用は別の`confirm-*` IDで、校正に使わない。
- 各caseでA/B/C/Dの4候補valueを生成し、正解labelをindex mod4で均等割付け。値はcolor-wordの任意の記号として使う。fillerはkey/valueを含まない別の簡単な算術文。
- native、gap=0/8192/16384、B2で同一traceの2row。nativeでは2rowは同じ出力になることを確認し、独立標本数は各gap8問として数える。
- **校正gate: 各gapでnative正答が7/8以上**。未達なら確認gridを開始せず、課題設計を調査して新versionへ進む。確認用結果を見て調整しない。
- 校正の前に1問gap0の全6条件を実行し、eviction前の4 label logitsがnativeと完全一致することを検証する。
- cacheのfact/value token保持数とolder-generated head間Jaccardを最終query token直前に記録。label probability massも保存し、条件付き4択確率だけを全vocabulary上のconfidenceと混同しない。
- 確認gridの候補（32問×2seeds×6条件×5gap）は校正通過後に最終固定する。teacher-forced workloadの速度を自由生成のserving速度とは呼ばない。
