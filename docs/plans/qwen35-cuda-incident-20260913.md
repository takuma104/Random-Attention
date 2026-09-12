# CUDA停止と再開記録 (2026-09-13)

## 停止

- 00:11:48 JST、主評価job `qwen35-main32k-b2-34f9`がCUDA `unspecified launch failure`で停止。実行開始から約61時間51分。
- 実行中: 224番目の問題 `test/number_theory/183.json`、SnapKV C1024、B2、run0/1。最後のheartbeatは17920 decode steps、両row active。
- tracebackの報告位置は公式RoPE演算。ただしCUDAエラーは非同期報告なので、ここが原因とは断定できない。
- 同時刻にkernel logでXid 13: `SKEDCHECK22_INVALIDATE_ACTIVE_QMD failed`。PythonのPIDと一致。実行開始以降のkernel logで他時刻のNVRM/Xidは見つからなかった。
- 根本原因は未確定。OOM、特定のeviction実装バグ、一時的なdriver/GPU問題のいずれかと断定しない。

## データと復旧前検証

- 223問×10回答が完了。224番目はnative/Random1024/Random2048/Recency1024の8回答が保存済み。合計2238回答。
- 失敗したSnapKVの2回答は未保存。recovery journalもなし。失敗batchは評価対象から除外せず、同じseedで再実行する。
- 保存済み2238回答のSHA256を`work/qwen35/pre-resume-20260913-hashes.json`へ保存。再開後に不変性を照合する。
- 完全な223問の派生snapshot監査は通過。EOS/counters/paired fields/batch accounting/frozen source hashに異常なし。`qwen35-main32k-progress-223.json`。
- 新しいprocessで既存GPUテスト15件が通過。停止後nvidia-smiは正常応答、GPU44℃、621MiB（desktopのみ）。reset/reboot/driver変更は行わない。

## 再開方針

元のコマンド、生成/採点ソース、model、sampling、seed、B2、全セル、32k上限を維持する。完了JSONはskipし、失敗したbatchを最初から再実行する。

```bash
.venv/bin/python scripts/qwen35/run_math_batch.py \
  --out results/qwen35/main32k_b2_v3 \
  --subset all --limit 500 --runs 2 --max-new-tokens 32768
```

直後の問題 `test/precalculus/190.json` が始まる時点を監視し、失敗batch完了と旧回答の不変性を確認する。再発時は繰り返し自動再試行せず、同期CUDA診断などの独立した再現試験を行う。

GPU時間の単純外挿は223問で約5.70日へ更新。以前の5.00日は100問時点の暫定値であり、入力分布と生成長による変動がある。停止/診断時間は生成性能の集計に混ぜず、運用時間として別に記録する。

## 再開確認: 00:22:49 JST

- 00:14:35、job `qwen35-main32k-b2-resume1-d641`として同一コマンドで再開。
- 失敗していたSnapKV batchは同じseedで32768 stepsまで完了。両rowともcap、通常の不正解回答として保存。除外・別seedへの置換はしていない。
- 旧2238回答のSHA256は全て不変。`qwen35-cuda-recovery-20260913.json`。
- 完全な224問・2240回答の監査も通過。`qwen35-main32k-progress-224.json`。生成61.610時間、外挿5.73日、完了目安は9月16日未明（暫定）。
- 00:23頃の確認でGPU53℃、約254W、使用約10GiB。再開後のkernel logに新しいNVRM/Xidはなし。
- 同じbatchが完了したことは、特定入力で直ちに再現する障害ではなかったことを示すが、根本原因や過去の全数値計算の完全性を証明するものではない。引き続き監視し、再発時は独立診断へ進む。
