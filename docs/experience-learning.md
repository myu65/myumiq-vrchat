# 実機経験からの運動候補学習

実機のtracker観測を開始姿勢として、既存の全身actorをPAMIQの学習専用モデルで更新する。
VRChat avatarの接触やワールド移動量を真値として学習する方式ではない。
予測は制約付きの運動モデルを使い、観測trackerと推定骨格の差を維持する。

必要な入力は、互換なactorと学習状態、別セッションの確定済みreplay二つ、
同じrigで作成したライセンス確認済みの参照評価corpusである。
モデル・replay・出力はリポジトリの外へ置く。認証情報は不要。

```powershell
python -m myumiq_vrchat.experience_training `
  --prior ../artifacts/baseline `
  --train-replay ../artifacts/session-a/experience.jsonl `
  --heldout-replay ../artifacts/session-b/experience.jsonl `
  --reference-corpus ../artifacts/licensed-corpus `
  --output ../artifacts/candidate-001 `
  --updates 50 --learning-rate 0.00001
```

必要な追加依存は`learning`。既存のSAC policy構造を読み込むが、確率的なSAC更新は行わない。
PAMIQの`TorchTrainer`、`DataUsersDict`、`SequentialBuffer`、推論との同期を持たない
`TorchTrainingModel`を使う。更新回数とCPU使用を制限し、候補重みとoptimizerを保存する。
途中で中断したジョブは不採用のまま残し、次回は元の入力から新しいジョブとして実行する。

実行管理から自動で候補を作る場合は、既存の自律設定へ次を追加する。

```json
{
  "purpose": {
    "learning": {
      "replay_refinement": {
        "prior": "../artifacts/baseline",
        "train_replay": "../artifacts/session-a/experience.jsonl",
        "validation_replay": "../artifacts/session-b/experience.jsonl",
        "reference_corpus": "../artifacts/licensed-corpus",
        "updates": 50,
        "learning_rate": 0.00001,
        "floor_weight": 50.0,
        "backtracking_steps": 2,
        "timeout_s": 300
      }
    }
  }
}
```

これは設定の抜粋であり、既存の`purpose.state`等は引き続き必要となる。
ジョブは既存実行管理の学習枠を使い、会話・身体処理を待たせない。
同じ設定の完了済みジョブは再実行せず、新しい経験集合を指定すると次の候補を作る。
終了時にはこのジョブの子プロセスだけを停止する。

`experience-manifest.json`は入力hash・session・使用したreadback根拠を記録する。
`result.json`は更新前後の到達、床制約、誤差と既存参照の比較を記録する。
床違反の増加、到達数の減少、既存参照の退行は不採用となる。
静止姿勢の参照評価は実行時と同じ到達保持条件（既存誤差内の3観測・150ms以上）を使い、
到達後は次の目標まで保持する。到達数の減少や、この軌道で床を下回る候補は不採用とする。
達成後もactorを動かし続ける従来の評価は`continuous_actor_reference_*`に別途残す。
この連続評価の床越えが保持によって修正されたと主張しない。保持を行わない周期動作や
ジェスチャーは、別に全区間の動作評価が必要となる。
`floor_weight`（CLIでは`--floor-weight`）で学習の床ペナルティを調整できる（0超〜1000）。
これは学習目的だけの重みであり、実行時の床制限を緩める設定ではない。
`reference_rehearsal: true`（CLIでは`--reference-rehearsal`）は、ライセンス確認済みcorpusの
train splitを各学習batchの約1/4に混ぜ、既存の身体形状を練習し直す設定。
床近くへ全身を平行移動した合成開始姿勢も含む。別のPAMIQ bufferで扱い、実機経験数へ加算しない。
heldout splitは評価専用のまま保つ。既定では無効。
`backtracking_steps`（CLIでは`--backtracking-steps`、既定0、最大3）は、
更新後の候補が不合格なら、元のactorからの重み差分を1/2、1/4、1/8へ順に縮める。
同じ到達・床・既存技能の評価条件を満たす最初の候補で止める。条件は緩めない。
全候補が不合格なら元の提案を不採用として保存する。縮小候補を採用した場合は、
元の大きな更新の勢いを引き継がないようoptimizer状態を初期化して保存する。
試した倍率と全評価結果を記録し、評価専用sessionは候補選択に使用したものとして扱う。
各試行の詳細姿勢は`backtracking-XX.json`へ保存し、主報告には判定指標と参照先を置く。
同じ詳細姿勢を繰り返し詰め込み、実行管理へ返す報告の容量制限を超えないようにする。
調整に繰り返し使用したsessionはvalidationであり、最終試験には新しいsessionを使う。

結果は`candidate_ready`または`candidate_rejected`として記録する。
`candidate_ready`は限定した実機試験の候補を意味し、live actorを自動置換しない。
参照clipの追加、候補重みの更新、実機の技能成功、avatarの自然さは別の根拠として扱う。
