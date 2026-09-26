# VRChat音声セットアップ

実装中の低遅延経路と検証手順: [Realtime voice](realtime-voice.md)。

## 認識と言語の設定

`asr_adapter`でモデルと実行環境を選ぶ。Qwen3-ASRをローカルの音声対応
chat-completionsサーバーで動かす例:

```json
{
  "name": "local_audio_chat",
  "options": {
    "base_url": "http://127.0.0.1:18531/v1",
    "model": "qwen3-asr-0.6b",
    "text_format": "qwen3_asr",
    "qwen3_language": "Japanese",
    "timeout_s": 3.0,
    "warmup_timeout_s": 60.0
  }
}
```

`qwen3_language`を省略すると自動言語判定。指定する場合は最終assistantメッセージの
prefillに対応するサーバーが必要（llama.cppでは`--prefill-assistant`）。
`context`に認識用の短い文脈を最大2048文字まで指定できるが、会話履歴は自動追加しない。
誤認識を動作名に置換する機能ではない。同じ録音で認識精度と時間を比較して設定する。
[Qwen3-ASRの言語指定](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)、
[llama.cppのprefill契約](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

`tts_leading_silence_max_s`は初回音声の先頭無音だけを短縮する任意設定（既定0、上限0.75秒）。
発声直前60msを残し、発声後の休止を削らない。声ごとの録音で冒頭が欠けないことを確認する。

Windows標準音声を使う場合は`streaming_tts_endpoint`を`null`にし、`sapi_voice`に
インストール済みの音声名（例:`Microsoft Haruka Desktop`）を設定する。
音声の自然さ、生成待ち時間、聞き取りやすさをモデル音声と比較して選ぶ。
どちらも出力デバイスの名前を照合する。起動後に失敗した際の自動切替は行わない。

## 初回・デバイス変更後（人が設定）

自律実行を停止し、同じ専用プロファイルをDesktopモードで開く。
Steamの起動引数は `--profile=1 --no-vr`。設定後はVRChatを通常終了し、
全身制御を行う際は `--no-vr` を付けずに専用ランチャーから起動する。

1. Windowsに送話用の仮想オーディオデバイスを用意する。
2. VRChatのSettings → Audioで、Microphoneを **CABLE Output (VB-Audio Virtual Cable)** に設定する。System Default任せにせず、実際の選択名を確認する。
3. MyuMIQのTTS出力を **CABLE Input (VB-Audio Virtual Cable)** に指定する。番号は変動するため、名前の照合も必要。
4. VRChatの再生音は受信用の別出力へ送る。MyuMIQのloopback入力はその出力と一致させる。TTS送信用Cableを受信すると自己音声が混入する。
5. ゲーム内の音声送信方式を確認する。Push-to-talkの場合、解除状態だけでは送信されないことがある。
6. TTSを再生し、WindowsのCABLE Output録音レベル、VRChat内のマイク入力レベル、相手に聞こえることを順に確認する。

ドライバー導入後に再起動が必要な場合は、利用者と再起動時刻を調整する。
DesktopとVRの切替後にも、同じプロファイルのデバイス選択が維持されたか確認する。

## 起動時・会話前・実行中に必要な確認

**ミュートは固定セットアップではなく、変化する実行時状態。**
デバイス設定済みでも、ゲーム内でミュートなら相手に音声は届かない。

- 起動後、ゲーム内のミュート表示と送信方式を確認する。
- 発話前に送信可能か確認し、相手側の受信をテストする。
- 操作やモード変更後は再確認する。状態不明のときにトグルを繰り返さない。
- 将来の自動化では、観測したミュート状態に基づいて必要な場合だけ解除し、結果を再観測する。音声送信を止める利用者の操作を優先する。

現在、ゲーム内ミュートの継続観測・自動解除は未実装。TTSへの投入成功を送話成功と判定しない。
`remote_delivery: pending_verification` は相手側未確認を意味する。

## 切り分け

| 確認 | 成功が意味すること |
| --- | --- |
| 受信loopbackのレベル変化 | 再生音を取得できる（人の声とは限らない） |
| VADとASRの発話イベント | 発話を検出し文字起こしした |
| LLM返答・TTS submitted | 返答を音声出力へ要求した |
| Cable録音レベル変化 | TTSが仮想ケーブルへ届いた |
| VRChatマイク入力レベル変化 | ゲームがマイク音声を取得した |
| ミュート解除・相手が聞いた | 実際の送話を確認した |

設定名やデバイス番号、実測結果はリポジトリ外の端末用手順・記録に保存する。
