# モデルとサービスの交換設定

実装: 2026-09-20。会話・身体判断・遅い思考を別の役割として設定する。
世界・身体・履歴・目的の正本、採用権限、PAMIQとMotorはローカルに残る。
APIモデルを採用しても、自然文や座標をそのまま身体へ流さない。

## 最短の設定手順

Python環境に本パッケージをインストールしてから実行する。
`python -m myumiq_vrchat`でも同じコマンドが使える。

```powershell
# ローカル用の既定設定。モデルとサーバーは次節の手順で準備する。
myumiq models init --output ../myumiq-config/autonomy.json

# OpenRouter用。会話Gemma 4 31B、思考Gemma 4 26B A4B、判断Jev。
myumiq models init --profile openrouter --output ../myumiq-config/api.json

# 会話・思考・身体判断で画像を使い、要求された動作の採否を会話へ戻す。
myumiq models init --profile openrouter-vision --output ../myumiq-config/vision-api.json

# 既存の音声デバイス、身体モデル、学習、記憶設定を保ってモデルだけ交換する。
myumiq models init --profile openrouter --base ../myumiq-config/autonomy.json --output ../myumiq-config/updated.json

# API鍵を設定済みのプロセスで、実際の役割ごとの要求を試す。
myumiq models check --config ../myumiq-config/api.json --output ../myumiq-config/model-check.json
```

生成したJSONの`llm`、`purpose.planner`、`decision`を編集すれば、役割単位で
モデル、ローカル/API、提供先、待ち時間、拡張adapterを交換できる。
`--profile hybrid`はローカル会話＋OpenRouter判断・思考。
`--profile-file <JSON>`には`speech`・`thought`・`decision`の3項目を持つ
独自profileを渡せる。モデル名の許可リストはない。
既存ファイルは上書きしない。新規設定のstateは出力先の`state/`配下で、
`--state-dir`で変えられる。`--base`を指定した場合は既存のstateを引き継ぐ。

| profile | 会話 | 思考 | 身体判断 |
| --- | --- | --- | --- |
| `local`（既定） | LFM2.5-1.2B-JP-202606、18487 | Gemma 4 26B A4B、18488 | 同じGemmaサーバーで候補選択 |
| `openrouter` | Gemma 4 31B | Gemma 4 26B A4B | Jev 1.13の候補採点 |
| `openrouter-vision` | Gemma 4 31B＋画像 | Gemma 4 26B A4B＋画像 | Gemma 4 26B A4Bで画像・要求能力・候補選択 |
| `hybrid` | ローカルLFM | OpenRouter Gemma | OpenRouter Jev |

API会話の配信先は`coreweave/fp4`、思考は`deepinfra/fp8`を初期値とする。
画像用の`openrouter-vision`は会話に`coreweave/fp4`、思考・身体判断に
`nextbit/bf16`を指定する。画像とJSON出力を同時に利用できる接続先が必要。
設定した画像経路も`models check`の検査対象になる。
[視界と行動の契約](visual-conversation-actions.md)も参照。
同じモデルでも配信先の待ち時間は変わるため、`route.provider`は役割別に編集できる。
対応する量子化・構造化出力は[OpenRouterのGemma 4 31B配信先一覧](https://openrouter.ai/api/v1/models/google/gemma-4-31b-it/endpoints)で確認する。
明示した配信先が失敗した場合に、別モデルや別providerへ黙って切り替えることはない。
`decision.refresh_s`は変化のない身体状態を再判断する間隔（既定5秒）。
新しい発話、対象、目的、動作結果は間隔を待たずに判断を起こす。
身体の観測・動作周期や会話の生成周期を、この値へ揃える設定ではない。

これは接続設定の既定値であり、全機種での性能保証ではない。Qwen2.5の暗黙起動や
不調時の古いモデルへの切替はない。音声認識・合成は別のvoice設定、
全身動作は別の身体policy設定で指定する。

### ローカルサーバーの準備

モデル自身のchat templateとJSON Schema出力に対応したローカルサーバーを使う。
例として、現在の[公式llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)なら:

```powershell
# 日本語会話。公式GGUFを取得し、CPUで起動する例。
llama-server -hf LiquidAI/LFM2.5-1.2B-JP-202606-GGUF:Q4_K_M --host 127.0.0.1 --port 18487 --alias LiquidAI/LFM2.5-1.2B-JP-202606 --ctx-size 8192 --parallel 1 --threads 4 --gpu-layers 0 --jinja

# 別の端末窓で、取得済みのGemma 4対応GGUFを起動する。
llama-server --model <GemmaのGGUFへのパス> --host 127.0.0.1 --port 18488 --alias google/gemma-4-26B-A4B-it --ctx-size 16384 --parallel 2 --reasoning off --jinja
```

LM Studio等でも同じモデルID（または設定へ実際のalias）とポートを指定できる。
サーバーが別名を返す場合、`model`を実際のalias、`model_identity`を重みIDにする。
GPU割当や量子化は使用環境で設定する。Gemmaの「A4B」は有効パラメータ数であり、
全重みが4B分のメモリに収まる意味ではない。26Bの重み・KV cache・VRChat・音声合成が
同時に動くメモリと処理速度が必要。収まらない役割だけAPIへ変更できる。
出典: [Gemma 4公式モデル](https://huggingface.co/google/gemma-4-26B-A4B-it)、
[LFM日本語公式GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606-GGUF)。

モデルチェックは、別の入力に応じた日本語返答、Purposeの構造化出力、
「しゃがむ→立つ」の候補切替と判断期限を検査する。合成したテキストだけを使い、
利用者の記憶を開かず、VRChat・マイク・Motorへ接続しない。
失敗した役割と修正箇所を報告し終了コード1、全項目成功なら0を返す。
ローカル判断の既定期限は5秒、Plannerは20秒。VR稼働中の同時処理でも間に合うかは
実機で確認する。チェック成功は音声到達・姿勢精度・新技能学習の証明ではない。

### 認証情報

OpenRouterの鍵を起動プロセスの`OPENROUTER_API_KEY`へ設定する。
PowerShell 7なら入力を画面に表示せず設定できる:

```powershell
$env:OPENROUTER_API_KEY = Read-Host 'OpenRouter API key' -MaskInput
myumiq models check --config ../myumiq-config/api.json
```

秘密情報管理ソフトから環境変数へ渡す方法も使える。別名を使うなら各API roleの
`api_key_env`を同じ名前へ変更する。JSONに鍵そのものは書かない。
リポジトリのコードは特定のアプリや端末の認証ファイルを探索しない。
API profileを生成するだけでは通信・課金は発生しない。`models check`や実行時に
設定したAPIを呼ぶ。実機の起動へは生成済みJSONを`body_console run`の
`--autonomous-config`で渡す（その他の機器設定は[身体コンソール](body-console.md)）。

## 設定の場所

`AutonomousConfig`を読む既存の起動経路で次の項目を指定する。
機種固有のJSONはこれまでどおりリポジトリ外へ置く。

| 役割 | 設定 | 対応する契約 |
| --- | --- | --- |
| 会話 | `llm` | reply/topic/remember_quotes。身体操作の出口なし |
| 遅い思考・目的提案 | `purpose.planner`（省略時は独立Planner無効） | world/body/drives/履歴/能力 → 検証済みPurpose提案 |
| 各候補の独立採点 | `decision.backend` | DecisionInput → ScoreResult |
| 候補をまとめて選択 | `decision.selection.llm` | DecisionInput → candidate_idまたはnull |
| ASR / VAD / 音声出力 | voice JSONの`asr_adapter` / `vad_adapter` / `output_adapter` | 既存の音声塊・検出・再生契約 |
| 画像検出 | vision JSONの`detector_adapter` | 画像 → SpatialDetection列 |

`decision.backend`と`decision.selection`はどちらか一つ。採点値と選択IDは
別の意味を持ち、選択結果から偽の点数を作らない。
会話と判断・Plannerは異なるモデル重みを指定する。同じAPIサーバーを使うことは可能。
ローカルの別名モデルでは`model_identity`に実際の重みIDを指定する。
ローカル会話と判断には別のサーバーポートも必要。
新しいremote/extension会話設定は、身体と会話が分離したpurpose＋decision経路で使う。

## ローカルとOpenRouterの切替

`llm`、`purpose.planner`、`decision.selection.llm`は同じ生成設定型を使える。
各役割へ個別に次のどちらかを入れる。全役割をまとめて切り替える必要はない。

ローカル例（`adapter`省略の既存設定も有効）:

```json
{
  "adapter": "local_chat",
  "base_url": "http://127.0.0.1:18488/v1",
  "model": "my-separate-thought-model",
  "timeout_s": 20,
  "structured_output": true
}
```

OpenRouterのGemma例:

```json
{
  "adapter": "openrouter_chat",
  "model": "google/gemma-4-26b-a4b-it",
  "route": {
    "provider": "deepinfra/fp8",
    "max_prompt_price": 0.2,
    "max_completion_price": 0.5
  },
  "api_key_env": "OPENROUTER_API_KEY",
  "timeout_s": 20,
  "max_tokens": 1024,
  "reasoning": {"enabled": false}
}
```

モデルID・providerは交換できる。例の提供先は2026-09-20に公式の
[Gemmaの提供先一覧](https://openrouter.ai/api/v1/models/google/gemma-4-26b-a4b-it/endpoints)
で構造化出力対応を確認したもの。配信状況は変わる。
`reasoning.enabled`は思考トークンの使用設定で、MyuMIQのPlanner役割を無効にする設定ではない。
対応モデルでは`enabled: true`と`effort: low/medium/high/minimal`も指定できる。
`max_tokens`は思考分も含む生成上限。無指定なら役割側の短い既定上限を使用する。

価格はUSD / 100万tokenの提供先選定上限。累積利用額を止める予算機構ではない。
API側のクレジット制限とは別に、入力128 KiB・応答64 KiB・timeoutも制限する。
返却されたモデルIDを検証する。提供先が日付付きIDを返す場合は
`expected_model`へその正確なIDを設定できる。revisionを提供しないAPIについて、
同一重みを独立に検証できたとは扱わない。

OpenRouter adapterは公式HTTPS endpoint固定で、鍵は環境変数からそのプロセス内だけで読む。
設定・ログへ鍵や認証応答を書かない。LocalChatのURLをリモートに変える抜け道は作らない。
[構造化出力](https://openrouter.ai/docs/guides/features/structured-outputs)と
[提供先固定](https://openrouter.ai/docs/guides/routing/provider-selection)を使用し、
`only/order`は単一provider、`allow_fallbacks: false`。モデルのfallback配列も送らない。
通信・形式・能力エラー時は失敗として記録し、次の設定済み試行まで現在の有限動作と姿勢保持を続ける。

## Jevの採点にAPIを使う

```json
{
  "backend": {
    "backend": "jev_openrouter",
    "jev_model": "typesafe/jev-1.13",
    "expected_model": "typesafe/jev-1.13-20260917",
    "route": {
      "provider": "typesafe",
      "max_prompt_price": 0.2,
      "max_completion_price": 0.5
    },
    "api_key_env": "OPENROUTER_API_KEY",
    "allow_state_only": true,
    "timeout_s": 5
  },
  "max_age_s": 5
}
```

このオブジェクトを`decision`へ入れる。Jevは
[Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request)
を使い、各候補を独立したscore質問として送る。Chat Completionsへ変換しない。
上記の日付付きIDは試験時の返却ID。提供モデルの更新時には明示的に設定を変更する。
ローカル採点なら既存の`backend: http`で隔離したローカルscorerを指定する。
Jevはテキストの状態だけを使用する。画像入力の省略を許すには
`allow_state_only: true`が必要で、結果にも`image_used: false`を記録する。
以下のGemma等の`selection`では、画像対応モデルに`use_image: true`を指定すると
画面と構造化状態を合わせて判断できる。この場合の`allow_state_only`は画像欠落時の
明示的な代替設定となる。画像への対応はモデル名だけでなくproviderでも確認する。
検出ラベルは推定であり、画像を使っても発話者やプレイヤー本人の確認にはならない。

Gemmaに候補選択させる場合は、上記の`backend`を使わず次の形にする:

```json
{
  "selection": {
    "llm": {
      "adapter": "openrouter_chat",
      "model": "google/gemma-4-26b-a4b-it",
      "route": {"provider": "deepinfra/fp8"},
      "max_tokens": 256,
      "timeout_s": 5
    },
    "allow_state_only": true
  },
  "max_age_s": 5
}
```

## Plannerと身体の分担

`purpose.planner`は既存のPurposeRunnerの別workerで動く。`planner_interval_s`
（既定30秒）、`planner_max_age_s`（既定30秒）を指定できる。各roleの処理中要求は最大1件。
世界・身体・欲求・確定会話・直近の動作結果から目的と手順を提案する。
同じPurposeでも、ここでは有限動作を直接開始せず、能力と対象を再検証したうえで
commitmentと`planner_proposal`へ保存する。身体判断がそれを文脈として使い、既存の
候補・実行検証を通す。会話生成には任意の身体操作権限を与えない。

発話や手動操作の世代が変わった古い結果・期限切れ・未対応能力の提案は採用しない。
API失敗時にも既存の目的を上書きせず、姿勢を自動的に標準へ戻さない。
任意の長期計画を完遂できることや未学習の身体技能が増えることは、この接続機構では保証しない。

## 別方式のアダプターを追加する

任意のAPIやローカルruntimeは、対応する契約を実装した小さなPythonパッケージで接続する。
設定に任意の実行コードを書かせず、インストール済みのentry point名を選ぶ。

```toml
[project.entry-points."myumiq_vrchat.generation"]
my_backend = "my_backend:create_generator"
```

```json
{
  "adapter": "extension",
  "model": "actual-model-identity",
  "extension": {"name": "my_backend", "options": {"api_key_env": "MY_API_KEY"}},
  "timeout_s": 20,
  "max_tokens": 512
}
```

Factoryは`options: dict`を受け取る。生成factoryには外側の`model`と`timeout_s`も
同名のoptionsとして渡す（同名の値をoptionsに書いても外側の設定が優先する）。
生成adapterの`request(messages, schema, name, max_tokens)`
は`generation.GenerationResult(value, elapsed_s, metadata)`を返し、`close()`を持つ。
URLなど固有の接続設定はextensionのoptionsに指定し、独自adapter自身が
通信期限とキャンセルを実装する。外側も応答の期限を検査するが、任意のPython処理を強制停止するものではない。

| entry point group末尾 | factoryが返す契約 | 選択設定 |
| --- | --- | --- |
| `generation` | JsonGenerator.request / close | 上記extension |
| `scorer` | Scorer.score / close → ScoreResult | `decision.backend: {backend: extension, extension: {name, options}}` |
| `asr` | transcribe(audio, sample_rate) → str | voiceの`asr_adapter: {name, options}` |
| `vad` | accept(AudioFrame) → SpeechEvent列、active | voiceの`vad_adapter` |
| `speech_output` | speak(text), stop(), speaking | voiceの`output_adapter` |
| `detector` | detect(image, timestamp) → SpatialDetection列 | visionの`detector_adapter` |

すべてのgroupには`myumiq_vrchat.`を付ける。未知・重複登録・メソッド不一致はエラー。
API鍵はoptionsにも直接入れず、adapterが読む環境変数名だけを指定する。
ASR/VADのextensionを指定した役割では、未使用のWhisper/Sileroモデル設定は不要。
音声出力extensionは現行の出力ポート全体を所有し、実際の出力デバイス設定もoptionsに持つ。
barge-inの`stop()`後も再利用可能であることが必要。

音声塊ASRをnative streaming ASRと呼んだり、Realtime STSをこの同期JSON契約に
押し込んだりしない。それらは[cognition-voice-adapters.md](cognition-voice-adapters.md)の
Session契約を実装してから追加する。Motor・backendの交換は既存の学習済みactor成果物と
明示的なデバイスadapter設定を使い、この生成拡張に生のcontroller操作を公開しない。

## 検証

オフラインテストはローカル設定互換、role分離、HTTP payload、鍵欠落、API異常・不完全JSON、
サイズ制限、未知adapter、Plannerの世代・期限・実行権限を検証する。
実API試験と費用・遅延の記録は端末側成果物へ置き、VRChatの動作確認とは分ける。
## Direct body selection with the thought model

Independent candidate scoring and choosing one candidate are separate configured
contracts. If a scorer keeps selecting wait in a situation where a new motion is
required, configure the existing Selection adapter explicitly. For example, an
OpenRouter decision role can use the discussed Gemma model:

```json
{
  "selection": {
    "llm": {
      "adapter": "openrouter_chat",
      "model": "google/gemma-4-26b-a4b-it",
      "route": {"provider": "deepinfra/fp8", "max_prompt_price": 0.2, "max_completion_price": 0.5},
      "api_key_env": "OPENROUTER_API_KEY",
      "timeout_s": 5.0,
      "max_tokens": 256,
      "reasoning": {"enabled": false}
    },
    "allow_state_only": true
  },
  "max_age_s": 5.0
}
```

Use this object as `decision`, replacing its scorer `backend`. Local Selection
uses the same structure with a `local_chat` model/endpoint. It selects one supplied
action ID or abstains and never manufactures independent scores. Thought and body
selection may share a model; speech remains a separate role. Changing this
configuration does not train or replace the whole-body actor. Jev remains available
through its scorer adapter. No runtime failure automatically swaps either model.
