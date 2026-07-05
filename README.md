# OpenHands Custom Critic

DeepSeek APIで動作するOpenHands用のカスタムCritic（検証機能）です。

## 概要

OpenHandsに組み込みの`APIBasedCritic`は公式LLM-proxyサーバー专用のエンドポイント(`/classify`)を必要としますが、このカスタムCriticは任意のOpenAI互換API（DeepSeek等）を使ってエージェントのアクションを検証します。

## セットアップ

### 1. ファイルを配置

`custom_critic.py` をOpenHandsがimportできるパスに置きます。

```bash
cp custom_critic.py ~/.openhands/agent-canvas/
```

### 2. settings.json を更新

`~/.openhands/settings.json` の `verification` セクションを以下のように設定：

```json
"verification": {
    "critic_enabled": true,
    "critic_mode": "finish_and_message",
    "enable_iterative_refinement": false,
    "critic_threshold": 0.6,
    "max_refinement_iterations": 3,
    "critic_server_url": "https://api.deepseek.com/v1",
    "critic_model_name": "deepseek-chat",
    "critic_api_key": "sk-xxxxx"
}
```

### 3. PYTHONPATH を設定（必要に応じて）

`~/.openhands/agent-canvas/` に配置した場合、OpenHandsの起動時にPYTHONPATHに追加する必要があります。

## 動作原理

1. エージェントがアクションを完了するたびに `evaluate()` が呼ばれる
2. OpenHandsのイベント履歴から会話内容を抽出
3. DeepSeek APIに送信して「タスクが正しく完了したか」を評価
4. スコア(0-1)と理由を返す
5. 閾値(60%)未満の場合、最大3回まで再実行を試みる（反復改良が有効な場合）

## メリット

- **DeepSeek APIで動作** — Ollamaや公式プロキシ不要
- **LLM-as-judge方式** — 柔軟な評価が可能
- **設定のみで変更可能** — コード修正不要（settings.jsonの更新でOK）
