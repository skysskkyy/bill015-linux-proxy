# BILL-015 Local Responses Proxy

本项目按 `BILL-015_local_codex_proxy_design.md` 实现本地 OpenAI-compatible 代理。

当前开发范围：

- 已实现本地服务端、`/healthz`、`/metrics`、`/v1/models`
- 已实现 `/v1/responses` 主路径：Responses 请求归一化、强制 `emit_value` function-call、SSE function arguments 累积、`arguments.done` 后主动关闭上游、本地 Responses 事件回放
- 已实现 `exploit`、`dry-run`、`normal`、`verify` 模式
- 已实现 `/v1/chat/completions` 兼容层
- 已实现审计日志、脱敏、日志轮转、并发限制、超时、简单熔断
- `0.3.1` 起：exploit/verify 模式对上游预流式 `HTTP 5xx` / `do_request_failed` / 网络抖动做安全重试；一旦已收到上游 SSE 事件则不重试，避免重复执行或影响 abort 语义。
- `0.3.2` 起：上游 `emit_value` 的 `tool_calls` 不再只靠文本工具目录，而是按本轮 Codex 工具动态生成强类型 `oneOf` schema，约束工具名、namespace 和参数结构。
- 本阶段不做 Codex 配置接入

## 本地配置

现在默认直接读取项目根目录：

```text
S:\hack\packyapi.com\bill015_local_proxy\config.local.json
```

该文件已加入 `.gitignore`，用于保存本机私有配置。环境变量不再是必须项。

第一次使用时编辑：

```text
S:\hack\packyapi.com\bill015_local_proxy\config.local.json
```

把：

```json
"api_key": ""
```

改成你的上游 API key。不要把该文件提交或发给别人。

可参考模板：

```text
S:\hack\packyapi.com\bill015_local_proxy\config.local.example.json
```

核心字段：

```json
{
  "server": {"host": "127.0.0.1", "port": 8787},
  "upstream": {
    "base_url": "https://packyapi.com",
    "api_key": "PASTE_YOUR_KEY_HERE",
    "cookie": "",
    "user_id": "192833"
  },
  "mode": "exploit"
}
```

## 启动

无需设置环境变量，直接启动：

```powershell
cd S:\hack\packyapi.com\bill015_local_proxy
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

健康检查：

```powershell
curl.exe http://127.0.0.1:8787/healthz
curl.exe http://127.0.0.1:8787/v1/models
```

如需临时不用真实上游，把 `config.local.json` 里的：

```json
"mode": "dry-run"
```

## 离线自检

```powershell
cd S:\hack\packyapi.com\bill015_local_proxy
python scripts/selftest.py
```

自检只调用本地 FastAPI TestClient，不请求上游，不需要 API key。

如需运行完整测试/静态检查，先安装开发依赖：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app scripts tests
```

## 模式

- `exploit`：强制 function-call，收到 `response.function_call_arguments.done` 后关闭上游，再本地重组 Responses 输出。
- `dry-run`：不请求上游，返回将要构造的脱敏请求摘要。
- `normal`：普通转发 `/v1/responses` 到上游，不主动 abort。
- `verify`：同 exploit，并在配置 `upstream.cookie` 后尝试记录 `/api/user/self` pre/post delta。

切换模式：

```powershell
curl.exe -X POST http://127.0.0.1:8787/admin/mode `
  -H "Authorization: Bearer change-me-local-admin" `
  -H "Content-Type: application/json" `
  -d '{"mode":"dry-run"}'
```

管理 token 来自 `config.local.json`：

```json
"admin": {"token": "change-me-local-admin"}
```

## 最小 Responses 测试

`dry-run`：

```powershell
$body = @{ model="gpt-5.5"; input="Return the word OK."; stream=$false } | ConvertTo-Json
curl.exe -X POST http://127.0.0.1:8787/v1/responses -H "Content-Type: application/json" -d $body
```

`exploit` 真实上游测试需要 `config.local.json` 中填好 `upstream.api_key`，并尽量使用最小 prompt。

## 审计日志

默认写入：

```text
S:\hack\packyapi.com\bill015_local_proxy\proxy_evidence\audit.jsonl
```

日志不记录完整上游 API key/Cookie；默认不记录 prompt，也默认不记录 answer；如需本地留存回答，可在 `config.local.json` 里将 `logging.store_answers` 改为 `true`。

## 502 / 上游 500 稳定性

如果上游偶发返回 `HTTP 500`、`do_request_failed`，本地代理会在**尚未收到任何上游 SSE 事件**时自动重试：

```json
"limits": {
  "upstream_retries": 2,
  "upstream_retry_backoff_ms": 700
}
```

审计日志会记录：

```json
"retry_count": 1,
"retry_reasons": ["http_500"]
```

重试只发生在预流式失败阶段；如果已经收到模型输出、function-call 参数或其他 SSE 事件，则不会重试。
