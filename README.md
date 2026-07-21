# BILL-015 Local Responses Proxy

本项目按 `BILL-015_local_codex_proxy_design.md` 实现本地 OpenAI-compatible 代理。

当前开发范围：

- 已实现本地服务端、`/healthz`、`/metrics`、`/v1/models`
- 已实现 `/v1/responses` 主路径：Responses 请求归一化、上游工具调用桥接、SSE tool arguments/input 累积、工具参数边界后主动关闭上游、本地 Responses 事件回放
- 已实现 `exploit`、`dry-run`、`normal`、`verify` 模式
- 已实现 `/v1/chat/completions` 兼容层
- 已实现审计日志、脱敏、日志轮转、并发限制、超时、简单熔断
- `0.3.1` 起：exploit/verify 模式对上游预流式 `HTTP 5xx` / `do_request_failed` / 网络抖动做安全重试；一旦已收到上游 SSE 事件则不重试，避免重复执行或影响 abort 语义。
- `0.3.2` 起：上游 `emit_value` 的 `tool_calls` 不再只靠文本工具目录，而是按本轮 Codex 工具动态生成强类型 `oneOf` schema，约束工具名、namespace 和参数结构。
- `0.3.3` 起：默认启用 `bill015.strict_zero=true`，直接阻断 `auto-passthrough` / `normal` 这类可能产生真实扣费的路径；同时修复带工具结果的后续提问里“第二句话被旧问题盖住”的排序问题。
- `0.3.4` 起：默认关闭未知工具 `tool_bridge.allow_unknown_tools=false`；未出现在当前 Codex 工具 registry 的工具会被丢弃，缺工具时先走 `tool_search` 暴露工具。
- `0.3.5` 起：上下文按 token 预算和优先级组织，不再按大字符数硬截断；最新用户请求去重置顶，最近并行工具批次完整保留并做 head/tail 摘要。
- `0.3.6` 起：默认提升输出/等待预算：普通输出 8192、compaction 输出 8192、answer 缓冲 65536、上游总等待 300s、args_done 300s、idle 180s，默认不重试。
- `0.3.7` 引入 `native_tool_first` 实验策略：把 Codex 原生工具直接暴露给上游；最终回答走 `submit_final_answer` 工具。
- `0.3.8` 起：因 `native_tool_first` 实测会出现上游用量记录，默认和 `strict_zero=true` 下都恢复/强制 `emit_value` 早断开策略；`native_tool_first` 仅在 `strict_zero=false` 时作为实验选项。
- `0.3.9` 起：图片/截图输入会先被本地代理替换成“本地不支持图片输入”的文字提示再送上游，不再 strict-zero 422 或转发图片；上游未返回 `response.function_call_arguments.done` 时也会安全收尾，避免客户端断流报错。
- `0.4.6` 起：GPT-5.6/Codex Responses Lite 仍会从客户端 `additional_tools` 提取本地工具、`tool_search` 和子智能体 schema，但在 `strict_zero=true` 时上游会被降回标准 Responses `tools`/`instructions` 的单 `emit_value` 桥接请求，并去掉 Lite header/metadata，避免进入已观测到会计费的 5.6 Lite 上游路径。
- `0.4.7` 起：不再提示模型“每次工具调用都写一段进度文字”；普通工具调用示例默认 `answer=""`，但模型自愿返回的有用工具前说明仍会正常展示。默认并发提升到 4，并降低超大文本精确 tokenization 阈值来改善多任务长上下文速度。
- `0.4.8` 起：对“上网查/联网搜索/打开 URL/latest/current”等明确实时网页需求增加本地网络检索任务流：优先使用已暴露的 Chrome/Browser/Playwright/node_repl/jshook 工具；只有缺少具体浏览器工具但存在 `tool_search` 时，代理先本地返回 `tool_search_call` 做发现，不走上游 `web_search`。
- `0.4.9` 起：补强 GPT-5.6 能力保真：保留 Codex `model_reasoning_effort` 别名、GPT-5.6 默认 high reasoning；修复 CLI custom `exec` 工具在 `input=""` 时吞掉 `arguments` 的问题；工具 schema 从旧 96 上限提高并优先保留 `exec/apply_patch/tool_search/browser/node_repl/subagent` 等核心工具。
- `0.4.10` 起：优化本地热路径耗时：小/中型请求跳过不必要的精确 payload token 预检，复用已解析的工具历史，并减少大文本裁剪/最新用户消息省略时的重复扫描；上下文压缩、工具桥接和 strict-zero 语义不变。
- `0.4.11` 起：把含义混杂的 `strict_zero` 拆为 `force_emit_value`、`block_passthrough`、`block_normal_mode` 三个安全桥策略；这些开关不再禁用错误重试。`cyber_policy` 会立即切换下一把 Key 重跑，并让失败 Key 冷却一段时间。
- `0.4.1` 起：支持多 API key 池；当上游错误字段为 `error.code="cyber_policy"` 且 `error.message` 为完整 cybersecurity-risk 提示时，等待 10 分钟后自动切换下一把 key 并继续原请求。
- `0.4.2` 起：参考 Codex CLI 的上下文窗口机制，在 90% 阈值前预压缩；遇到 `context_length_exceeded` 时从最旧历史开始裁剪并重跑，同时保留最新用户请求和最新工具批次；reasoning-only 空流改为有限重试，不再伪装成成功回答。
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
  "mode": "exploit",
  "bill015": {
    "bridge_strategy": "emit_value",
    "final_answer_tool_name": "submit_final_answer",
    "force_emit_value": true,
    "block_passthrough": true,
    "block_normal_mode": true
  },
  "tool_bridge": {
    "allow_unknown_tools": false
  }
}
```

## API key 轮询

保留 `upstream.api_key` 作为首选 key，并在 `upstream.api_keys` 中按顺序配置额外 key。代理会去重后形成进程级密钥池。

当前轮询只针对日志中确认的上游 cyber policy 错误字段：SSE/JSON 错误对象中 `error.code="cyber_policy"`，且 `error.message` 等于完整的 `This content was flagged for possible cybersecurity risk... https://chatgpt.com/cyber` 提示。命中后，代理立即切到下一把尚未在本次请求中尝试过的 Key，并用原请求继续工作；失败 Key 默认进入 600 秒冷却，不会阻塞当前请求等待。

该专用切换由 `key_pool.cyber_policy_rotate` 控制，不受通用 `upstream_retries` 开关控制。每个 Key 在一次请求中最多尝试一次，全部耗尽后返回最后一个上游错误，不会无限循环。冷却时间由 `key_pool.failed_key_cooldown_seconds` 控制。

健康检查只暴露 key 数量、当前序号和不可逆短指纹，不返回完整 key；审计日志记录 `upstream_key_index`、`upstream_key_count` 和 `key_switch_count`。

配置示例：

```json
{
  "upstream": {
    "api_key": "PRIMARY_KEY",
    "api_keys": ["SECOND_KEY", "THIRD_KEY"]
  },
  "key_pool": {
    "cyber_policy_rotate": true,
    "failed_key_cooldown_seconds": 600
  }
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

- `exploit`：默认强制上游调用 `emit_value`；收到 `response.function_call_arguments.done` 后立即关闭上游，再本地重组 Responses 输出。`native_tool_first` 实验模式可处理 function/custom/tool_search/computer 等工具参数边界，但不用于 strict-zero。
- `dry-run`：不请求上游，返回将要构造的脱敏请求摘要。
- `normal`：普通转发 `/v1/responses` 到上游，不主动 abort。`bill015.strict_zero=true` 时会被拒绝。
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

## 安全桥策略与自动重试

默认安全桥配置为：

```json
"bill015": {
  "force_emit_value": true,
  "block_passthrough": true,
  "block_normal_mode": true
}
```

- 阻断 `auto-passthrough`：例如图片/文件等当前桥接层无法安全 early-abort 的原生请求，不再偷偷普通转发上游。
- 阻断 `normal` 模式：防止运行时误切换到普通转发。
- 强制使用 `emit_value` 提前断开策略。

这些安全桥开关与重试相互独立。预流式 HTTP 5xx、连接错误可以按 `limits.upstream_retries` 自动重试；空流、流超时和上下文错误仍由各自恢复配置控制。旧的 `strict_zero` 字段仅作为兼容回退，新配置不应再使用它。

## 工具桥严格模式

默认桥策略：

```json
"bill015": {
  "bridge_strategy": "emit_value",
  "final_answer_tool_name": "submit_final_answer",
  "native_tool_choice": "required",
  "native_parallel_tool_calls": false
}
```

默认效果：

- 上游只看到一个强制 `emit_value` function tool；代理在 `response.function_call_arguments.done` 立刻断开。
- `emit_value.tool_calls` 仍会按本轮 Codex 工具 registry 生成强类型 schema，避免完全靠自然语言描述工具。
- `bill015.force_emit_value=true` 时，即使配置了 `"native_tool_first"`，运行时也会回落到 `"emit_value"`。
- 如需实验更原生的工具选择，可同时设置 `"bridge_strategy": "native_tool_first"` 和 `"force_emit_value": false`；注意这可能产生上游用量记录。

默认配置：

```json
"tool_bridge": {
  "allow_unknown_tools": false,
  "auto_expand_search": false,
  "local_web_research_preflight": true,
  "schema_max_tools": 256
}
```

效果：

- 上游模型只能请求当前 Codex 请求里显式提供的工具，或 `tool_search` 暴露后的 deferred 工具。
- 未注册工具不会被下发给本地 Codex，避免模型幻觉工具名造成乱调用。
- 如果本轮没有任何工具 registry，`emit_value` 会限制为 `mode="answer"` 和 `tool_calls=[]`。
- 明确需要实时网页/URL 信息时，提示词会要求模型使用本地浏览器/HTTP 工具；若本轮只有 `tool_search` 可用，代理会先合成一次本地 `tool_search_call` 暴露工具，避免 gpt-5.6 停在“没有本地工具”或误走上游 `web_search`。
- 大工具目录下不再按字母顺序截掉后面的关键工具；`exec`、`apply_patch`、`tool_search`、浏览器/MCP、子智能体等会优先进入 `emit_value.tool_calls` 枚举，降低 5.6 “看得到目录但调不了工具”的概率。

## Reasoning / 能力保真

默认示例配置建议：

```json
"reasoning": {
  "effort": "high",
  "summary": "auto"
}
```

代理还会识别 Codex 常见的 `model_reasoning_effort` / `model_reasoning_summary` 字段以及 `x-codex-turn-metadata` 中的同名字段；客户端显式值优先。若 GPT-5.6 请求完全没有 reasoning 配置，代理默认给上游 `{"effort":"high"}`，避免 strict-zero de-Lite 后退回偏低推理强度。

## 上下文预算策略

代理会先按当前请求预留输出预算，再按优先级组织上游输入：

1. 当前用户请求
2. 最近一批本地工具结果 / 当前任务状态
3. 相关历史尾部与 compaction 摘要
4. 工具 schema

旧历史不再按固定字符数从头硬塞；长输出也会保留退出码、错误线和 head/tail 关键内容，避免最近工具批次超过 3 个时丢结果。

## 输出与等待预算

默认预算面向长答案、补丁和 JSON function arguments：

```json
"bill015": {
  "max_output_tokens": 8192,
  "compaction_max_output_tokens": 8192,
  "max_answer_chars": 65536
},
"upstream": {
  "timeout_seconds": 900
},
"limits": {
  "args_done_timeout_ms": 900000,
  "upstream_idle_timeout_ms": 900000,
  "stream_recovery_retries": 2,
  "upstream_retries": 2
}
```

如果要做超长重构/长补丁，可以临时把 `bill015.max_output_tokens` 和 `bill015.compaction_max_output_tokens` 调到 `16384`。

## 502 / 上游 500 稳定性

默认 `upstream_retries=2`，用于恢复尚未收到任何上游 SSE 事件时的 HTTP 5xx 和网络失败。一旦已收到上游事件，普通预流式重试不会执行，避免重复工具意图；流中断使用单独的 `stream_recovery_retries`。

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

预流式 HTTP 5xx 重试仍由 `upstream_retries` 控制；`ReadTimeout` / `response.function_call_arguments.done`
等待超时属于流式恢复路径，由 `stream_recovery_retries` 控制，会强化 final-action 指令并在需要时压缩历史后重新开流，避免把
`stream disconnected before completion` 直接透给 Codex 客户端。
