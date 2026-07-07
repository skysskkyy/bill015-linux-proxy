# BILL-015 本地 Codex 代理：工具调用闭环优化方案

本文档记录对 `S:\hack\packyapi.com\bill015_local_proxy` 的工具调用闭环优化设计。目标是让本地代理在 Codex 执行完工具后，下一轮上游模型能像原生 Codex 一样清楚知道：刚才调用了什么工具、参数是什么、成功/失败、关键输出是什么、下一步该继续调工具还是最终回答。

## 目标

当前流程已经支持：

```text
上游模型 emit_value(tool_call)
  ↓
本地代理合成 Codex Responses tool-call 事件
  ↓
Codex 本地执行 shell/apply_patch/browser/node_repl 等工具
  ↓
下一轮请求带 function_call_output/custom_tool_call_output/tool_search_output
```

但下一轮上游模型看到的工具结果仍然偏“拍平文本”，不够结构化。闭环优化后的目标流程：

```text
用户请求
  ↓
上游模型 emit_value(tool_call)
  ↓
本地代理合成 Codex 原生 tool_call 事件
  ↓
Codex 本地执行 shell/browser/apply_patch/node_repl
  ↓
下一轮请求带 function_call_output/custom_tool_call_output 等结果
  ↓
代理解析工具结果，生成 Tool Result Feedback
  ↓
上游模型明确知道工具执行结果
  ↓
继续调工具 or 最终回答
```

---

## 1. 新增工具历史解析层

新增文件：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\tool_history.py
```

负责解析原生 `input[]` 里的工具历史项：

```text
function_call
function_call_output
custom_tool_call
custom_tool_call_output
tool_search_call
tool_search_output
web_search_call
computer_call
computer_call_output
reasoning
assistant/user/developer message
```

建议统一输出结构：

```python
@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    call_type: str
    arguments: str
    index: int


@dataclass
class ToolOutputRecord:
    call_id: str
    name: str
    call_type: str
    output: str
    exit_code: int | None
    success: bool | None
    stderr_excerpt: str
    stdout_excerpt: str
    raw_chars: int
    index: int


@dataclass
class ToolHistory:
    calls: list[ToolCallRecord]
    outputs: list[ToolOutputRecord]
    latest_outputs: list[ToolOutputRecord]
    pending_calls: list[ToolCallRecord]
    failed_outputs: list[ToolOutputRecord]
    successful_outputs: list[ToolOutputRecord]
    exposed_deferred_tools: list[dict]
```

重点：所有工具调用和结果都用 `call_id` 关联。

---

## 2. 在 `NormalizedRequest` 里增加闭环状态

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\models.py
```

给 `NormalizedRequest` 增加字段：

```python
tool_history: ToolHistory | None = None
latest_tool_summary: str = ""
pending_tool_call_count: int = 0
latest_tool_failed: bool = False
```

如果不想让 `models.py` 直接依赖 `tool_history.py`，可以先把 `tool_history` 标成 `Any`：

```python
tool_history: Any = None
```

---

## 3. 在请求归一化时解析工具闭环

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\proxy.py
```

在 `normalize_responses_request()` 里加入：

```python
from .tool_history import parse_tool_history, render_tool_feedback_for_model

...

tool_history = parse_tool_history(raw_input)
latest_tool_summary = render_tool_feedback_for_model(tool_history)
```

然后塞进 `NormalizedRequest`：

```python
return NormalizedRequest(
    ...
    tool_history=tool_history,
    latest_tool_summary=latest_tool_summary,
    pending_tool_call_count=len(tool_history.pending_calls),
    latest_tool_failed=bool(tool_history.failed_outputs and tool_history.failed_outputs[-1] in tool_history.latest_outputs),
)
```

注意：不要再只依赖 `flatten_responses_input()`。工具结果必须作为单独的结构化反馈块传给上游模型。

---

## 4. 生成专用 Tool Result Feedback 块

在 `tool_history.py` 增加：

```python
def render_tool_feedback_for_model(history: ToolHistory) -> str:
    ...
```

输出示例：

```text
Recent local Codex tool results:

[1] shell_command call_id=call_xxx status=success
arguments:
{"command":"python scripts/selftest.py"}

result:
Exit code: 0
Wall time: 1.2s
Key output:
[ok] selftest passed

[2] apply_patch call_id=call_yyy status=success
input:
*** Begin Patch ...

result:
Success. Updated files:
- app/proxy.py

Decision guidance:
- If the tool output fully answers the user, return mode=answer.
- If more local action is needed, return mode=tool_call.
- If a tool failed, inspect the error and either retry with corrected arguments or explain the blocker.
```

---

## 5. 输出裁剪规则

避免工具输出过大撑爆上下文。

建议规则：

```text
每个工具输出最多 12k chars
最新 3 个工具输出保留较多
更早输出只保留：
- 工具名
- 参数摘要
- exit code
- 关键错误行
- 最后 100 行
```

不同工具的特殊处理：

### shell_command

提取：

```text
Exit code
Wall time
stdout tail
stderr/error lines
```

### apply_patch

提取：

```text
新增/修改/删除文件列表
成功/失败原因
patch 摘要
```

### tool_search_output

提取：

```text
暴露了哪些 deferred tools
namespace
tool names
description 前 300 chars
```

### browser/chrome/node_repl

提取：

```text
url
title
visible text excerpt
console error
network error
```

---

## 6. 修改 BILL-015 上游 prompt

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\proxy.py
```

在 `build_bill015_payload()` 的 system prompt 里加入固定规则：

```text
You are in a Codex local tool loop.

If recent local tool results are provided:
- First inspect those results.
- Do not repeat the same tool call unless the previous call failed and you change the arguments.
- If the result is sufficient, call emit_value with mode="answer".
- If another local action is needed, call emit_value with mode="tool_call".
- Preserve Codex native tool names exactly.
```

然后在 user content 前插入：

```python
user_content = n.user_input
if n.latest_tool_summary:
    user_content = n.latest_tool_summary + "\n\n--- Conversation / user request ---\n" + n.user_input
```

---

## 7. 防重复工具调用

在 `tool_history.py` 增加：

```python
def is_repeated_call(name: str, arguments: str, recent_calls: list[ToolCallRecord]) -> bool:
    ...
```

在 feedback 中提示：

```text
Recently repeated calls:
- shell_command {"command":"..."} already succeeded.
Do not call it again unless arguments change.
```

后续可以更进一步：代理层直接拦截“完全重复且刚成功”的工具调用，返回提示给 Codex，而不是让它真的执行。

---

## 8. 多工具并行闭环

现在代理已支持 emit 多个 tool calls。闭环需要处理多个输出：

```text
call_A -> output_A
call_B -> output_B
call_C -> output_C
```

`render_tool_feedback_for_model()` 应按原始顺序展示，并标注：

```text
Parallel batch result:
- 3 calls requested
- 3 completed
- 2 success
- 1 failed

Failed calls:
- call_C shell_command timeout
```

---

## 9. pending call 处理

如果原生 input 里有 call 但没有 output：

```text
function_call 有了
function_call_output 没有
```

说明工具还没返回或上下文不完整。

summary 中应写：

```text
Pending local tool calls without outputs:
- call_x shell_command {...}
Do not assume their result.
```

避免模型编造工具结果。

---

## 10. deferred tools 闭环

当前代理已经会解析 `tool_search_output.tools` 并合入 catalog。下一步要把它也写进工具反馈：

```text
Tool search exposed these local tools:
- mcp__node_repl.js
- mcp__node_repl.js_reset
- browser.open
- chrome.navigate
```

并提示：

```text
These tools are now callable by exact name.
Prefer them over another tool_search if suitable.
```

这样模型不会每轮重复 `tool_search`。

---

## 11. 审计日志增强

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\audit.py
```

或在 `audit_from_result()` 中增加：

```json
{
  "latest_tool_outputs": 3,
  "latest_tool_failed": false,
  "pending_tool_calls": 0,
  "tool_loop_decision": "answer/tool_call",
  "repeated_tool_call_detected": false
}
```

方便排查模型为什么一直调工具。

---

## 12. 自测覆盖

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\selftest.py
```

新增测试：

### A. shell_command 闭环

输入：

```json
[
  {"type":"function_call","name":"shell_command","call_id":"call_1","arguments":"{\"command\":\"pwd\"}"},
  {"type":"function_call_output","call_id":"call_1","output":"Exit code: 0\nOutput:\nS:\\hack\\packyapi.com"}
]
```

断言：

```text
latest_tool_summary 包含 shell_command
latest_tool_summary 包含 Exit code: 0
latest_tool_failed == false
```

### B. apply_patch 闭环

断言能识别：

```text
custom_tool_call
custom_tool_call_output
Success. Updated files
```

### C. 失败工具

输入：

```text
Exit code: 1
Traceback...
```

断言：

```text
latest_tool_failed == true
summary 提示 retry/change args
```

### D. pending call

只有 call 没有 output。

断言：

```text
pending_tool_call_count == 1
summary 提示 Do not assume result
```

### E. deferred tools

输入 `tool_search_output.tools`。

断言：

```text
mcp__node_repl.js 被加入 catalog
summary 里列出 exposed tools
```

---

## 13. 原生日志 replay 测试

新增脚本：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\replay_native_logs.py
```

功能：

```text
读取 proxy_evidence/native_long_chat/extracted/*.json
逐个 normalize_responses_request()
检查：
- 不崩
- tool_history 能解析
- latest_tool_summary 不为空
- compaction 正确识别
- deferred tools 正确合并
```

这是防止以后改坏原生兼容的关键。

---

## 推荐实施顺序

一步到位按以下顺序实现：

1. 新建 `app/tool_history.py`
2. `NormalizedRequest` 增加闭环字段
3. `normalize_responses_request()` 接入 `parse_tool_history()`
4. `build_bill015_payload()` 注入 Tool Result Feedback
5. 增加防重复工具调用提示
6. 增加 deferred tools feedback
7. 增强 audit 日志
8. 补充 `scripts/selftest.py`
9. 新增 `scripts/replay_native_logs.py`
10. 跑全量自测并重启代理

---

## 最终效果

优化完成后，模型在每个工具调用后会明确看到：

```text
刚才调用了什么
参数是什么
工具是否成功
关键输出是什么
是否还有 pending call
是否暴露了新的 deferred tools
是否重复调用了已成功工具
下一步应该 answer 还是继续 tool_call
```

这会显著减少：

- 重复执行同一个 shell 命令
- 工具失败后继续瞎猜
- tool_search 重复暴露同一批工具
- 忽略 apply_patch/shell 输出
- 长会话中忘记刚才工具执行结果

目标是让本地代理的工具调用闭环尽量接近 Codex 原生体验，同时继续保持低/零上游用量路径。
