# BILL-015 本地 Codex 代理：Responses 事件完整还原优化方案

本文档记录对 `S:\hack\packyapi.com\bill015_local_proxy` 的 Responses SSE/JSON 事件还原优化设计。目标是让本地代理合成给 Codex Desktop 的事件流尽量贴近原生 Responses API，提升 Codex UI 展示、工具调用、reasoning 展示、compaction、usage、错误恢复和长会话稳定性，同时继续保持 BILL-015 低/零上游用量路径。

## 目标

当前代理已支持基础事件：

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.function_call_arguments.delta
response.function_call_arguments.done
response.custom_tool_call_input.delta
response.custom_tool_call_input.done
response.completed
response.failed
error
```

但原生 Codex / Responses 里还有更多细节事件和字段。优化目标：

```text
1. 更完整还原 message / function_call / custom_tool_call / tool_search_call / web_search_call 事件
2. 更完整还原 reasoning summary 事件
3. 更完整还原 response object 字段
4. 更完整还原 output item 生命周期
5. 支持 annotation / refusal / incomplete / failed / cancelled 等边界状态
6. 支持多 content part 和多 output item 并行
7. 支持 response.completed 里的完整 usage / metadata / parallel_tool_calls / tool_choice
8. 建立 native event replay 测试，防止格式回退
```

---

## 1. 新增事件模型层

新增文件：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\event_models.py
```

目的：不要在 `response_events.py` 里到处手写 dict。先定义统一事件构造器。

建议结构：

```python
@dataclass
class EventContext:
    response_id: str
    model: str
    created_at: int
    sequence_number: int = 0
    output_index: int = 0


@dataclass
class OutputItemRef:
    id: str
    output_index: int
    item_type: str
    call_id: str | None = None
    name: str | None = None
```

提供基础函数：

```python
def response_event(event_type: str, **payload) -> dict[str, Any]
def make_response_object(...)
def make_message_item(...)
def make_function_call_item(...)
def make_custom_tool_call_item(...)
def make_tool_search_call_item(...)
def make_web_search_call_item(...)
def make_reasoning_item(...)
```

所有事件都必须自动带：

```json
{
  "type": "...",
  "sequence_number": N
}
```

---

## 2. 统一 Response object 字段

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\response_events.py
```

当前 response object 字段偏少。建议统一为：

```json
{
  "id": "resp_local_xxx",
  "object": "response",
  "created_at": 1780000000,
  "status": "in_progress|completed|failed|incomplete|cancelled",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-5.5",
  "output": [],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {...},
  "store": false,
  "temperature": null,
  "text": {"format":{"type":"text"},"verbosity":"low"},
  "tool_choice": "auto",
  "tools": [],
  "top_p": null,
  "truncation": null,
  "usage": {...},
  "user": null,
  "metadata": {}
}
```

不要强行填所有字段；但建议保留以下核心字段：

```text
id
object
created_at
status
model
output
parallel_tool_calls
tool_choice
tools
usage
error
incomplete_details
reasoning
text
metadata
```

`NormalizedRequest` 里已有：

```text
reasoning
metadata
client_metadata
parallel_tool_calls
tool_choice
text_config
prompt_cache_key
usage_estimate
```

response object 应尽量从这些字段继承。

---

## 3. 完整 message 文本事件生命周期

普通文本回答应严格按生命周期输出：

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta*
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
[DONE]
```

### output_item.added

```json
{
  "type": "response.output_item.added",
  "output_index": 0,
  "item": {
    "id": "msg_xxx",
    "type": "message",
    "status": "in_progress",
    "role": "assistant",
    "content": []
  }
}
```

### content_part.added

```json
{
  "type": "response.content_part.added",
  "item_id": "msg_xxx",
  "output_index": 0,
  "content_index": 0,
  "part": {
    "type": "output_text",
    "text": "",
    "annotations": []
  }
}
```

### delta

```json
{
  "type": "response.output_text.delta",
  "item_id": "msg_xxx",
  "output_index": 0,
  "content_index": 0,
  "delta": "...",
  "logprobs": []
}
```

### done

```json
{
  "type": "response.output_text.done",
  "item_id": "msg_xxx",
  "output_index": 0,
  "content_index": 0,
  "text": "完整文本",
  "logprobs": []
}
```

### item done

```json
{
  "type": "response.output_item.done",
  "output_index": 0,
  "item": {
    "id": "msg_xxx",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [
      {"type":"output_text","text":"完整文本","annotations":[]}
    ]
  }
}
```

---

## 4. 支持多 content part

虽然当前大部分回答只有一个 `output_text`，但原生可能有多个 content part：

```text
output_text
refusal
annotation-bearing text
```

建议内部结构支持：

```python
@dataclass
class MessagePart:
    type: str
    text: str = ""
    annotations: list[dict] = field(default_factory=list)
```

生成时按 `content_index` 递增。

### refusal part

如果上游返回或本地策略返回 refusal，可合成：

```json
{
  "type": "refusal",
  "refusal": "..."
}
```

对应事件：

```text
response.refusal.delta
response.refusal.done
```

如果 Codex Desktop 对 refusal 事件支持不稳定，可以先只把 refusal 降级为 `output_text`，但数据模型要预留。

---

## 5. function_call 事件完整还原

函数工具调用应输出：

```text
response.output_item.added
response.function_call_arguments.delta*
response.function_call_arguments.done
response.output_item.done
response.completed
[DONE]
```

### added item

```json
{
  "id": "item_xxx",
  "type": "function_call",
  "status": "in_progress",
  "call_id": "call_xxx",
  "name": "shell_command",
  "arguments": ""
}
```

### arguments.delta

```json
{
  "type": "response.function_call_arguments.delta",
  "item_id": "item_xxx",
  "output_index": 0,
  "call_id": "call_xxx",
  "name": "shell_command",
  "delta": "{...chunk...}"
}
```

### arguments.done

```json
{
  "type": "response.function_call_arguments.done",
  "item_id": "item_xxx",
  "output_index": 0,
  "call_id": "call_xxx",
  "name": "shell_command",
  "arguments": "完整 JSON 参数"
}
```

### item.done

```json
{
  "type": "response.output_item.done",
  "output_index": 0,
  "item": {
    "id": "item_xxx",
    "type": "function_call",
    "status": "completed",
    "call_id": "call_xxx",
    "name": "shell_command",
    "arguments": "完整 JSON 参数"
  }
}
```

---

## 6. custom_tool_call / apply_patch 完整还原

原生日志里 `apply_patch` 经常表现为：

```text
custom_tool_call
response.custom_tool_call_input.delta
response.custom_tool_call_input.done
```

生命周期：

```text
response.output_item.added
response.custom_tool_call_input.delta*
response.custom_tool_call_input.done
response.output_item.done
response.completed
[DONE]
```

### item

```json
{
  "id": "ctc_xxx",
  "type": "custom_tool_call",
  "status": "in_progress|completed",
  "call_id": "call_xxx",
  "name": "apply_patch",
  "input": ""
}
```

### input.delta

```json
{
  "type": "response.custom_tool_call_input.delta",
  "item_id": "ctc_xxx",
  "output_index": 0,
  "delta": "*** Begin Patch\n..."
}
```

### input.done

```json
{
  "type": "response.custom_tool_call_input.done",
  "item_id": "ctc_xxx",
  "output_index": 0,
  "input": "完整 freeform input"
}
```

---

## 7. tool_search_call 完整还原

本地代理推荐用 `tool_search` 暴露 browser/chrome/node_repl/shell/curl 等本地工具。

`tool_search_call` 是 Responses item，不是普通 function_call。

生命周期建议：

```text
response.output_item.added
response.output_item.done
response.completed
[DONE]
```

### item

```json
{
  "id": "tsc_xxx",
  "type": "tool_search_call",
  "status": "completed",
  "call_id": "call_xxx",
  "execution": "client",
  "arguments": {
    "query": "browser chrome node_repl web search",
    "limit": 8
  }
}
```

对于 `in_progress` item：

```json
{
  "arguments": {}
}
```

完成时再放完整 arguments。

---

## 8. web_search_call 处理策略

本项目当前策略：

```text
不走上游原生 web_search
web_search 自动重写为本地 tool_search
```

所以默认不应输出 `web_search_call`。

但为了兼容原生日志 replay，可以保留构造器：

```python
def make_web_search_call_item(...)
```

只在明确配置允许时启用：

```json
"responses_events": {
  "allow_web_search_call_event": false
}
```

默认：

```text
web_search -> tool_search_call
```

---

## 9. reasoning summary 事件还原

原生 Responses 可能出现 reasoning item 或 summary 事件。

建议支持以下事件：

```text
response.reasoning_summary_part.added
response.reasoning_summary_text.delta
response.reasoning_summary_text.done
response.reasoning_summary_part.done
```

### reasoning item

```json
{
  "id": "rs_xxx",
  "type": "reasoning",
  "status": "in_progress|completed",
  "summary": []
}
```

### summary part added

```json
{
  "type": "response.reasoning_summary_part.added",
  "item_id": "rs_xxx",
  "output_index": 0,
  "summary_index": 0,
  "part": {
    "type": "summary_text",
    "text": ""
  }
}
```

### summary text delta

```json
{
  "type": "response.reasoning_summary_text.delta",
  "item_id": "rs_xxx",
  "output_index": 0,
  "summary_index": 0,
  "delta": "..."
}
```

### summary text done

```json
{
  "type": "response.reasoning_summary_text.done",
  "item_id": "rs_xxx",
  "output_index": 0,
  "summary_index": 0,
  "text": "完整 reasoning summary"
}
```

### summary part done

```json
{
  "type": "response.reasoning_summary_part.done",
  "item_id": "rs_xxx",
  "output_index": 0,
  "summary_index": 0,
  "part": {
    "type": "summary_text",
    "text": "完整 reasoning summary"
  }
}
```

### 是否默认开启

建议默认关闭真实 reasoning summary 伪造：

```text
默认不编造 reasoning summary
只有当上游 emit_value 返回 summary 字段，或本地 compaction/tool-loop 生成明确摘要时才输出
```

避免 UI 误导。

---

## 10. 支持 annotation

`output_text` 可带：

```json
"annotations": []
```

后续本地 web fetch / local search 可以把来源塞成 annotation：

```json
{
  "type": "url_citation",
  "start_index": 10,
  "end_index": 20,
  "url": "https://example.com",
  "title": "Example"
}
```

计划：

```text
1. 先保留 annotations=[]
2. 本地 web 搜索工具实现后，把来源链接转为 url_citation
3. output_text.done 和 content_part.done 都带相同 annotations
```

---

## 11. 支持 response.incomplete

当本地代理遇到输出截断、max_output_tokens、上游中断但已有部分结果时，不应总是 failed。

新增：

```text
response.incomplete
```

response object：

```json
{
  "status": "incomplete",
  "incomplete_details": {
    "reason": "max_output_tokens|content_filter|tool_output_truncated|upstream_interrupted"
  }
}
```

事件流：

```text
response.created
response.in_progress
...partial output...
response.incomplete
[DONE]
```

适用场景：

```text
answer 被 max_answer_chars 截断
工具参数过长被截断
上游 SSE 提前断开但已得到可用部分结果
本地 compaction 输出超过限制
```

---

## 12. 支持 response.failed / error 细节

当前已有 failed，但可更完整。

建议结构：

```json
{
  "type": "response.failed",
  "response": {
    "id": "resp_local_xxx",
    "status": "failed",
    "error": {
      "code": "local_proxy_error|upstream_error|invalid_emit_value|tool_bridge_error",
      "message": "...",
      "type": "server_error|invalid_request_error"
    }
  }
}
```

然后再发：

```json
{
  "type": "error",
  "error": {
    "code": "...",
    "message": "...",
    "type": "..."
  }
}
```

---

## 13. 支持 response.cancelled

如果客户端断开、请求取消、本地超时：

```text
response.cancelled
```

response object：

```json
{
  "status": "cancelled",
  "error": null
}
```

在 FastAPI StreamingResponse 中可捕获：

```python
except asyncio.CancelledError:
    yield response.cancelled
    raise
```

注意：实际客户端断开时可能无法继续发送事件，但审计日志应记录 cancelled。

---

## 14. 多工具并行 output_index 保真

当前已支持多个 tool_calls 依次 output_index。

继续强化：

```text
每个 output item 必须有稳定 output_index
response.completed.output 数组顺序必须和 output_index 一致
parallel_tool_calls=true 时保留多个 item
parallel_tool_calls=false 时仍可顺序输出，但不应合并 item
```

测试：

```text
shell_command + apply_patch + tool_search 三个并行输出
断言 output_index = 0,1,2
completed.response.output 长度 = 3
```

---

## 15. item id / call id 稳定规则

建议统一：

```text
message id: msg_<response suffix>_<index>
function item id: item_<call_id suffix>
custom item id: ctc_<call_id suffix>
tool_search item id: tsc_<call_id suffix>
web_search item id: wsc_<call_id suffix>
reasoning item id: rs_<response suffix>_<index>
```

`call_id` 必须保持：

```text
call_xxx
```

不要每个事件重新生成。

---

## 16. SSE 编码严格化

确认 `app/sse.py`：

```text
event: <event_name>
data: <json>

```

要求：

```text
1. 每个事件都带 event: 行，除了 [DONE]
2. JSON 使用 ensure_ascii=False, separators=(',', ':')
3. 不输出空 data
4. 最后输出 data: [DONE]\n\n
5. sequence_number 从 0 递增，不跳号
```

新增测试：

```text
解析 stream 后 sequence_number == range(len(events))
最后一帧为 [DONE]
```

---

## 17. 非流式 JSON 与流式 completed 一致

`response_json()` 应与 `response.completed.response` 的结构一致。

也就是说：

```python
response_json(result, n) == response_object(..., status="completed", ...)
```

工具调用非流式响应也应包含：

```json
"output": [function_call/custom_tool_call/tool_search_call]
```

而不是只返回 assistant message。

---

## 18. Chat Completions 兼容事件

虽然 Codex 主要走 Responses，但 `/v1/chat/completions` 仍保留。

增强：

```text
chat.completion.chunk 第一帧带 role
中间 delta.content
最后 finish_reason=stop/tool_calls/length
usage 可选在最后帧或非流式响应中返回
```

如果桥接出工具调用，Chat 兼容层可先降级为文本提示或 OpenAI chat tool_calls 格式。

---

## 19. collect_bill015_result_from_events 增强

当前只关心：

```text
response.function_call_arguments.delta/done
response.completed
```

上游可能返回：

```text
response.output_text.delta
response.output_text.done
response.failed
response.incomplete
response.reasoning_summary_text.delta
```

增强解析：

```python
if response.output_text.delta:
    collect answer fallback
if response.output_text.done:
    answer = text
if response.failed/error:
    raise/record
if response.incomplete:
    record incomplete
```

虽然 BILL-015 强制上游 call `emit_value`，但增强后更抗异常。

---

## 20. emit_value schema 扩展可选字段

当前：

```json
{
  "mode": "answer|tool_call",
  "answer": "...",
  "tool_calls": []
}
```

可扩展但保持兼容：

```json
{
  "mode": "answer|tool_call|incomplete",
  "answer": "...",
  "tool_calls": [],
  "reasoning_summary": "",
  "annotations": [],
  "incomplete_reason": ""
}
```

注意：如果使用 strict schema，新增字段必须同步更新 parser 和 selftest。

建议分两阶段：

```text
阶段 1：只在 parser 容忍这些字段，但 schema 暂不要求
阶段 2：确认稳定后加入 schema
```

---

## 21. response_events.py 拆分

当前 `response_events.py` 可能继续变大，建议拆分：

```text
app/response_events.py              # 对外 generator 入口
app/event_models.py                 # response/item/event dict 构造
app/event_stream.py                 # sequence/SSE encode 生命周期
app/event_items.py                  # message/tool/reasoning item 事件生成
```

拆分后结构：

```python
responses_sse_generator()
  -> ResponsesEventStream
      -> emit_created()
      -> emit_message()
      -> emit_tool_calls()
      -> emit_reasoning_summary()
      -> emit_completed()
```

---

## 22. 原生日志事件样本库

建立样本目录：

```text
S:\hack\packyapi.com\bill015_local_proxy\proxy_evidence\native_event_samples
```

从 `logs_2.sqlite` 提取：

```text
message answer stream
function_call stream
custom_tool_call/apply_patch stream
tool_search_call stream
web_search_call stream
compaction stream
failed/error stream
incomplete stream（如果有）
reasoning summary stream（如果有）
```

每种保存：

```text
request.json
events.jsonl
completed.json
summary.md
```

---

## 23. 新增 native event replay 测试

新增脚本：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\replay_response_events.py
```

功能：

```text
1. 构造本地 Bill015Result(answer/tool_calls)
2. 生成 SSE
3. 解析 SSE
4. 验证事件顺序、字段、sequence_number、completed.output
5. 对照 native_event_samples 的 shape
```

检查项：

```text
- response.created 第一帧
- response.in_progress 第二帧
- sequence_number 连续
- output_item.added/done 成对
- content_part.added/done 成对
- function_call_arguments.done 存在完整 arguments
- custom_tool_call_input.done 存在完整 input
- tool_search_call 没有 function_call_arguments 事件
- response.completed 倒数第二帧
- [DONE] 最后一帧
```

---

## 24. selftest 增强

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\selftest.py
```

新增测试：

### A. message lifecycle

断言事件顺序：

```text
created -> in_progress -> output_item.added -> content_part.added -> delta -> done -> content_part.done -> output_item.done -> completed -> DONE
```

### B. function_call lifecycle

断言：

```text
function_call_arguments.delta 至少一条
function_call_arguments.done arguments 完整
completed.output[0].type == function_call
```

### C. custom_tool_call lifecycle

断言：

```text
custom_tool_call_input.delta
custom_tool_call_input.done
completed.output[0].type == custom_tool_call
```

### D. tool_search_call lifecycle

断言：

```text
tool_search_call item added/done
没有 function_call_arguments.done
execution == client
```

### E. parallel outputs

断言：

```text
output_index 连续
completed.output 长度正确
```

### F. usage fields

断言：

```text
completed.response.usage.input_tokens >= 0
completed.response.usage.output_tokens >= 0
input_tokens_details.cached_tokens 存在
```

### G. failed event

构造 HTTPException，断言：

```text
response.failed
error
DONE
```

---

## 25. 审计日志增强

在 `audit_from_result()` 里增加：

```json
{
  "event_fidelity": {
    "output_item_count": 2,
    "event_types": ["response.created", "response.output_item.added", "..."],
    "completed_status": "completed",
    "has_reasoning_summary": false,
    "has_annotations": false,
    "sequence_count": 12
  }
}
```

便于排查 UI 不显示、工具不执行、上下文用量不更新等问题。

---

## 26. 配置项

在 `config.local.example.json` 增加：

```json
"responses_events": {
  "fidelity_level": "native",
  "emit_reasoning_summary": false,
  "emit_annotations": true,
  "allow_web_search_call_event": false,
  "emit_incomplete_on_truncation": true,
  "strict_sequence_numbers": true,
  "chunk_size": 256
}
```

`Settings` 增加：

```python
responses_fidelity_level: str
responses_emit_reasoning_summary: bool
responses_emit_annotations: bool
responses_allow_web_search_call_event: bool
responses_emit_incomplete_on_truncation: bool
responses_strict_sequence_numbers: bool
responses_chunk_size: int
```

---

## 推荐实施顺序

一步到位按以下顺序实现：

```text
1. 新建 app/event_models.py
2. 重构 response object 构造，统一字段
3. 重构 message lifecycle 事件生成
4. 重构 function_call lifecycle 事件生成
5. 重构 custom_tool_call lifecycle 事件生成
6. 固化 tool_search_call 原生 item 生成
7. 保留但默认禁用 web_search_call
8. 增加 reasoning summary 事件构造器，默认不伪造
9. 增加 incomplete/failed/cancelled 事件构造器
10. 强化 sequence_number 连续性
11. 确保非流式 JSON 与 completed.response 一致
12. collect_bill015_result_from_events 增强异常 fallback
13. 增加 responses_events 配置项
14. 增强 selftest 生命周期断言
15. 新增 replay_response_events.py
16. 从 logs_2.sqlite 提取 native_event_samples
17. py_compile + selftest + replay
18. 重启代理
```

---

## 验收标准

完成后应满足：

```text
1. 普通文本回答事件生命周期完整
2. function_call / custom_tool_call / tool_search_call 事件结构接近原生
3. parallel tool calls 的 output_index 和 completed.output 正确
4. response.completed.response 与非流式 response_json 结构一致
5. usage 字段完整存在
6. sequence_number 从 0 连续递增
7. 失败时输出 response.failed + error + DONE
8. 截断时可输出 response.incomplete
9. web_search 默认不走上游，仍改写为 tool_search
10. selftest 和 replay_response_events 全部通过
```

---

## 最终效果

优化完成后，本地代理给 Codex Desktop 的 SSE/JSON 会更接近原生 Responses：

```text
- UI 更稳定显示输出、工具调用和 usage
- apply_patch/custom_tool_call 更接近原生日志
- tool_search/browser/node_repl 搜索链路更自然
- 多工具并行更稳定
- compaction/failed/incomplete 边界更清楚
- 后续新增本地 web 搜索、图片处理、工具闭环时不容易破坏事件格式
```

核心原则：

```text
只还原本地代理需要的原生事件外观和状态机。
不为了“像原生”而强行走上游正常 web_search 或额外计费路径。
```
