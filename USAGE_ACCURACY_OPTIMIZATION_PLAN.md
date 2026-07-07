# BILL-015 本地 Codex 代理：Usage 显示精度优化方案

本文档记录对 `S:\hack\packyapi.com\bill015_local_proxy` 的 usage / context 用量显示精度优化设计。目标是让 Codex 右下角上下文用量、请求完成后的 token usage、长上下文 compaction 触发判断更接近原生 Codex 表现，同时不引入额外上游正常计费。

## 目标

当前代理已经会合成非零 usage：

```json
{
  "input_tokens": 12345,
  "output_tokens": 123,
  "total_tokens": 12468,
  "input_tokens_details": {
    "cached_tokens": 11111
  }
}
```

但目前估算主要基于 `len(json)/4`，精度有限。

优化目标：

```text
1. 更准确估算 input_tokens
2. 更准确估算 output_tokens
3. 更合理估算 cached_tokens
4. 区分 text / tools / image / reasoning / tool output
5. compaction 前后 usage 更可信
6. Codex UI 右下角上下文显示更接近原生
7. 保持 usage 只是本地元数据，不触发额外上游计费
```

---

## 1. 新增 Token 估算模块

新增或重构文件：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\usage_estimator.py
```

替代或增强当前：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\token_usage.py
```

建议保留 `token_usage.py` 作为兼容层，然后把核心逻辑迁移到 `usage_estimator.py`。

核心数据结构：

```python
@dataclass
class UsageEstimate:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    text_tokens: int = 0
    tool_schema_tokens: int = 0
    tool_history_tokens: int = 0
    image_tokens: int = 0
    reasoning_tokens: int = 0
    metadata_tokens: int = 0
    estimate_method: str = "heuristic"
    confidence: str = "medium"
```

---

## 2. Tokenizer 优先级

实现分层估算策略。

### 优先级 A：tiktoken / o200k_base

如果本地安装 `tiktoken`：

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
len(enc.encode(text))
```

适合 OpenAI GPT-4o/GPT-5 系列近似。

### 优先级 B：transformers tokenizer，可选

如果以后有本地 tokenizer：

```python
AutoTokenizer.from_pretrained(...)
```

当前不建议强依赖，避免环境复杂。

### 优先级 C：字符启发式 fallback

没有 tokenizer 时：

```text
英文/代码：chars / 4
中文：chars / 1.6 到 chars / 2.2
混合文本：按字符类别加权
JSON / schema：chars / 3.4
base64 image：不按文本算，单独走图片估算
```

建议函数：

```python
def estimate_text_tokens(text: str) -> int:
    if tokenizer_available:
        return len(enc.encode(text))
    return weighted_char_estimate(text)
```

---

## 3. 按请求结构拆分 input usage

不要再对完整 request JSON 粗暴 `len/4`。

新增：

```python
def estimate_responses_input(body: dict[str, Any]) -> UsageEstimate:
    ...
```

按这些部分分别估算：

```text
instructions
input[] messages
tool call history
tool outputs
tools schema
reasoning encrypted_content / summary
metadata / client_metadata
images / files
```

建议拆分函数：

```python
def estimate_instructions_tokens(instructions: Any) -> int

def estimate_input_items_tokens(input_items: Any) -> tuple[int, int]
# 返回普通文本 tokens 和 tool_history tokens

def estimate_tools_schema_tokens(tools: Any) -> int

def estimate_reasoning_tokens(input_items: Any, include: Any) -> int

def estimate_metadata_tokens(metadata: Any, client_metadata: Any) -> int

def estimate_image_tokens(input_items: Any) -> int
```

---

## 4. 原生 Responses input[] 项目计数规则

对 `input[]` 里的类型分类处理。

### message

```json
{"type":"message","role":"user","content":[...]}
```

计入：

```text
role overhead + content text tokens
```

建议 overhead：

```text
每个 message 固定 +4 到 +8 tokens
每个 content part 固定 +2 tokens
```

### function_call / custom_tool_call

计入工具历史：

```text
工具名
call_id
arguments/input
结构 overhead
```

建议：

```python
serialize compact JSON 后 tokenizer 计数 + 8
```

### function_call_output / custom_tool_call_output

工具输出通常非常大，需要更准确计入。

规则：

```text
如果 output 很短：完整计数
如果 output 很长：按实际会进入上游 prompt 的裁剪后内容计数
```

注意：如果 `tool_history.py` 已经生成 `latest_tool_summary` 并替代部分原始输出给上游，则 usage 应按实际发给上游的内容估算，不应重复按完整原生 input[] 全量估算。

### tool_search_output

计入：

```text
暴露的工具 schema
namespace description
tool names
parameters
```

但如果代理已经把 deferred tools 合入 catalog，需要避免重复算两遍。

### reasoning

原生日志里经常有：

```json
{"type":"reasoning","summary":[],"encrypted_content":"..."}
```

建议：

```text
encrypted_content 不按明文全部计入本地上下文，默认只给很小 overhead
summary 若存在则计入 summary 文本 tokens
```

理由：本地代理通常不会把 encrypted_content 明文发给上游模型。

---

## 5. tools schema token 估算

Codex 每轮会携带完整 `tools`。

原生上下文显示通常会受工具 schema 影响，所以应计入。

实现：

```python
def normalize_tools_for_usage(tools: Any) -> str:
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
```

计数：

```python
tool_schema_tokens = estimate_text_tokens(normalize_tools_for_usage(tools))
```

但需要限制：

```text
如果代理实际发给上游的是压缩后的 tools_catalog，就按 tools_catalog 算
如果只是给 Codex UI 显示，可以按原生 tools 算
```

建议保留两套：

```python
native_context_tokens
upstream_payload_tokens
```

其中：

```text
native_context_tokens：用于 Codex UI context 显示
upstream_payload_tokens：用于审计本地上游实际 prompt 大小
```

---

## 6. 图片 token 估算

解决图片导致 usage 不准和 413/431 问题。

图片来源：

```text
input_image
image_url: data:image/...;base64,...
image_url: http(s)://...
local file path marker
```

估算策略：

### data URL 图片

解析 base64 后用 PIL 获取尺寸：

```python
from PIL import Image
Image.open(BytesIO(raw)).size
```

然后按视觉模型粗估：

```text
low detail：约 85 tokens
high detail：85 + 170 * tiles
tile = ceil(width/512) * ceil(height/512)
```

可实现：

```python
def estimate_image_tokens_by_size(width: int, height: int, detail: str = "high") -> int:
    if detail == "low":
        return 85
    tiles = ceil(width / 512) * ceil(height / 512)
    return 85 + 170 * tiles
```

### 无法读取尺寸

fallback：

```text
按 base64 原始字节大小估计
小图 300
中图 800
大图 1500+
```

---

## 7. cached_tokens 估算

原生 response 里常见：

```json
"input_tokens_details": {
  "cached_tokens": 205312
}
```

本地代理无法知道真实服务端 prompt cache 命中，但可以模拟 Codex UI 需要的“上下文大部分已缓存”效果。

建议缓存估算：

```python
def estimate_cached_tokens(n: NormalizedRequest, estimate: UsageEstimate) -> int:
    if estimate.input_tokens < 4096:
        return 0
    if n.prompt_cache_key:
        return int(estimate.input_tokens * 0.85)
    if n.client_metadata and n.client_metadata.get("thread_id"):
        return int(estimate.input_tokens * 0.75)
    return int(estimate.input_tokens * 0.50)
```

更进一步：维护本地 thread cache 状态。

新增文件或 state：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\usage_cache.py
```

记录：

```python
thread_id -> last_input_fingerprint
thread_id -> last_estimated_input_tokens
thread_id -> stable_prefix_tokens
```

然后：

```text
同 thread 且 input 前缀高度相似：cached_tokens = 上轮 input_tokens 的 80%-95%
新 thread：cached_tokens = 0 或较低
compaction 后：cached_tokens 重置或降低
```

---

## 8. output_tokens 更准确

当前 output 可能是：

```text
最终 answer
function_call arguments
custom_tool_call input
tool_search_call arguments
```

实现：

```python
def estimate_response_output_tokens(answer: str, calls: list[BridgeToolCall]) -> int:
    if calls:
        return sum(estimate_tool_call_output_tokens(call) for call in calls)
    return estimate_text_tokens(answer)
```

工具调用 output 估算：

```text
function_call：name + arguments + structure overhead
custom_tool_call：name + input + structure overhead
tool_search_call：arguments + structure overhead
```

建议 overhead：

```python
FUNCTION_CALL_OVERHEAD = 12
CUSTOM_TOOL_CALL_OVERHEAD = 10
MESSAGE_OVERHEAD = 8
```

---

## 9. Responses usage 字段完整还原

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\response_events.py
```

返回 usage 时包含更多原生兼容字段：

```json
{
  "input_tokens": 123,
  "output_tokens": 45,
  "total_tokens": 168,
  "input_tokens_details": {
    "cached_tokens": 100
  },
  "output_tokens_details": {
    "reasoning_tokens": 0
  }
}
```

注意：不要编造太多上游不兼容字段。建议默认只加：

```text
input_tokens
output_tokens
total_tokens
input_tokens_details.cached_tokens
output_tokens_details.reasoning_tokens
```

---

## 10. Chat Completions usage 映射

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\response_events.py
```

Chat 兼容字段：

```json
{
  "prompt_tokens": input_tokens,
  "completion_tokens": output_tokens,
  "total_tokens": total_tokens,
  "prompt_tokens_details": {
    "cached_tokens": cached_tokens
  },
  "completion_tokens_details": {
    "reasoning_tokens": reasoning_tokens
  }
}
```

---

## 11. compaction usage 处理

compaction 请求特点：

```json
"request_kind":"compaction"
"tools": []
"parallel_tool_calls": false
"text": {"verbosity":"low"}
```

估算规则：

```text
input_tokens：按 compaction 原始 input[] 或压缩 transcript 估计
output_tokens：按 handoff summary 估计
cached_tokens：通常较高，但 compaction 后下一轮应重置部分上下文缓存
```

建议：

```python
if n.is_compaction:
    cached_tokens = int(input_tokens * 0.90)
```

同时审计日志记录：

```json
{
  "request_kind": "compaction",
  "usage_estimate": {
    "input_tokens": ...,
    "output_tokens": ...,
    "cached_tokens": ...
  }
}
```

---

## 12. 根据原生日志校准

已有样本目录：

```text
S:\hack\packyapi.com\bill015_local_proxy\proxy_evidence\native_usage_compaction
S:\hack\packyapi.com\bill015_local_proxy\proxy_evidence\native_long_chat\extracted
```

新增脚本：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\calibrate_usage_estimator.py
```

功能：

```text
1. 读取原生 request_*.json / extracted/*.json
2. 读取对应 completed_*.json 或 logs 中 response.completed usage
3. 对每个请求运行 estimate_responses_input()
4. 输出 estimated vs native 差异
5. 计算误差百分比
6. 给出推荐 heuristic 系数
```

输出示例：

```text
request_5079554_turn.json
native input_tokens: 205688
estimated input_tokens: 198420
error: -3.53%

compaction_request_2725280.json
native input_tokens: 276340
estimated input_tokens: 289120
error: +4.62%
```

目标：

```text
普通 turn 误差控制在 ±10%
compaction 误差控制在 ±15%
图片请求单独统计
```

---

## 13. 本地 usage 审计日志

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\proxy.py
```

在 `audit_from_result()` 增加：

```json
{
  "usage_estimate": {
    "input_tokens": 123,
    "output_tokens": 45,
    "total_tokens": 168,
    "cached_tokens": 100,
    "text_tokens": 80,
    "tool_schema_tokens": 30,
    "tool_history_tokens": 10,
    "image_tokens": 0,
    "estimate_method": "tiktoken:o200k_base",
    "confidence": "high"
  }
}
```

方便后续排查右下角显示异常。

---

## 14. 请求 normalize 阶段接入

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\proxy.py
```

在 `normalize_responses_request()` 中：

```python
usage_estimate = estimate_responses_usage_from_body(body, upstream_payload_preview=None)
```

然后：

```python
return NormalizedRequest(
    ...
    estimated_input_tokens=usage_estimate.input_tokens,
    usage_estimate=usage_estimate,
)
```

`NormalizedRequest` 增加：

```python
usage_estimate: Any = None
```

---

## 15. response_events 接入

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\app\response_events.py
```

替换当前：

```python
response_usage(n, answer=answer)
response_usage(n, calls=calls)
chat_usage(n, answer=result.answer)
```

为：

```python
build_response_usage(n, answer=answer, calls=calls)
build_chat_usage(n, answer=result.answer)
```

其中内部使用 `n.usage_estimate` 的 input 部分，再动态估算 output。

---

## 16. dry-run 输出 usage 详情

`dry_run_response()` 中增加：

```json
"usage_estimate": {
  "input_tokens": ...,
  "tool_schema_tokens": ...,
  "image_tokens": ...,
  "cached_tokens": ...,
  "method": "..."
}
```

这样不用真实请求也能检查 usage 是否合理。

---

## 17. 自测覆盖

修改：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\selftest.py
```

新增测试：

### A. 普通文本 usage

```python
n = normalize_responses_request({"input":"hello world"})
assert n.estimated_input_tokens > 0
```

### B. 中文文本 usage

```python
中文长文本不能明显低估
```

### C. tools schema usage

```python
带 tools 的请求 input_tokens > 不带 tools 的请求
```

### D. tool output usage

```python
function_call_output 很长时 tool_history_tokens 增加
```

### E. image usage

```python
data:image/png;base64,... 会产生 image_tokens
```

### F. compaction usage

```python
request_kind=compaction 时 cached_tokens / input_tokens 比例较高
```

### G. stream completed usage

断言：

```text
response.completed.response.usage.input_tokens > 0
response.completed.response.usage.output_tokens >= 0
response.completed.response.usage.input_tokens_details.cached_tokens 存在
```

---

## 18. replay 原生日志测试

新增或增强：

```text
S:\hack\packyapi.com\bill015_local_proxy\scripts\replay_native_logs.py
```

检查：

```text
1. 所有 native request 都能 normalize
2. usage_estimate 不为 0
3. compaction usage 合理
4. tools schema 大请求 usage 明显大于普通请求
5. 不因超大 input/tool output 抛异常
```

---

## 19. 性能优化

tokenizer 可能较慢，需要缓存。

新增：

```python
@lru_cache(maxsize=4096)
def estimate_text_tokens_cached(text_hash: str, text: str) -> int:
    ...
```

更简单：

```python
TOKEN_COUNT_CACHE: dict[str, int]
key = sha256(text.encode()).hexdigest()
```

对大文本：

```text
超过 200k chars 的文本按分块计数
超过 2MB 的输出先按裁剪策略处理再计数
```

---

## 20. 配置项

在 `config.local.example.json` 增加：

```json
"usage": {
  "estimator": "auto",
  "prefer_tiktoken": true,
  "include_tools_schema": true,
  "include_images": true,
  "cache_ratio_default": 0.85,
  "max_text_for_exact_tokenize": 500000,
  "audit_breakdown": true
}
```

`Settings` 增加字段：

```python
usage_estimator: str
usage_prefer_tiktoken: bool
usage_include_tools_schema: bool
usage_include_images: bool
usage_cache_ratio_default: float
usage_max_text_for_exact_tokenize: int
usage_audit_breakdown: bool
```

---

## 推荐实施顺序

一步到位按以下顺序实现：

```text
1. 新建 app/usage_estimator.py
2. 在 usage_estimator.py 中实现 tokenizer fallback
3. 实现 text/tools/input/tool-output/image/reasoning/metadata 分项估算
4. NormalizedRequest 增加 usage_estimate 字段
5. normalize_responses_request() 接入 usage_estimate
6. response_events.py 使用 usage_estimate 构造 Responses usage
7. chat_json/chat_sse 使用 usage_estimate 映射 Chat usage
8. dry_run_response 输出 usage_estimate
9. audit_from_result 写入 usage_estimate breakdown
10. config.local.example.json 增加 usage 配置
11. selftest.py 增加 usage 精度测试
12. 新增 calibrate_usage_estimator.py 对比原生日志
13. 新增/增强 replay_native_logs.py 防回归
14. py_compile + selftest
15. 重启代理
```

---

## 验收标准

完成后应满足：

```text
1. 任意 Responses 请求 completed usage 非 0
2. 带 tools 的请求 input_tokens 明显高于纯文本
3. 长 tool output 会反映在 usage 中，但不会无限膨胀
4. 图片请求会产生 image_tokens
5. compaction 请求 cached_tokens 比例更接近原生
6. dry-run 能看到完整 usage breakdown
7. 原生日志 replay 不崩
8. 与原生样本误差普通 turn 控制在 ±10%-15%
9. 不提交/打印真实 API key
10. 不改变 BILL-015 低/零上游用量路径
```

---

## 最终效果

优化完成后，Codex UI 中的上下文用量会更可信：

```text
- 普通文本：按 tokenizer 或加权字符估算
- 中文/代码/JSON：分类加权
- 工具 schema：单独计入
- 工具历史：按实际发给上游的裁剪内容计入
- 图片：按尺寸估算视觉 token
- cached_tokens：按 thread/cache/compaction 估算
- output_tokens：按 answer 或 tool_call 精确估算
```

这会让本地代理的 usage 显示更接近原生 Codex，同时保持 usage 仅作为本地合成元数据，不引入额外上游正常计费。
