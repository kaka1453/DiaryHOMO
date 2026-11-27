# 阶段二任务 C · 本地无状态推理服务 v1

`lm_trainer.chat.inference` 是独立的 Python v1 契约；`lm_trainer.serve.app` 是可选 FastAPI 适配。旧版 [Task C 有状态 API](TASK_C_CONTRACT.md)、Gradio 与 eval 的入口、历史提交规则和数据格式不变。服务端使用既有 HF 单设备运行时，不转发远程 API，不接受客户端指定路径、权重、角色、Jinja、停止符或 ACL。

## Python / 任务 D 接入

```python
from lm_trainer.chat import ChatRuntime, load_registry, load_roles
from lm_trainer.chat.inference import InferenceRequestV1, InferenceServiceV1, ScopeBindingV1

runtime = ChatRuntime(load_registry("models.json"), load_roles("roles.json"))
service = InferenceServiceV1(runtime, "service-model", (ScopeBindingV1("desk.mock", "support"),))
service.start()  # 校验、加载、预热；未就绪时不接收生成
try:
    result = service.generate(InferenceRequestV1(
        "service-model", "desk.mock", messages=(("user", "虚构光源无法点亮。"),),
    ))
    if result.error_code is None:
        reply = result.text
finally:
    service.close()
```

- Python 是**可信调用边界**：调用方先绑定授权 scope，再提交 `InferenceRequestV1(model_id, scope, messages=… | observation=…, settings=…)`。`messages` 必须完整、按 `user, assistant, …, user` 交替；结构化输入由任务 A 的 `parse_catalog` / `observe` 产生。配置 `ScopeBindingV1(scope, role_id, "observation", conversation_id, target_person, conversation_mode)` 才可使用该结构化输入。角色、对话 ID、`direct`/`group` 类型及目标客服身份全部由服务端绑定；角色 namespace 来自 `RoleConfig`。这些校验不独立鉴定消息作者或上游来源，群聊 scope 只应分配给能验证事件归属、时间、观察范围和权限的可信网关。服务不是 IM 协议接入层。
- `generate()` 同步返回 `InferenceResultV1`；`stream()` 返回单消费者 `InferenceHandleV1`，`request_id` 在首个事件之前可用。`next_event(timeout)` 超过一次队列等待抛 `queue.Empty`（不是生成超时）；遍历得到零或多个 `delta`，然后恰一个 `done` / `cancel` / `error`（携带 `result`）。`handle.cancel()` / `close()` 只请求取消，不强制 join；`service.close()` 停止接收、取消并等待在途工作、释放模型。`submit()` 是 transport 预约接口，`owner` 必须由可信适配器指定；`service.cancel(request_id, owner=…)` 用于同一所有者取消。入场失败抛稳定 `ChatError`，已入场失败返回终态。finish 为 `eos/stop/length/cancel/error`；失败的 `text` 为空，已经接收的流式增量只是暂定结果。服务不读取或写入 legacy session，不保留请求间的历史或 KV；同一 runtime 在服务持有模型期间，旧入口仍可用同一模型，切换到其他模型或手动卸载返回 `DEVICE_BUSY`。
- `ContextBuilder.build_inference_v1(...)` 先用真实 tokenizer 检查**完整**历史/观察、可信 system、原生模板和预留生成 token 是否同时适配。超限返回 `CONTEXT_LIMIT`，不裁剪输入/目标；随后尝试独立标界的 `reference_section`，仅丢弃不安全、模板不能对齐、超数量或超预算的可选卡。默认最多 3 张、400 个增量 token，并受 registry、模型和 tokenizer 限制。组聊唯一事实源是任务 A 的观察渲染和训练同源前缀编码；模板可为 token 对齐做内部对照，但模型输入中不会外包第二层模板。无卡且训练与服务采用同一 system/policy 时，输入前缀必须逐 token 相同；注入卡会有意改变 system 段，不能声称与无卡训练前缀等同。旧 `build` / `build_conversation` 签名、裁剪及无卡行为仍保留。
- 任务 D 注入 `InferenceMemoryProviderV1.select(AuthorizedContextV1) -> (tuple[ContextCard, ...], tuple[CardAudit, ...])`；无 provider 时不检索、不注入。上下文含服务器绑定的 model、scope、role、request_id 和已观察文本（不含完整 source catalog、未观察事件或被保护正文）；Python 集成无须修改 HTTP 或模型后端。Provider 负责查询权限、最新版本、撤销和授权 scope，应限时并且不得重入推理服务；接口当前同步、占用串行执行槽。返回的已授权候选仍要通过 namespace、精确 scope、active、read、唯一最新版本检查：卡须为 `scope="session_only", session_id=<该授权 scope>`，这里复用字段名、不创建会话；只带 `role:<id>` 不能证明跨群可读。Provider 可为确实允许共享的资料返回授权到当前 scope 的卡，不接受客户端的 ACL 或卡正文。HTTP 不公开卡或审计内容，Python 返回卡 ID/版本/原因；日志只保留版本与固定原因，不记录卡 ID、来源指针或正文。不建设 RAG/长期存储/全局撤销库。
- 角色头、模板控制 token、伪造群聊归因、保留协议/资料界标与危险格式走任务 A 和既有控制 token 防护。模型输出由现有 `VisibleStreamer` 及原生 EOS/stop 检查过滤；客户端无权开工具或 reasoning。此防线拒绝结构性冒充，不承诺检测所有自然语言诱导；资料必须视为非指令，不能依靠模型输出执行有副作用的操作。

## HTTP 兼容矩阵（OpenAI 形状，不是完整协议）

| 接口或字段 | v1 行为 | 拒绝 |
|---|---|---|
| `Authorization: Bearer …` **或** `X-API-Key: …` | 所有路由要求一个且仅一个凭据。`allowed_models` 与 `allowed_scopes` 分别绑定；生成必须携带唯一 `X-Inference-Scope`。 | 无效/重复/混用 `401 AUTH_REQUIRED`；越权 `403 FORBIDDEN` |
| `GET /v1/models` | 已就绪时返回该 key 可见的唯一 model ID（或 `data: []`）；无路径/未注册模型扫描。 | 未就绪/加载问题 `503` |
| `POST /v1/chat/completions` | UTF-8 `application/json`；`model` 必须是固定注册 ID。默认正文上限 256 KiB，同时校验声明长度与 ASGI 实收长度；读取正文默认限时 10 秒。 | `404 UNKNOWN_MODEL`、`413 PAYLOAD_TOO_LARGE`、`504 TIMEOUT`；无效 JSON、非有限数字、重复 JSON 字段 `400 INVALID_INPUT` |
| `messages` | `{role,content}` 原始字符串，user/assistant 交替并以 user 结尾；不接受已经渲染的 prompt、token ID 或多模态 content parts。 | system/developer 一律 `403 FORBIDDEN`；其它不支持角色/字段 `400 UNSUPPORTED_FIELD`；非法结构 `400 INVALID_INPUT`；标记和超限 `422 UNSAFE_CONTENT/CONTEXT_LIMIT` |
| `observation`（扩展） | 代替 `messages`：`{"source": <任务 A source catalog>, "observed_ids": [源事件 ID, …]}`。只接收一个与 scope 匹配的 conversation，其 mode 和目标身份固定；Task A 会独立验证因果投影。`display_name`/`fixture_only`/样本训练标注不参与授权或角色渲染。 | 混用或非法 schema `400 INVALID_INPUT`；额外字段 `400 UNSUPPORTED_FIELD`（Task A 的关系字段按其自身契约校验）；scope/模式不匹配 `403 FORBIDDEN` |
| `stream`, `n` | `stream` 默认 false，只接受布尔；`n` 只能为整数 `1`，只有一个 choice。 | 非法类型或模式 `400` |
| `max_completion_tokens` / `max_tokens` | 同义、二选一；正整数，不超过服务上限（默认 512）。`temperature` 0（贪心）或 `(0,2]`（采样）；`top_p` 在 `(0,1]`、`seed` 是 `[0,2^63)` 整数；单独提供后二者时开启采样，显式温度 0 则不可组合。未提供的使用服务器配置。 | `400 INVALID_INPUT` |
| `tools/tool_choice/functions/function_call`、`response_format`、`reasoning*`、`stop`、`stream_options`、`logprobs`、`user` 等其他字段 | 不支持、不静默忽略。不接受请求级身份、自签卡、模板、目录或外部模型。 | `400 UNSUPPORTED_FIELD` |
| `DELETE /v1/requests/{request_id}`（扩展） | 原 credential principal 可取消自己的在途请求；`202` 表示已接受取消，不保证 kernel 已退出。流客户端也可主动断连。 | 缺失、已结束、或其他 principal 的请求统一 `404 REQUEST_NOT_FOUND` |

非流式成功返回 `chat.completion`，只有可见助手正文、标准 `finish_reason=stop/length`、`usage` 与 `metrics`。流式为 `text/event-stream`，以 assistant role 开始，真实 HF 增量使用 `data: {...}\n\n`，最后发送 finish+usage+metrics 与 `data: [DONE]\n\n`。无真流支持的测试替身仅在成功终态缓冲交付正文，不伪造 delta 时序或 TTFT。失败若发生在 SSE 响应头发送后，HTTP 状态仍为 200，发一个带 `{error:{code,message,request_id}}` 的错误数据帧并关闭，**无 `[DONE]`**；客户端必须丢弃先前增量。非流式用相应 4xx/5xx，错误同形且不含底层异常。每次请求服务器生成 `X-Request-ID`，忽略客户端自带同名 header；响应 `Cache-Control: no-store`。

**调度和指标。** 单进程固定一个模型，FIFO，运行槽 1，默认另有 4 个等待槽，满载 `429 QUEUE_FULL`。默认生成截止 60 秒（包含排队、可能重加载、上下文和生成）；HTTP 可先返回 `504 TIMEOUT`，物理执行仍占槽直至检查点。排队取消、流式/非流式断连都会清理；生成只在 token 边界合作停止，不抢占加载、provider 或 CUDA kernel。显式取消非流式在结束后为 `409 CANCELLED`。`503 MODEL_UNAVAILABLE/ARTIFACT_MISMATCH/TEMPLATE_UNSUPPORTED/OUT_OF_MEMORY/LOAD_FAILED/DEVICE_BUSY/SERVICE_UNAVAILABLE`，其余后端/输出错误 `500 GENERATION_FAILED/UNSAFE_OUTPUT`；不回落到别的模型。失败后可用同一固定模型重载，不能换模型。

`usage.prompt_tokens/completion_tokens/total_tokens` 用实际输入和生成步数（终止 token 计入）；`metrics` 另有 `input_tokens/output_tokens/ttft_ms/tokens_per_second/latency_ms/peak_vram_mb/queue_wait_ms/cancellation_count`。TTFT 是入队至首个实际可见增量，非流式或替身为 `null`；吞吐是生成阶段（非整体请求），峰值为生成阶段 PyTorch 分配 MiB，CPU/mock 为 `null`；排队等待截止到拿到执行槽，若未进入则截止到退出队列；`cancellation_count` 是单请求 0/1，取消/超时均为 1。HTTP 早返回时未取得指标为 `null`，最终审计在线程退出时记录可测值；审计 `status` 表示该请求的逻辑终态，不是已发送的 SSE HTTP 头状态。CLI 默认只把白名单审计字段以结构化日志输出，不启用 access log；无请求/回复原文、资料原文、凭据或私有路径。

## 启动、停止与复跑

可选服务模块复用现有 `fastapi/starlette/uvicorn`（当前 `.venv` 中已存在）；测试用 `httpx`。不要求修改训练依赖/锁文件。使用可信 `models.json`（`lm_trainer.chat-models.v1`）和 `roles.json`（`lm_trainer.chat-roles.v1`）；`service.json` 的路径相对自身，凭据仅填环境变量名。示例是虚构配置，`service-model` 未代表实际可用产物：

```json
{
  "schema": "lm_trainer.inference-service.v1",
  "registry": "models.json",
  "roles": "roles.json",
  "model_id": "service-model",
  "scopes": [
    {"scope":"desk.mock","role_id":"support"},
    {"scope":"fixture.acceptance.group","role_id":"support","input_kind":"observation","conversation_id":"group","conversation_mode":"group","target_person":"support_agent"}
  ],
  "credentials": [{"token_env":"INFERENCE_API_KEY","principal_id":"fictional-workbench","allowed_models":["service-model"],"allowed_scopes":["desk.mock","fixture.acceptance.group"]}],
  "generation": {"max_new_tokens":128,"do_sample":false},
  "max_queued":4,
  "max_output_tokens":512,
  "timeout_seconds":60,
  "max_body_bytes":262144,
  "body_timeout_seconds":10
}
```

Registry 走任务 B `inference_adapter`、manifest SHA、inventory、base fingerprint、tokenizer hash、模板/量化/dtype 校验；结构化 scope 还要求 manifest renderer/template/stop identity 相符，正式产物的 pinned policy system 必须精确匹配可信 role。部署使用不可变的已核验文件，校验/运行期间不得替换。用 `--test-mode` 才能同时打开 registry 和 service 测试允许位，需产物另带可信 `fixture_only:true`；不得将 fixture 模式用于生产。生产正式入口不接受 base-only 代替 adapter。单机默认/CLI 强制 `127.0.0.1`，单 worker，不信 proxy header；容器内该地址是容器自身，宿主机端口映射不会让只绑定容器回环的服务自动可达。自定义 ASGI 装配应保留回环/进程/日志/鉴权边界并关闭 access log；本轮不部署代理、不对外监听。

```bash
read -rs -p '本地高熵服务 key: ' INFERENCE_API_KEY; printf '\n'; export INFERENCE_API_KEY
HF_HUB_OFFLINE=1 .venv/bin/python -B -m lm_trainer.serve --config /path/to/service.json --check
# 只有正式模型验收/授权后才打开本地端口；Ctrl+C 或 SIGTERM 优雅停止。
HF_HUB_OFFLINE=1 .venv/bin/python -B -m lm_trainer.serve --config /path/to/service.json --port 8000
```

```bash
curl -sS -N http://127.0.0.1:8000/v1/chat/completions \
  -H @<(printf 'Authorization: Bearer %s\n' "$INFERENCE_API_KEY") \
  -H 'X-Inference-Scope: desk.mock' -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"model":"service-model","messages":[{"role":"user","content":"虚构的 SABLE 工位灯不亮，先检查什么？"}],"max_completion_tokens":64,"stream":true}
JSON
```

## 推理后端评估与待办

| 后端 | 环境/本次状态 | 使用门槛与结论 |
|---|---|---|
| HF 单设备 / `HFBackend` | reference；合成 tokenizer + 真实 HF 小型 CPU fixture 已运行，另测本地真实 tokenizer，不加载其模型 | 复用原生 LoRA、模板、stop、可见流和指标。本次不另写引擎。正式模型 `MODEL_ARTIFACT_PENDING` |
| vLLM | `NOT INSTALLED / NOT RUN`（包、CLI 均未发现） | 原生服务的 OpenAI 形状不等于本契约；先独立实测 adapter、量化支持、逐位输入、显式 stop/可见流/实际指标、取消权限，现无兼容结论 |
| llama.cpp server | `NOT INSTALLED / NOT RUN`（Python 包、`llama-server` 未发现） | 现有 safetensors/PEFT pin 不可假定直用；如需 GGUF/量化，另验模板/输出/精度/显存及保留原权重回退，不在本轮转换 |
| Ollama | `NOT INSTALLED / NOT RUN`（CLI 未发现） | 导入/模板与 LoRA pin、token/停止/取消指标需独立实证；无兼容结论，不 merge、不下载 |

正式产物选定并获授权后，用与 CPU 证据同模块的正式入口回放 Task A 的合成 group（上例 acceptance scope 必须保留该 ID/mode/target）；只走 ASGI 内存传输，不监听端口。`--check` 先验证加载。生产消息、GPU 并发/显存/吞吐压测仍需独立批准和可复现测试，本轮 `NOT RUN`。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -B -m lm_trainer.serve --config /path/to/service.json --check
HF_HUB_OFFLINE=1 INFERENCE_FORMAL_CONFIG=/path/to/service.json \
  INFERENCE_FORMAL_SCOPE=fixture.acceptance.group \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  lm_trainer/tests/test_inference_service.py -k formal_model_acceptance
```
