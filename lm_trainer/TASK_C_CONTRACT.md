# Task C · Headless chat API v1

从 `lm_trainer.chat` 导入；仅构造配置、导入包或运行 `MockChatRuntime` 不会导入 Torch、Transformers、PEFT 或 Unsloth。默认纯内存、离线、无 UI。调用方（Task D）只依赖 `ChatRuntimeProtocol` 和公开的 dataclass，不访问 HF 对象。

| API | v1 语义 |
|---|---|
| `list_models() -> list[ModelInfo]` | 本地预检（校验文件、指纹、tokenizer/模板，不加载权重到 GPU）；`available=false` 附稳定 `reason`。`supports_streaming` 表示真实 token 流能力。 |
| `create_session(model_id, role_id, settings=None) -> SessionInfo` | 独立角色/模型配置；懒加载模型。`get_session(id)` 返回不可变快照和 `active_request_id`；历史只有成对的真实 `Message(user, assistant)`，各带会话、轮次、时间、role version、模型与模板 hash。 |
| `send(id, user_text, memory_cards=None) -> ChatResult` | 同步生成。已接收请求的失败返回 `finish_reason=error`、`error_code`；提交前的参数/会话错误抛 `ChatError`。 |
| `stream(id, user_text, memory_cards=None) -> ChatStream` | 单消费者迭代器，立即预约并启动请求，`stream.request_id` 在首个 delta 前可用。`delta` 是已验证的可见增量；随后恰一个 `done` / `cancel` / `error`，携带 `result`。提前退出请用 `with` 或 `close()`（请求取消并等待退出）。尚未提交的 delta 可撤回；在 error/cancel 后丢弃。Mock 标记无流能力，只发终态。 |
| `cancel(id, request_id) -> CancelResult` | 只取消该会话当前且未提交的请求；`accepted=false` 表示 ID 不匹配或已结束。已排队请求不必等待前一个生成；进行中的请求在 token 边界停止，不能抢占当前 CUDA kernel/加载。 |
| `clear_session(id)` / `switch_model(id, model_id, *, new_session=True)` | 忙会话的修改及重复发送拒绝 `SESSION_BUSY`。切换实际激活目标模型；默认返回新会话，旧会话保持原绑定和历史。明确指定 `new_session=False` 时变更模型即清空原会话历史。更换角色应 `create_session`。 |

**结果。** `ChatResult` 关联 `request_id/session_id/turn_id/model_id/role_id`，带 `template_hash/fixture_only`、原格式可见 `text`、`finish_reason=eos/length/stop/cancel/error`、`committed`、`card_audit`（仅 ID、版本、原因）、`dropped_turns`。仅完整、非空白的成功回复原子提交用户/助手对；cancel/error 的 `text` 为空。`input_tokens` 含原生模板开销；`output_tokens` 是实际解码步数（含采样到的终止 token）；`ttft_ms` 从接收到首个可见 delta，仅真流式；`tokens_per_second` 按生成阶段耗时；`latency_ms` 含排队、可能的加载和预热。`peak_vram_mb` 是生成阶段 PyTorch allocated 峰值（单位 MiB）；未测项 `None`。`ActivationInfo` 单列新加载的 `load_ms/warmup_ms/resident_vram_mb`；已加载模型可由 `active_model()` 查询，释放用 `unload()`。

**配置和信任边界。** `load_registry` 配置使用 `lm_trainer.chat-models.v1`；`load_roles` 使用 `lm_trainer.chat-roles.v1`。注册表、模型目录、模板 hash、角色文件和候选卡片来自可信应用层，不接受终端用户自签 ACL、目录或模板。部署使用不可变本地 safetensors、经审阅的原生模板 SHA 和固定的 Task B manifest SHA；加载前要求 base/tokenizer/policy/LoRA/量化及 dtype 匹配。fixture 双显式开关（条目标签 + `allow_fixture=True`），禁止用于生产。支持 HF 单设备；不支持远程代码、预量化基座、多卡/CPU offload、工具或无法明确识别可见回复的模板。切换时卸载重载，不 merge，进程内每 GPU 仅一个 runtime 持有活动模型；需由部署方隔离其他进程。`do_sample` 的显式 `seed` 使用请求私有 RNG。

**资料与错误。** `NoMemoryProvider` 默认拒绝注入。`ManualCardProvider` 按绑定 namespace、`role:<id>` 或精确 `session_id` 的 `session_only`、最新唯一版本、active 与严格 read ACL 筛选；`supersedes` 指同卡 ID 的旧整数版本，调用方必须提供完整当前候选快照（没有全局撤销库）。资料序列化在系统消息内独立标界，非角色/历史消息、不赋指令权；拒绝保留标记。默认最多 3 卡、400 个增量 token，先拒绝超预算卡、再裁剪最早完整轮，绝不裁剪当前问题。默认不写盘；显式 `save_session(id, dir)` 要求调用者所有的已有 `0700` 目录并新建 `0600` 文件。公共错误不携带原文/绝对路径：`INVALID_CONFIG/INVALID_INPUT/UNKNOWN_MODEL/UNKNOWN_ROLE/UNKNOWN_SESSION/SESSION_BUSY/MODEL_UNAVAILABLE/ARTIFACT_MISMATCH/TEMPLATE_UNSUPPORTED/CONTEXT_LIMIT/UNSAFE_CONTENT/DEVICE_BUSY/OUT_OF_MEMORY/LOAD_FAILED/GENERATION_FAILED/UNSAFE_OUTPUT/SAVE_FAILED`。

[可复跑验证与虚构配置](../examples/vision_support/TASK_C_VALIDATION.md)
