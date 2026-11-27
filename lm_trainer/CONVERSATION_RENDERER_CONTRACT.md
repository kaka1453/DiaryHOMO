# Conversation Renderer / 阶段二任务 A

Renderer 标识 `lm_trainer.conversation-renderer.v1`。读取候选、验证发布和编译 token 是独立步骤；能渲染不表示能训练。

## 三层数据

1. **源事实** `lm_trainer.conversation-source.v1`（JSON）。`conversations` 为数组，每项有 `id`, `event_group`, `mode` (`direct|group`), `events`，可带 `relations`。事件为 `source_id`, `position` (非负、严格递增), `speaker` (账号 ID), `person` (作者 ID), `timestamp` (有时区的 ISO 8601、非递减), `kind` (`text|image|file|audio|video|unknown`), `body_available`, `readable`, `selection` (`text|structure|hidden`), `selection_basis` (`online|witness|hidden`，必须显式给出)，`content` 仅在可读、可用且选作正文时必填。`available_for_training: true` 必须显式存在才可用作目标。账号不能变更作者；目标作者在一个会话内只映射一个账号。昵称不是身份。边由 `id`, `owner_id`, `kind` (`reply|quote|mention`), `resolution` (`resolved|unknown|protected|display_handle_only`) 标识：已解析 quote/reply 只能用 `anchor_id` 指向较早同会话事件；已解析 mention 只能用 `speaker`；不可解析的边不得附加内容、锚点或身份。原始边正文从不恢复。
2. **候选视图** `conversation.stage3.structured.v3`。`parse_catalog` + `inspect_candidate` 是只读入口；候选为 `training_authorized=false`，尚未分区。`adapt_window` 校验同一 v3 结构（也供发布验证使用）。正文、位置、作者、时间、可用性、事件槽与源事实核对；`target_mask`、`target_ids`、独占 `target_causal_prefix_lengths` 和 `generation_blocks` 必须精确重算。按同账号、同监督标记、相邻源位置、≤20 秒且无媒体/关系打断合并。`timeline` 允许较早历史整体不入窗，但不能跳过其后窗口内的在线事件；不可用或未选的正文不自动恢复。`history_relation_refs` 校验为选中文字 owner 的边；`target_relation_audit_only` 仅审计。`scene`、全窗人名、时间和标签不作为 prompt 元数据。
3. **时点投影** `Window.observation(focus_id)` 严格用 `timeline[:n]`；在线 `observe(catalog, conversation_id, observed_ids)` 不需要任何目标字段。远端锚点使用 `selection_basis=witness`，必须带 `witness_source_ids` 和 `visible_from_position` 的源事件（两者成对，至少一个较晚的在线 owner，均有解析的引用边）；只有所有见证 owner 已观测且已过激活点才显示。缺独立依据的候选远端选择不被信任：源适配者必须标为 `hidden` 或提供证据，不得冒充在线连续历史。只显示已可见 owner 的边；锚点必须为可读可见正文；mention 和 quote/reply 分别呈现。resolved mention 不猜从未出现的账号；未解析或 protected 边不显示。

## 渲染与 token API

- `render_observation(observation, target_person, system_prompt)`：只接受 `system/user/assistant`。direct 正文无群聊标记；group 在原生 user turn 内逐气泡使用稳定 `[P1]` 式短别名；目标账号的历史在 assistant（关系标记中为 `[客服]` 的称谓）。别名用源观察顺序分配、不随裁剪重编号、不会提前使用未激活锚点分配别名。仅群聊 user 侧显示紧凑长间隔。媒体在 user 侧作为观察呈现，用类型占位；如有可见说明可随占位呈现，不进入 loss。独立开场、无新观察的纯客服延续 hold，不填虚构问句。只有调用方的模板策略能配置 system；拒绝保留角色/工具标记，保留正常换行、代码和 emoji。
- `ContextBuilder.build_conversation(tokenizer, resolved_policy, observation, *, target_person, context_limit, max_new_tokens)` / `encode_conversation_prefix`：调用原生模板 `add_generation_prompt=True`，返回 `BuiltContext.input_ids` 与 `conversation_renderer` 标识。现有有状态的双人 API 不变；不隐式加入 cards、平台字段或未来内容。
- `encode_conversation(tokenizer, window, template, resolved_policy, plan)`：每个块单独返回一个 `Encoded`；joint 是一组独立 replay，而非把未来观察塞入第一条或重复监督上一目标。结果带 `prefix_ids`、半开 `TokenSpan(start, end, kind, source_id)`、`focus_ids`、`template_hash`、`stop_hash`、`renderer_version`；可把推理 IDs 和标识交给 `verify_training_prefix(..., identity=...)`。差分重建原生 literal 槽，比较独立分段与完整 native token 流，再以训练正文之前的实际 tokens 和独立推理 tokens 逐位比对；不使用正文字符串查找。正文 token 监督，`
` 气泡分隔（换行）不监督；每机会最多一个配置的原生 stop（`supervise_target_eos`），padding 即使 PAD=EOS 也不监督。保留角色头、历史、隐藏块和上下文元数据为 `-100`。跨界合并、非原生可见正文、首位/仅 EOS/空目标、前缀不符和超预算均拒绝。v1 当前策略不截断；模型预算应该为相同完整观察留足空间。
- `validate_route(windows, routes)` 在预检中校验默认选路由、跨 joint/primary 的唯一源目标。发布文件中仅有默认选中的视图。`author_blocks` 供 CPT 阶段构造同作者的连续原文（不跨他人、关系、时间间隔或媒体）；既有 CPT 数据契约不变。

## 发布、兼容与限制

既有 `lm_trainer.corpus.v1`、兼容 v1 别名、`lm_trainer.release.v1`、legacy 与双人入口不变，旧路径的多 speaker guard 保留。v3 仅以 `corpus_schema_allowlist: [conversation.stage3.structured.v3]` 明确启用；使用独立 `lm_trainer.release.v2`（目前仅结构化 SFT），拒绝未知的结构化版本。v2 复用 v1 文件、hash、审批、source-group 及无 test payload 读入规则；另外必需 `conversation_source`, `routing` 两个 `{path, sha256}` 引用和 `partition_splits`。每个打开的分区验证精确 ledger 覆盖，formal 须有 train 和至少一个评估分区；有真实可见性或目标的源（含选择见证）落在同一来源分组。各分区用独立正文 catalog（catalog 声明 `split` 和 `fixture_only`），共用只含元数据的 ledger 和 source index，不混放或读取 held-out 的正文文件。v2 正式记录需 `kind=sft`, `scope=release`, `fixture_only=false`, `review_status=candidate_status=approved`, `disposition=approved`, `training_authorized=true`、独立记录/manifest `approval_ref` 及逐条权限。另需独立 `source_index`，source IDs 对所有分区的 ledger 覆盖恰好相同，每条含 `event_group`、`split`、S/F 裁决及显式 train 权限和授权。记录的 `source_index_ref` 指向清单中核验后的文件（fixture 未配置 index 时指向核验的 catalog）。SHA256 锁定审批者给出的内容，不验证审批者现实身份。v2 执行入口仅允许 HF；Unsloth 未通过真实环境 parity，继续拒绝。

合成 fixture 独立使用 `scope=fixture, fixture_only=true, training_authorized=false`，通过显式测试批准、`permissions.train=true` 和测试分区跑 CPU 测试；不会改写候选或得到正式产物。若只有参考 Markdown，则用其 joint 投影、配套自建纯合成源事实与回归；无 ZIP 不声称验证原始全量交付包。CLI 仍复用 `--dry-run` / `--validate-release` / `--simulate-only`，结构检查与实测 token 统计明确区分。不会下载模型或运行正式训练。


## 最小只读调用

```python
catalog = parse_catalog(independent_source_document)
window = inspect_candidate(candidate_row, catalog)   # 不授予训练权限
seen = window.timeline[:window.boundaries["joint-m02"]]
observation = observe(catalog, "joint", seen)          # 在线调用直接传已观察的源 ID
prompt = ContextBuilder.build_conversation(
    tokenizer, resolved_policy, observation, target_person=window.target_person,
    context_limit=1024, max_new_tokens=128,
)
```

`parse_catalog` / `inspect_candidate` / `observe` 从 `lm_trainer.conversation` 导入，`ContextBuilder` 从 `lm_trainer.chat` 导入。线上调用方须先做读取 ACL、身份映射与观察选择；不得把客户端自行声明的账号、见证或已读集合当作可信事实。Renderer 不是自然语言注入检测器，也不赋予用户原文任何系统角色。工程验证与复跑命令见 `PHASE2_TASK_A_REPORT.md`。
