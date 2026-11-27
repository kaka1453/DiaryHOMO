# Task D · 本地权限化事件检索契约 v1

状态：离线工程验证；不是正式模型效果或生产接入批准。实施及实测见 [Task D 报告](PHASE2_TASK_D_REPORT.md)。

## 1. 所有权边界

| 层 | Owner | Task D 边界 |
|---|---|---|
| 当前输入、可见历史、引用 | 上游调用方 + [任务 A](CONVERSATION_RENDERER_CONTRACT.md) | 只借用可信 scope 的本次输入和已观察文本；不维护会话历史、不扩大观察范围。 |
| 获准知识、人工审核事件、原文指针 | 授权数据负责人 | 审批后由离线管理入口导入；查询只注入短卡，不写回对话、索引或训练。 |
| 权重与训练 release | 训练/产物管理员 | `train` 只是独立元数据；D 不导出训练样本。关闭 RAG 不会清除模型记住的受限信息，群聊必须使用与披露范围相容的已授权产物。 |
| 模型生成、未确认发言、独立评测证据 | 调用方 / 评测负责人 | 不作为自动入库事实；留出集用独立 purpose 与 scope，推荐独立物理库；在线 adapter 默认拒配 `heldout_eval`。 |

## 2. 查询与记录

- `lm_trainer.memory`: `MemoryQuery` (`lm_trainer.memory-query.v1`) → `RetrievalMemoryProvider.retrieve()` → `RetrievalResult(cards, audit, reason, elapsed_ms, scored_count)`。输入为当前文本（≤4096 字符）、可选可信上下文关键词/实体 ID、事件时间半开范围 `[event_from, event_until)`、可选 `knowledge_as_of`、`k`（0–16）和可选受限关联补充（默认 0，上限 2）。中文日期线索由调用方显式归一为带时区时间；Task C 默认适配器不猜测日期、实体或历史摘要。用户文字（包括自称 scope）决不决定授权。候选上限 256、查询倒排词上限 192，超限返回 `budget`；未匹配 `no_evidence`，同名冲突 `ambiguous_entity`，未认证绑定 `scope`。这些是工程原因，不等于模型已学会澄清。
- `EventRecord` (`lm_trainer.memory-event.v1`): 全库稳定 `event_id`、namespace/purpose、类型、简明事实、`fact_type`、不确定性、实体 ID、绝对时间、版本/状态/替代版本、ACL、来源范围与 SHA-256；可选人工确认别名、检索词及相关 ID。`content_hash` 是 canonical JSON `facts` 数组的 SHA-256；`sources[].content_hash` 是原文切片的 SHA-256，按 Python Unicode code-point 下标切片。即使有摘要，也不能冒充原话。`unknown` 是明确的未知时间：发生时间未知仍可检索但不匹配显式日期范围；记录时间未知不能证明当时可见，因此不召回。`knowledge_as_of` 仅比当前时间更严格，不恢复历史 ACL/版本。合法的历史描述需独立已审核、仍有效的历史事件；不会把同一 ID 已替代的版本带回。
- 原文先按日期、自然段、连续事件切分；审核摘录与事实类型、SHA 和跨度后再导入。事件最多 6 条短事实（合计 ≤640 字符）。一个事件不混淆客户判断、客观确认和计划。多来源派生记录应由数据负责人取**各源 ACL 的交集**，不得自创跨 scope 关系。D 核心导入只信受控操作者签发的规范化记录，无法验证任意外部文件的真实性；[虚构数据适配器](../examples/memory_rag/fixture.py) 实际校验压缩包、原文哈希、片段边界及证据外键，并交集授权。
- 原有四种 `ContextCard` kind、`CardPermissions(read, train)`、`MemoryProvider.select`、`ManualCardProvider`、`NoMemoryProvider` 与 `reference_section()` 保持兼容。新增 `shift_log`、`historical_judgment`、`confirmed_fact`。仅短卡内容、时间、kind、带 `fact_type` 的不确定性注入；不渲染 `source_ref`、ID、ACL 或检索词。不信任资料里的命令；渲染前阻断保留协议标记、伪造角色头、控制字符及原生 tokenizer 的控制 token，原有标签分区和 JSON 转义仍生效。限预算时按排序先尝试高分卡，无法容纳才尝试更小卡；当前问题、整个显式历史/观察前缀和生成余量优先，超出必要前缀预算就拒绝，不截断。

## 3. 授权与撤销

- `TrustedScope` 由服务端部署配置预先绑定，须精确匹配 SQLite 注册的全部字段：群聊 `channel + client_id + room_id + thread_id`，工作台 `web_account + session_id`，均同时核对 `scope_id + role.namespace + purpose`。频道和主体使用稳定 ID，不使用昵称或来源自称的身份；认证凭证到 scope 的授权由任务 C/网关负责，普通请求参数不构造 `TrustedScope`。内部、不同群、群主题、平台、账号、会话和留出默认不互通。
- SQLite 先物化权限集合（精确 scope、read、index、active、有效时间和最新版本；群聊另需 `group_disclose=true`），才查倒排和排序。`train=true` 不代表任何读取或披露权限。索引用标准库 SQLite 字符二元组/英文词倒排，结合 NFKC、大小写归一、审核的术语、型号或变体；同时支持保守的中文字面长锚点。别名只从该 scope/时间下已授权的证据解析；同名多主体不猜测。关联边不能授权访问：邻段也要独立过权限，并与高分锚点有共享实体。
- 唯一权威是受控私有 SQLite 的 scope、当前状态、版本 digest、事件与来源撤销墓碑。相同版本相同数据可重导；较低版本不替换；同版本不同内容、ACL 或身份进入终止冲突状态，不被普通更新解除。改变 ACL 必须发布递增版本、停用或撤销。终止 ID 不复用；如果确需纠正，按组织流程发布新的、独立审核的记录，不保留已撤销来源或借新 ID 洗白。`delete`/`revoke` 单事务移除活动倒排、grants 和事件 payload，只保留阻止重放所需的最小 ID、版本、digest 和来源关联。没有跨请求查询缓存。服务端卡快照还要在 `ContextBuilder` 真正构造引用 prompt 前由 authority 再核对，比较整个规范投影而非只比较版本；对同批重复或伪造卡拒绝。SQLite 写保留锁将最终读校验与同步 prompt 组装对撤销操作线性化。已组装的在途 prompt（或已输出的文本）无法事后追回；应急拆权须在网关停发并中止/排空在途请求。
- 权威损坏、缺失、版本不兼容、数据库繁忙或 ID 冲突不得回退到未验证卡。只允许通过同一在线 authority 恢复派生索引（`rebuild`），或向它 `update` 历史数据包，旧卡不恢复。**禁止将旧的整个 SQLite 文件拷回当在线权威**：回滚含墓碑的唯一权威文件、恶意宿主机管理员或篡改系统时钟不在本地单机威胁模型内。若最新权威/撤销流水无法核实，保持 RAG 停用，不初始化一个旧备份当替代权威。撤销/删除不担保 SSD、快照或导出件上的物理擦除；备份和原文的留存/销毁须单独按数据治理流程处理。

## 4. 接入任务 C

```python
from lm_trainer.memory import MemoryStore, SqliteMemoryProvider
from lm_trainer.chat.inference import InferenceServiceV1

# trusted_rag_scopes 来自部署配置，与网关凭证绑定及 C 服务 binding 一一对应。
provider = SqliteMemoryProvider(MemoryStore(private_data_dir), inference_scopes=trusted_rag_scopes)
service = InferenceServiceV1(runtime, model_id, c_bindings, memory_provider=provider)
```

`AuthorizedContextV1` 的 binding/role 均来自服务端，provider 只取最后一条已观察到的文本做默认查询，未传完整 source catalog。任务 C 仍有 namespace、`session_only`（其中 `session_id` 等于授权 binding ID）、read/版本等旧防线；D 的 `prompt_guard` 通过同步 ContextBuilder 附加二次验证。自定义 Python 集成按相同顺序：`retrieve` → `builder.build_inference_v1(..., card_guard=partial(store.prompt_guard, query.scope, query=query))`。不要把 `retrieve().cards` 当成可绕过最后校验的注入凭证。旧 Manual / NoMemory 继续用于旧离线对照；没有 authority 的 Manual 只验证候选快照，**不得作为真实群聊或已配置 RAG 的授权后备路径**。HTTP 启动器未自动读取 RAG 配置，接入方显式装配 Python provider；没有修改路由或接入网络。日志只保留计数、耗时、token/版本和固定理由，不公开卡 ID、查询、正文、路径或 ACL；授权 Python 接口可返回卡的 ID/版本用于审计。

## 5. 本地管理与回滚

```sh
set -o pipefail
D=$(mktemp -d /tmp/lm-rag-d.XXXXXXXX)
.venv/bin/python -B -m examples.memory_rag.fixture |
  .venv/bin/python -B -m lm_trainer.memory --data-dir "$D" build --bundle -
.venv/bin/python -B -m lm_trainer.memory --data-dir "$D" audit --scope scope_a
.venv/bin/python -B -m lm_trainer.memory --data-dir "$D" revoke --event revoked_a_note
.venv/bin/python -B -m lm_trainer.memory --data-dir "$D" delete --source src_repair_a
.venv/bin/python -B -m lm_trainer.memory --data-dir "$D" rebuild
.venv/bin/python -B -m lm_trainer.memory --data-dir "$D" audit
# 增量导入：memory --data-dir "$D" update --bundle <受控 JSON 路径或 ->
```

`build` 唯一可创建 authority 的指令，对已有库只增量导入，绝不覆盖。`update`、`rebuild`、`audit`、`revoke`、`delete` 不创建缺失的 authority。目录必须为当前用户私有并在 Git 工作区之外；新建 directory/database 分别用 0700/0600，拒绝宽权限或链接，默认启用 SQLite `secure_delete`。使用 [导入 JSON Schema](memory/import-v1.schema.json)（结构约束）；Python importer 另外校验权限绑定、内容哈希、时间、版本、跨表字段关系。服务如需回滚关闭 D provider，恢复无卡基线，保留最新 authority 与墓碑；数据和模型回滚分开处理。上面的临时数据只含虚构记录，完成后可手动移至回收站。
