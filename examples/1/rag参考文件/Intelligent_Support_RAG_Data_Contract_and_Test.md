# 智能客服长期记忆 RAG｜数据契约、虚构样例与实际测试任务

> 通用工程补充任务。继续当前多人会话、训练、推理和记忆接口的改造，在记忆模块中补齐可复用的检索能力。所有附带资料都是为测试独立构造的虚构设备、工单、服务群和客户记录，不代表实际业务或真实客户。不得用生产客户数据替换测试数据，亦不自动部署或启动训练。

## 1. 真实任务及现有能力

为智能客服提供准确的历史工单、设备状态、维修记录、历史聊天和知识资料检索。单人和多客户群聊均需理解当前会话中的真实人物、历史问题与状态更新；无关聊天不应强行插入历史。数据检索与对话模型训练独立，RAG 不要求将事实写入模型权重。

先核对实际当前源码的 memory Provider、ContextCard、ContextBuilder、eval 及已实现的通用接口，保留已有 `ManualCardProvider` 和 `NoMemoryProvider`。旧版可见的 `ContextCard` 字段包括 card_id、namespace、kind（device_spec / incident_fact / troubleshooting_note / plan）、facts、event_time、recorded_at、version、status、supersedes、source_ref、uncertainty、permissions(read/train)、scope、session_id；旧 `select(cards, role, session_id)` 只筛手工卡快照，没有查询和全局撤销机制。以当前 HEAD 为准，若已实现则复用。不要构建第二套独立推理 runtime 或把附带的示例字段强制硬编码为生产接口。

## 2. 附带可执行测试资料

使用 `Intelligent_Support_RAG_Fixtures.zip`，其中包含独立虚构的原文、证据片段、事件版本、实体和别名、证据关系、权限策略、查询黄金结果、旧版手工卡示例与离线校验工具。所有内容均为 `fixture_only`。字段是可移植参考语义而非当前代码已实现的生产 schema；允许通过版本化适配器转换。实际数据契约以最终实现及校验命令为准。

| 文件 | 数据语义 |
|---|---|
| sources.jsonl | 虚构工单/日志/会话原文，来源哈希、时间、归属、范围 |
| chunks.jsonl | 原文字符区间及准确 hash、可追溯证据 |
| events.jsonl | 稳定 event_id + version、断言及证据、状态、时间、审核、权限 |
| scopes.jsonl | 测试服务端信任的身份/会话/用途映射，不从用户文本决定权限 |
| permissions.jsonl | 可测试的读、索引、披露、训练和允许 scope 独立策略 |
| entities.jsonl / aliases.jsonl / event_links.jsonl | 设备、不同人员、代号与有证据的关系 |
| revocations.jsonl | 权威撤销记录，不依赖请求中是否附带旧卡 |
| queries.jsonl | 固定查询、允许/禁止的证据及预期行为 |
| manual_cards.jsonl | 兼容旧 ContextCard 语义的人工卡对照 |
| negative_cases.jsonl | 非法输入和冲突/撤销/权限回归描述 |
| manifest.json / SHA256SUMS / validate_fixtures.py | 固定版本、哈希及可执行的结构/参考 ACL 验证 |

## 3. 原文、事件与短记忆卡分层

例如虚构的客户服务群：客户甲报告打印机离线，技术人员解释该设备端口，随后确认维修完成；另一服务群也有一个昵称相同的客户，故障是扫描失败。系统应识别具体群和设备，不能把两人的问题合并。工单可能先记录“计划换线”后改为“网线无需更换，端口松动已修复”，历史计划不等于实际结果。

一条来源证据有稳定 source_id、记录时间和原文；chunk 用半开字符区间 `[start,end)`（Python Unicode code point）指向来源，并保存切片 hash；event 的每条事实有明确证据引用。源数据与生成摘要不可互相覆盖。事件时间与记录时间分别表达；未知值为 null，不能硬填当前时间。版本用于更正，过去真实观察可以保留为独立事件，而已撤销记录永远不能通过旧卡绕开。

下面是**旧版手工 ContextCard 投影**的示意；它不是原始 RAG 数据，更不能作为客户端自行授权的输入。附包 `manual_cards.jsonl` 中有相同结构的实际 fixture：

```json
{
  "card_id": "incident_a_status",
  "namespace": "service_demo",
  "kind": "incident_fact",
  "facts": ["服务群甲的演示打印机当前已恢复在线；确认原因是网口松动。"],
  "event_time": "2026-07-08T10:30:00+08:00",
  "recorded_at": "2026-07-08T10:32:00+08:00",
  "version": 2,
  "status": "active",
  "supersedes": 1,
  "source_ref": "src_repair_a#chunk_repair_a",
  "uncertainty": "虚构测试中由维修记录确认",
  "permissions": {"read": true, "train": false},
  "scope": "session_only",
  "session_id": "group_a_main"
}
```

旧 ContextCard 没有全局 ACL、跨租户策略、逐断言证据和撤销持久状态；新 Provider 必须先查权威授权及最新版本才能投影为卡。`read=true` 或 `scope` 不能由外部请求自签。原文来源、哈希、权限只用于校验与审计，不作为模型指令。注入使用既有 `reference_section` 等资料标界及控制字符转义；检索内容是不可信资料。

## 4. 实际开发与接口要求

1. 定义版本化数据导入、源/片段/事件外键、状态和权限校验；复用旧 ContextCard 类型和手工卡路径。允许业务类型扩展但兼容已有配置，未知 schema/权限默认拒绝。所有正式源必须可追溯；fixture 与 production 隔离。
2. 新查询接口接收可信 session/account、namespace、platform/bot/chat/thread、当前查询、必要历史、时间线索和 budget。聊天里自称管理员、提交任意 scope 或恶意控制符不能获得授权。服务端解析可信 scope 后先过滤 ACL/有效版本/用途，再召回，返回短证据和可审计引用或空结果。注入前二次验证权限/撤销版本，避免缓存旧证据。
3. 优先复用 SQLite/FTS 或已有轻量索引。中文短词、代号、同义词、时间及上下文指代纳入测试。需要时再增加本地 embedding/混合检索，不能为了实现“智能”默认上传原文。模型接收经筛选的证据卡，不把所有命中拼入 prompt，不重复维护聊天历史；ContextBuilder 使用真实 tokenizer 决定预算。现有 3 卡/400 增量 token 可作起始参考，不能承诺每次都适用。
4. 当前事实使用最新有效版本；未确认计划、客户猜测、自动日志等注明归属和不确定性。对历史问题可引用明确有权限的历史记录，但撤销事实、待审核候选、未来记录、被权限拒绝的材料均不能通过日期/别名/关联召回。别名按会话和时间解析，同名不得跨群合并；关联边不授予读权限。
5. 保留无卡、人工卡、自动检索三种可独立使用的路径。权限、召回、排序、生成、实际 token 预算分别评估。RAG 数据和模型训练数据权限分离，测试时使用本包的独立测试 scope，不将 fixture 发布成正式客户事实或训练集。

## 5. 必测场景与验收

| 测试类 | 必须验证 |
|---|---|
| 多客户群聊与同昵称 | 正确定位客户和设备；不跨群/线程/平台读取 |
| 关键词、短代号、上下文指代 | 中文短词可检索；指代不明应澄清而不是猜 |
| 历史与现状 | 历史记录可查，当前事实用有效版本；计划不冒充结果 |
| 撤销及冲突 | 权威撤销后旧 active 卡重放无效；同版本重复冲突拒绝 |
| 权限与留出 | 私聊资料、跨租户、未批准索引、独立评测证据不能混入日常请求 |
| 恶意资料内容 | 转义/隔离资料内伪造角色和指令，不提升工具或系统权限 |
| 无关及无证据 | 不强行插入工单历史，不编造故障 |
| 预算与兼容 | 真实 tokenizer 时检查卡片增量 token、裁剪和原有手工卡入口 |

运行附带的 `python validate_fixtures.py`，它仅检查虚构数据哈希、外键、原文范围和参考权限/版本断言，**不是**生产检索器，也不是已通过实际项目的证据。把 fixture 转为当前实际接口后编写测试并运行真实代码；报告真实 query→候选→权限筛选→排序→最终卡片的差异（仅虚构数据）。黄金结果中的 required 证据须可召回，alternative 可作为合法替代，forbidden 在任何阶段不能进入可见输出；不要把全部可读记录都当相关结果。区分参考策略与实际实现，发现不匹配先核对契约，不降低门禁。

## 6. M0–M5 效果测试

固定模型和生成参数，分别跑 M0 无卡、M1 人工正确获准卡、M2 无关获准卡（仅隔离诊断）、M3 过期/冲突（生产路径须过滤，受控测试验证历史时序）、M4 无证据/指代不清、M5 自动检索。至少记录 recall@k（固定相关证据集合）、错误权限命中、过期/撤销命中、无关卡插入、实际检索耗时；有真实 tokenizer 时记录卡片增量 token。有实际模型时才评事实准确、自然、不硬插及承认不知道。没有权重时只交付工程测试，不声称生成质量通过。若 M1 给对仍然无法正确回复，先修卡片注入/模型适配，不投入复杂检索；M1 好而 M5 差再改召回或排序。

## 7. 交付给数据接口负责人

请在当前代码改造完成后，依据实际实现提交 `MEMORY_DATA_CONTRACT.md`、机器可读 schemas、真实可运行的 synthetic fixtures、版本迁移/导入与校验命令、权限及撤销测试、真实 query→card 的预期/实际结果、已有接口兼容矩阵，以及 `test_report.md`（commit、依赖、命令、结果、未完成能力）。与当前多人训练/推理的数据接口交付整合；仅交付获准的通用接口及独立虚构测试资料。若模型、tokenizer、持久存储或外部框架未验证，明确 BLOCKED/NOT RUN，不把 mock 当实测。优先复用已有模块，任何越权、旧卡重放、留出泄漏、跨会话记忆混合均阻断上线。
