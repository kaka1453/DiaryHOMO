# 智能客服群聊：多参与者会话数据契约、训练适配与测试任务

> 2026-11-27 · 通用编码 AI 使用 · 所有数据均为独立编造的 synthetic fixture，不是真实公司业务日志，不用于评价真实客服模型质量。本文授权在指定代码工作区内实现通用格式支持及轻量测试，不授权使用生产数据、正式训练或部署。

## 1. 实施目标

现有智能客服聊天可能包括客户、现场工程师、其他客服、值班机器人；同一人可连续多条，多个成员可能同名，消息可包含附件、异步引用、长时间间隔。目标是扩展通用训练器，使其能理解多人交流并只训练指定客服身份的实际答复，同时保留既有双人对话、CPT/SFT、精确 mask、训练与推理格式一致性及正式 release 门禁。

读取实际仓库和 git 状态，保护未提交修改；核对 `lm_trainer/train/schema.py`、`encoding.py`、`data.py`、`sampling.py`、`engine.py`、`modeling.py`、`stages.py`、`template_policy.py`、`chat/context.py` 和现有 `TASK_A/B/C_CONTRACT.md`（若实际路径变化，以现有源码为准）。若已有实现，补缺口。不要将训练器变为业务专用代码，不创建另一个训练器，不直接删除多人安全限制。所有示例必须与真实业务数据隔离。

## 2. 数据层契约

使用明确中性候选 schema `conversation.stage3.structured.v3`；这是本测试规定的输入契约，不是现有 trainer release schema。实现可配置、显式注册的等价版本适配器。禁止简单改名后绕过验证、宽松接收未知版本或将结构化输入降级 legacy。先导入统一事件，再按真实基座原生模板生成实际 tokens/labels。下面字段及文件语义固定；补充字段、版本迁移需记录。

| 文件 | 用途 |
|---|---|
| primary.jsonl / joint.jsonl | 基础窗口及联合窗口；统一结构，包含历史和候选目标 |
| routing.jsonl | asset/targets/view/status/active_example/source/available_for_training；未物化库存可能仅 asset/targets/status/reason。每个 source target 唯一默认 route；备选替换而非增加 |
| source_events.jsonl | 源事件及媒体类型；id/conversation/position/platform/speaker/person/timestamp/record_kind/event_types/raw_status/source_ref/body_available/protected/payload |
| relations.jsonl | 原生 relation edge_id/source/kind/resolution；reply/reference/quote 可有 target/availability/target_position/target_ts/target_speaker；mention 可有 identity/mentioned_speaker/mentioned_person/not_a_reply_inference |
| source_position_index.jsonl | source_ref、组件、源位置、身份及 split，不是正文 |
| partition_components.jsonl | source_ids 同一依赖组件；候选未分区、未授权 |
| cpt/support_chat_cpt.jsonl / support_author_blocks.jsonl | 客服本人历史文字与连续作者块，供 CPT 逻辑测试 |
| notes/note_candidates.jsonl / note_events.jsonl | 虚构值班笔记原文片段及事件组，用于 CPT/RAG 接口测试 |
| selection_witnesses.proposed.jsonl | 提案格式，专门验证选择因果；不是必然存在的现有字段 |
| source_fixture/*.jsonl | 本次生成的原始虚构记录，可核对 SHA256；manifest 为 fixture 清单，不是正式发布清单 |
| aliases_fixture.json / expected.json | 昵称冲突与期望前缀、关系、监督目标和阻断行为 |

窗口规则：
- messages 是选中正文气泡，逐条 source_id/position/speaker/person/timestamp/content/supervised。speaker 是账号身份，person 是作者归属；别名短且稳定，同昵称不能合并。
- target_mask 与 messages 严格等长，target_ids 按顺序等于 supervised=true 的 source_id。target_person 为当前指定客服，而非写死模型角色。
- timeline 是全部选中的正文和结构，按源位置去重排序。text_slot 决定是否有正文；body_available 是来源可用性；structure_selected 是结构事件选择。结构槽有可用正文不代表被选中，禁止自动恢复。timeline.target 及历史 supervised 属于审计标签，不进提示词。
- generation_blocks 按同账号、同监督状态、原始位置相邻、≤20秒、无媒体等打断条件组织，ids 必须恰好覆盖 messages。只是真实表达候选组，不等于原生 role turn。结构和历史不入 loss。
- `target_causal_prefix_lengths[focus_id]` 是 timeline 的独占边界：只使用 `timeline[:n]`，且 `timeline[n].id==focus_id`。不能误用为 messages 索引；随后按 text_slot 提取实际历史文字。
- history_relation_refs 为 [owner_id,edge_id]；仅 owner 当时可见才能呈现该边，引用锚点须当时已在可见前缀且权限允许；未解析或不可读引用不补正文。target_relation_audit_only 是目标自己的引用/@ 等，不能作为同一目标的输入。之前目标只在后来成为已观察的历史时可见。
- `scene.window_speakers`、segments、trace、全窗关系、target 自身的时间和附件、选择标签均不能放在第一轮模型输入前。每个生成时点重新计算实际可见人物、间隔与关系；mention 不等于 reply。display_handle_only 的 @ 不推断账户 ID。
- 选择因果：旧锚点即使时间早于目标，也可能是更晚消息的引用才使它被选入窗口。若没有当时已成立的独立选择依据，不允许提前呈现此锚点；候选证据不明时隔离该视图或用可证明的在线策略。样例 sidecar 用 witness_source_ids/visible_from_position 表达提案，真实实现可等价版本化。
- adapter_candidate 仅候选，不是训练批准；review_only、retained_no_exposure 不训练；无 user 历史的 independent_opening/author_continuation 不伪造提问，默认 hold 或独立 CPT/实验。

## 3. 可机读样例

下面是附件中 joint 窗口的实际字段投影（完整行及 companion 索引见 fixture ZIP；此投影没有列出全部审计字段）。文字均为新编客服场景。

```json
{
  "id": "win-joint-main",
  "schema": "conversation.stage3.structured.v3",
  "target_person": "support_agent",
  "conversation": "joint",
  "platform": "fixture_im",
  "view": "dialogue_continuous",
  "disposition": "adapter_candidate",
  "training_authorized": false,
  "messages": [
    {
      "source_id": "joint-m01",
      "position": 1,
      "speaker": "acct-customer",
      "person": "other",
      "timestamp": "2026-01-01T09:02:00+00:00",
      "content": "网关重启后仍然报错",
      "supervised": false
    },
    {
      "source_id": "joint-m02",
      "position": 2,
      "speaker": "acct-agent",
      "person": "support_agent",
      "timestamp": "2026-01-01T09:02:02+00:00",
      "content": "请先确认电源指示灯",
      "supervised": true
    },
    {
      "source_id": "joint-m03",
      "position": 3,
      "speaker": "acct-tech",
      "person": "other",
      "timestamp": "2026-01-01T09:02:05+00:00",
      "content": "指示灯正常，日志也能导出",
      "supervised": false
    },
    {
      "source_id": "joint-m04",
      "position": 4,
      "speaker": "acct-agent",
      "person": "support_agent",
      "timestamp": "2026-01-01T09:02:08+00:00",
      "content": "那先不要恢复出厂设置",
      "supervised": true
    }
  ],
  "timeline": [
    {
      "id": "joint-m01",
      "text_slot": true,
      "target": false,
      "event": false,
      "body_available": true,
      "structure_selected": false,
      "gap_seconds": null,
      "gap_class": "unknown"
    },
    {
      "id": "joint-m02",
      "text_slot": true,
      "target": true,
      "event": false,
      "body_available": true,
      "structure_selected": false,
      "gap_seconds": 2.0,
      "gap_class": "within_30s"
    },
    {
      "id": "joint-m03",
      "text_slot": true,
      "target": false,
      "event": false,
      "body_available": true,
      "structure_selected": false,
      "gap_seconds": 3.0,
      "gap_class": "within_30s"
    },
    {
      "id": "joint-m04",
      "text_slot": true,
      "target": true,
      "event": false,
      "body_available": true,
      "structure_selected": false,
      "gap_seconds": 3.0,
      "gap_class": "within_30s"
    }
  ],
  "target_mask": [
    false,
    true,
    false,
    true
  ],
  "target_ids": [
    "joint-m02",
    "joint-m04"
  ],
  "generation_blocks": [
    {
      "speaker": "acct-customer",
      "person": "other",
      "target": false,
      "ids": [
        "joint-m01"
      ]
    },
    {
      "speaker": "acct-agent",
      "person": "support_agent",
      "target": true,
      "ids": [
        "joint-m02"
      ]
    },
    {
      "speaker": "acct-tech",
      "person": "other",
      "target": false,
      "ids": [
        "joint-m03"
      ]
    },
    {
      "speaker": "acct-agent",
      "person": "support_agent",
      "target": true,
      "ids": [
        "joint-m04"
      ]
    }
  ],
  "target_causal_prefix_lengths": {
    "joint-m02": 1,
    "joint-m04": 3
  },
  "history_relation_refs": [
    [
      "joint-m02",
      "edge-003"
    ],
    [
      "joint-m04",
      "edge-004"
    ]
  ],
  "target_relation_audit_only": [
    [
      "joint-m02",
      "edge-003"
    ],
    [
      "joint-m04",
      "edge-004"
    ]
  ]
}
```

第一目标 joint-m02 的观察前缀只有 joint-m01，其 own reply edge 不可见；第二目标 joint-m04 可看到 m01–m03，并看到前一次客服回复对应的历史 reply，但不能看 m04 自己的关系。两个目标的正文各监督一次。关联的 primary 两个单窗是备选，默认路由只能选 joint 一次。

## 4. 交付包的测试场景与预期

本任务附带可独立生成的 `Intelligent_Support_Groupchat_Fixtures.zip`。解压可直接读取 JSONL 和 `expected.json`；其中 `build_support_fixture.py` 可在空目录重新生成全部数据（仅 Python 标准库，无网络）。命令：`python build_support_fixture.py /path/to/empty_fixture_dir`。该脚本不运行模型、不修改训练器，不是生产发布工具。初始包含 9 个 primary（含2个 joint 的单窗替代）、2个 joint、10条路由、33个源事件、8条原生关系、16个独立库存目标；另有17个客服 CPT 正文和13个作者块、1个值班笔记。真实生成结果以文件清单为准。

| 场景 ID | 必须验证 |
|---|---|
| direct | 原生双人 user/assistant，两个连续客服气泡组成生成机会，后一泡不插虚构用户消息 |
| group | 同昵称的不同人保持身份，第三人补充，旧客服发言不监督，明确目标的 own reply 不提前暴露 |
| joint | 两次独立生成机会，前一目标成为后一历史；单窗与联合窗不可重复监督 |
| media | 图文混合、可用但未选的文件正文不得注入；媒体打断同作者表达块；未知媒体不能编造内容 |
| anchor | 两小时前的旧消息仅因后来的引用选中。早一目标过滤该旧锚点（除非另有前置选择证据），后一个目标可使用；后续所有者关系不能提前出现 |
| relation | display-only @ 不猜账号，resolved @ 不推断回复，unknown quote 无正文；protected source 永不恢复 |
| opening | 无历史的客服独立开题，不虚构问句 |
| continuation | 只有客服历史，保持 CPT/实验性路线，不能默认标准 SFT |
| review | 未确认记录即使有 mask 也不等于正式批准 |
| inventory | 无 active_example、缺 available_for_training 的库存，不默许为 true |

## 5. 本地 AI 要实际实施的代码

A. 通用 importer + renderer：候选数据以只读预检入口处理，保留 schema、稳定 ID、目标身份、时间、结构、来源与可见性。建议 `lm_trainer/conversation/` 公用事件与渲染模块（按现有架构调整）；训练和直接推理共用，同角色模板和 stop。对单人用原生 messages；群聊在原生 user 中有明确且短的多说话人别名，连续客服回复作为 assistant。每个 generation opportunity 只有当时可见历史；若多个目标气泡之间出现新的观察、媒体或引用必须分开，不能把未来信息带到前面。以模型的真实 `apply_chat_template` 渲染，不自造 chat template。历史文本中角色标记、假系统指令和引用需拒绝或安全转义，不把原文元数据变成可信权限。

B. 目标 token 验证：每个 source bubble 对应可核实的正文 span，明确分隔符/EOS 的监督策略；`labels` 只监督实际指定客服正文和约定的结束符，其他人、旧客服、system、padding、媒体占位等为 -100。用真实本地 tokenizer 校验逐位 token/span、causal shift、PAD=EOS、短句、重复文字、中文/emoji、长段落与气泡分隔符。若模板跨边界导致无法证明标签，拆样/拒绝，不使用字符近似或把全部 assistant 设为 true。多轮联合与独立机会回放的训练/推理 token prefix 一致；不得重复监督同源目标。

C. 发布及采样：现有 `lm_trainer.corpus.v1` / `lm_trainer.release.v1` 等契约原样兼容（以实际源码为准），新的会话结构如需新正式版本，应独立定义并验证。附件的 manifest 是 synthetic fixture 清单，不是训练授权；候选记录保持 training_authorized=false、split=unassigned。可在独立测试目录生成明确的 `scope=fixture, fixture_only=true, training_authorized=false` 的 trainer fixture release，按现有 fixture 契约设置测试批准字段及仅用于测试的 `permissions.train=true, split=train`，产物保持 fixture_only。严禁将真实候选/生产记录通过此方法伪造批准。正式 release 仍要求独立审批、逐条权限及分区、hash、来源闭包，source/group 同时覆盖训练与评估；任何冲突拒绝。默认只选择连续/精确适配候选，经 fixture 授权进入测试；按真实目标 token 统计来源、有效预算和 EOS，而不是窗口数。CPT 按原作者连续块组织，不跨他人/媒体拼接或把事件指针当文本。

D. 训练器回归：保持 legacy、双人、旧业务样例、CPT/SFT、LoRA artifact 正确性。现有 HF 作为基线，验证同一 adapter 阶段继承、冻结基座、真实有效 token 梯度、断点和保存重载；Unsloth 只有实际环境验证 parity 后放行，不删除 guard 冒充支持。无现成模型或真实 tokenizer 时，运行可行的 CPU/mock/静态测试并明确阻塞；不下载大权重、不使用真实生产数据、不启动正式训练。禁止把测试人工授权或 fixture 产物发布为正式模型。

## 6. 反例测试必须实际编写

除附件正例，复制 fixture 到独立临时目录，构造以下负例并自动测试（不要改变交付原件）：

1. 篡改 target_mask、body_available 或 block IDs，伪造 protected 内容；输入拒绝或隔离且明确原因。
2. 未解析 quote 伪造 anchor/body；目标 own relation/未来 owner/全窗 speaker roster 注入第一个前缀；验收必须失败。
3. 把 joint 两个单窗也放入默认路由；同一 source target 重复监督应失败。source group 同时放 train/test；缺少部分必要源分配也应拒绝正式发布。
4. 声称同昵称是同一身份、将所有非目标人合成一个 user、错误地跨结构事件合并目标；token trace 必须能定位错误。
5. 目标为空/只有 EOS、模型 max length 不足、未知 schema、tokenizer special 控制符夹在正文、同一人既有旧历史又有新目标；不得静默修造训练样本。
6. 修改 source_fixture 文件和哈希、缺正式 approval_ref、将 fixture artifact 当生产模型发布；门禁拒绝。缺实际 tokenizer 时只能报告 mock，不声称真 tokenizer 通过。

交付实际代码 diff、接口/版本文档、可复跑 fixture 测试、真实 tokenizer 测试（若本地已有）、聚合统计和 BLOCKED 项。模型输入可用完全虚构的 token trace 进行验证；不得收集真实客户、账号、日志或业务凭据，不擅自联网、push 或启动生产机器人。开发目标是通用能力，不以增加框架功能数量验收。最强反例是群聊字段在 JSON 存在，但模型 token 输入仍把多人合为一人，或联合监督无意引入未来上下文；任一失败即不得合入正式路径。
