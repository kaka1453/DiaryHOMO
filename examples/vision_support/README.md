# 通用多阶段训练数据结构示例｜虚构机器视觉客服版

**用途：通用训练器接口开发与合成单测。** 全部聊天、工作笔记、身份、消息 ID、来源指针均为从零编写的虚构测试数据，不来自任何真实业务对话、内部文档或实际公司客服记录。内容只用于展示数据结构，不是机器视觉设备操作规范或真实故障诊断。

采用统一的 `corpus.v1` 语料封装格式（训练器实现时应支持通过配置指定或兼容期望的 `schema` 标识，不要将单一字符串写死在核心校验逻辑中）；虚构角色 `CS_AGENT`、`CUSTOMER` 用于验证训练器身份可配置。虚构 `source_id` / hash / 平台字段不指向实际文件，不得用于验证真实文件完整性。生产权限默认关闭，测试授权示例另列。

## Task A：可以直接预检的虚构训练包

`task_a_fixture/` 是为 Task A 独立生成的 **fixture-only** 包：使用专用的 `sft_approved_test.jsonl` 与虚构 CPT 文本，不修改下方任何待审批候选。用 `train_cpt.yaml`（混合 chat/notes）或 `train_sft.yaml`（多轮选定回复）做预检、模拟；旧配置例仍为 `../industrial_support/train.yaml`。复跑命令、最小测试和真实 tokenizer 逐 token 结果见 [Task A 验证报告](TASK_A_VALIDATION.md)，数据格式见 [契约](../../lm_trainer/TASK_A_CONTRACT.md)。

## 1. 文件组织

```text
examples/vision_support/
  chat_cpt.jsonl                     # 客服/客户的模拟发言块
  notes_cpt.jsonl                    # 虚构客服工作笔记，模拟长文本 CPT
  sft_dyadic.jsonl                   # 双人多轮对话
  sft_multi_identity_candidate.jsonl # 多人候选（默认隔离）
  source_index.jsonl                 # 单消息来源和裁决
  sample_source_index.jsonl          # 样本到来源的映射
  event_proposals.jsonl              # 暂定事件组
  sft_approved_test.jsonl            # 合成授权测试样例（仅供隔离单测）
```

实际项目可按角色分目录；上述文件命名用于工程接口联调，不代表正式生产发布包。

## 2. chat_cpt.jsonl — 聊天 CPT

每行是独立发言块。两条记录分别模拟客服和客户的表达；客服模型只能筛选自己的 `target_person` 数据。客服两条气泡通过换行 `\n` 归并。

## 3. notes_cpt.jsonl — 长文本 CPT

用虚构的客服工作记录模拟长文本 CPT 的结构；同样只读取 `text`，不添加虚构问答角色。

## 4. sft_dyadic.jsonl — 双人 SFT

- **客服视角（`DEMO-S001`，目标 `CS_AGENT`）**：第一轮 assistant 仅作为历史上下文（`target_mask[0]=false`）；第三、第五、第七个 `messages` 元素才参与正文监督。客户输入保留但不监督。
- **客户视角（`DEMO-S002`，目标 `CUSTOMER`）**：同样的原始消息 ID 与事件组，重新映射 `role` 和 `target_mask`，用于验证通用加载器不会把固定人名当 `assistant`。此例是隔离测试视角，不建议把客户目标混入客服训练。样本终止于最后一个已选中的客户回复。

## 5. sft_multi_identity_candidate.jsonl — 多人候选

客户、现场工程师（`OTHER_1`）、客服分别说话。不能把两个 `user` 无声解释为同一个人。群聊字段结构与双人相同，当前通过文件分类隔离。

## 6. 来源侧车与事件（节选示例）

- `source_index.jsonl`：展示一个原始气泡记录。实际全量应覆盖所有 `source_ids`；此处仅演示字段，未提供完整原始输入。`disposition="F"` 表示语义判定两类均适合，并非训练授权。
- `sample_source_index.jsonl`：`selected` 用 `|` 分轮，`-` 表示原始位置范围，`*` 表示 SFT 目标轮；`target_ids` 是实际被监督的原始气泡 ID，`bubble_ids` 保留每轮原始边界。
- `event_proposals.jsonl`：暂定事件组；同一事件的不同目标视角不得分到互相独立的 train/test。

## 7. sft_approved_test.jsonl — 独立的合成训练入口测试

`DEMO-APPROVED-TEST-001` 为专门构造的测试 fixture，仅供隔离的 loader/单测使用，模拟预检中的 `approved` + `train` + `permissions.train=true`，不构成生产发布清单。真实候选禁止照此修改授权。若 loader 要求配合 release manifest，应使用单测专用的虚构清单与临时目录，不能关闭正式门禁。

## 训练器接入约定

以下区分**候选数据格式**与**正式发布门禁**，开发时应先核对现有 `lm_trainer/` 代码再集成。

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `schema` | string | 示例值为 `corpus.v1`（建议支持配置化 `schema` 标识） |
| `id` | string | 样本唯一 ID；不是单条原始消息 ID |
| `kind` | string | `cpt` 或 `sft`，显式区分目标函数 |
| `target_person` | string | 当前样本的目标发言者；角色标识应可配置，不将具体角色写死到通用训练器 |
| `event_group` | string | 暂定事件组或正式事件组；跨人物/来源/派生样本共同约束分区 |
| `source_index_ref` | string | 来源侧车定位：`文件名#样本ID`；不进模型输入 |
| `permissions` | object | `local_review`、`train`、`external` 为不同用途授权，不互相推导 |
| `review_status` | string | 候选实样通常为 `pending_owner`；合成单测可使用 `approved` |
| `split` | string | 候选实样为 `unassigned`；正式 `train` 须另行授权和发布 |
| `text` | string | CPT 独立作者发言块或原文文档；与 `messages` 互斥 |
| `messages` | array | SFT 原始语义轮；每轮含 `speaker`、`role`、`content`、`source_ids` |
| `target_mask` | boolean[] | 按 **messages 轮**标记，不是 token mask，长度必须等于 `messages` |

### 监督语义（P0）

- **CPT**：读取 `text`，采用标准 causal LM next-token loss；独立文档边界、BOS/EOS、padding 按实际 tokenizer 验证。不能把其他样本拼成无边界独白。保留旧 `text` 数据入口，不静默把 SFT 当 CPT。
- **SFT**：使用基座原生 chat template；训练不追加空的生成提示（`add_generation_prompt=False`）。仅 `role=assistant` 且 `speaker=target_person` 且 `target_mask[i] == true` 的正文可监督；其他角色、历史 assistant、padding、身份前缀、模板头、内部推理等 labels 为 `-100`。建议通过显式配置区分“仅正文监督”与“含目标轮结束符（EOS）监督”两种策略，并做逐 token 对照验证。
- 不能把 `target_mask` 直接当成 token mask 或 `attention_mask`。使用真实 tokenizer 和模板确定正文 span；验证 special token 边界、换行、中文、emoji、重复内容、历史 assistant 开头、连续同 role、左右 padding、超长和零目标。超长样本建议显式拒绝或按完整轮策略处理，不静默截掉目标回复。
- `speaker` 与 `role` 不等价。普通原生模板可能忽略 `speaker`；多人候选必须有经验证的人物标识序列化（例如只在上下文添加稳定作用域别名），默认不加载。`OTHER_1` 等临时别名只在单样本内有效；稳定 ID 留在侧车索引中。
- `source_ids` 是当前轮按原顺序合并的原始气泡 ID 数组。允许历史复用，但正式 SFT 同一目标消息不可无意重复监督；CPT/SFT 跨阶段复用需单列统计。原始 ID、权限、审核理由、事件分组和目标 mask 均不是训练正文。

### 正式训练门禁（建议实现/核查）

1. `kind`、schema、字段类型、非空目标、合法 role/mask、目标身份及 provenance 校验失败均显式报错。不要通过改写候选样本权限绕过检查。
2. 实际训练还必须校验用户授权、已批准的 release/manifest/hash、split、测试保护和资产目录。样本自报 `train:true` 不足以授权真实训练；缺失/未知权限默认拒绝。多人候选不自动并入双人数据。
3. 校验目标 ID 与上游数据发布的语义裁决一致；仅 CPT 或待审的来源不得作为 SFT 目标。所有同源事件及上下文重叠按分区约束处理，不能随机按单条消息切分。
4. 实验配置、tokenizer revision、模板 hash、目标 EOS 策略、实际有效监督 token、数据版本及父 adapter 留痕。不要在共享或调试日志中打印完整业务原文。

### 训练正确性检查重点

在验证多阶段/多长度训练链路时，需核对现有 `lm_trainer` 实现：确认梯度累积按完整累计组的总有效目标 token（因果 shift 后的 `labels != -100`）归一化而不出现重复除以累积步数；确认不同长度 microbatch 按有效目标 token 数而非样本均值加权；确认尾批更新与零目标保护正确；并测试变长、尾组、padding、单卡及支持的多卡情形。

正式投入训练前，仍需在目标基座（例如 7B/14B 级中文 Instruct 模型）的真实 tokenizer 与原生 chat template 上实测编码长度与监督边界。工程开发阶段只使用合成 fixtures。

### 建议的验收用例

| 用例 | 期待结果 |
| --- | --- |
| CPT 单条/多气泡/Markdown | 只读取 `text`，正确预测和独立边界 |
| SFT 多轮 + 历史 `assistant=false` | 历史正文全部 `-100`，仅选中正文进入 loss |
| 同一事件切换目标发言者 | `role`/`target_mask` 随视角变化，不混训目标 |
| `target_mask` 长度错误、`user=true`、目标人物错误、全 `false` | 拒绝 |
| 连续 `user`、历史 `assistant` 开头 | 原生模板支持则正确编码，否则明确拒绝 |
| 多人交换 `speaker` | 模型实际输入应变化，选中回复的正文 labels 不变 |
| 超长、padding、目标 EOS 两策略 | 目标不静默丢失；padding/非目标结束符屏蔽 |
| 未授权候选、错误 hash/split、测试保护命中 | 拒绝训练 |
| 同目标跨批重复、裁决冲突 | 阻断正式发布，记录原因 |
| 变长微批、最后不足累计步数 | 梯度与同一目标 token 集的整批参考一致 |

### Task B 生命周期和工程基准

[复跑与实测报告](BENCHMARK_REPORT.md) 包含微型随机 Qwen2 的两阶段配置、暂停/恢复命令、GPU 分级结果和限制。[Task C 消费用 fixture adapters](task_b_fixture_adapters/) 显式标记 `fixture_only`，不能用作正式客服模型。

## Task D：评测、匿名盲评与本地调试台

[Task D 验证报告](TASK_D_VALIDATION.md) 与 [虚构评测集](task_d_eval/README.md) 覆盖：headless 批量评测（`fixed_history` 与 `multi_turn_rollout` 分开统计）、单卡串行与断点续跑、匿名盲评与揭盲、M0–M4 卡片对照实验、以及仅监听 `127.0.0.1` 的 Gradio 调试台。全部数据为虚构 fixture，`fixture_only: true`。
