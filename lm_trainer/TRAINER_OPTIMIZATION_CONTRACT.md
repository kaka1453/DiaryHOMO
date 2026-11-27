# 阶段二 · Trainer Optimization Contract（任务 B）

## 数据入口及标签

- 数据入口仍是 `legacy`、`fixture`、`release`。结构化 SFT 只有显式注册的 `conversation.stage3.structured.v3` / `lm_trainer.release.v2`，由任务 A 的门禁和 renderer 处理。v2 目前只处理结构化 SFT。CPT 保留既有 v1 受审文本入口；示例以任务 A `author_blocks` 切作者连续文本，使用与 SFT 分开的虚构历史和笔记，不把媒体、引用指针或他人消息拼进 CPT 文本。
- SFT 消费 `Encoded.{input_ids,labels,spans,focus_ids,prefix_ids}`，不按文本查找目标、不推测角色或重建模板。`token_audit` 按渲染器提供的连续 token 坐标检查每一位：只有 `target_body` 和按配置选中的一个原生原子 `target_stop` 可监督，body source 顺序必须等于 `focus_ids`。`target_separator`、template、历史、元数据和 padding 一律不入 loss。模型不接收 `CausalBatch.trace`。任务 A 负责原生模板／独立时点推理前缀证明和 release routing 唯一性；任务 B 不替代该证明。
- 有效预测量由 `labels[1:] != -100` 实测；非 legacy 样本首位必须是 `-100`，有效 label 必须等于对应的 input token。右填充依 attention mask 置 `-100`，不依 token ID（所以 PAD=EOS 安全）。CPT 可配置 BOS/EOS；无 BOS 时首正文不可预测。默认拒绝 EOS-only，显式允许时计数并报告。所有版本继续不支持未经证明的 packing。
- `--validate-release` 只测结构、权限、hash、分区和路由，不加载 tokenizer/权重。`--simulate-only` 加载本地 tokenizer，完整编码所选 release，不加载权重或创建训练输出。对于结构化数据，超长不会截断、也不会从正式计划静默丢弃：同次编码扫描累积超长**窗口/文档**按源与原因计数，并拒绝整个计划；其他无法证明的标签或因果错误立即拒绝。所有报告保持 `fixture_only`。
- Markdown 示例工具只接受已知的虚构 JSON 投影，并用独立的合成源事实重新核对。文件文字、候选的 `training_authorized=false`、可读性均不代表批准；投影不作为通用 Markdown 训练入口。只在新 fixture 目录独立物化：默认 joint 和替代 primary 是两个 release，从不合并；开启路由变体需重新通过原门禁。review/opening/continuation 仍未进入默认计划。

## Optimizer 边界与数值口径

令 microbatch 的 shifted 非忽略数为 `n_i`，原生 HF mean causal NLL 为 `m_i`。单进程单卡的一次实际窗口优化 `sum(m_i*n_i)/sum(n_i)`；逐 microbatch 反传 `m_i*n_i/sum(n_i)`，Accelerate 累积固定为 1。尾窗按实有 token 重新归一。没有把 token 预算或梯度除两遍。测试用真实本地随机 Qwen2 FP32，逐 token 以 `logsumexp(logits)-logit(label)` 手工计算、比对 loss、梯度、更新后的逐参数值（loss abs 1e-6；梯度/参数 atol 2e-7, rtol 2e-5）。

零目标窗不反传/更新；真实预处理也拒绝零目标样本。FP16 先让 AMP unscale；若 GradScaler 跳过，清梯度、失败退出，不推进 scheduler 或提交本窗口的采样/预算，也不发布最终产物。可从此前完整、已验证的 optimizer 边界重启；不自动重复溢出窗口或冒充成功。每次调用可能有 DataLoader 的预取游标；恢复和持久化以**成功更新**计算的游标为准，且阶段训练在 forward 前比对计划中的批顺序。

## 采样与统计

- 既有 `target_loss_token_share` 保留：自然分布或 CPT 显式各源份额。v3 SFT 可选 `structured_sft_loss_token_share`，只对已经过相同门禁的所有选定来源生效；不走旧 SFT 的显式 mix 接口。按 manifest `files[].source` 可区分 single-person、group、joint／primary、不同语域或来源；份额必须为正数、和为 1、键与已选源精确匹配。默认不设置，仍用全量已批准自然分布。分层命名不能代替 source closure，更不能把替代版复制成新来源。
- 采样计划原子单位为完整样本，成本为 shifted 有效 token。保留 `token_budget`、`max_sample_repeats`、`max_effective_epochs`；加权/受限计划用确定性虚拟 token finish 排序，按 caps 和预算执行，不做背包最优/精确配比分配。零可行样本直接失败。无法填满的普通预算按原契约报告余量及原因。对严谨实验可设 `require_exact_token_budget: true` 和/或 `sampling_max_share_error: 0.05`（份额绝对误差）；检查整个 `max_steps` 截止后的实际投影，当前确定性计划无法达标时预检拒绝，不宣称数学上所有排列均无解。
- `data_plan.statistics` 是分词前记录／字符，`data_plan.token_statistics` 是实际 tokenizer、shifted labels 和分布（p50/p90/p99/max、token、目标 ID 数、事件组数、各 format）。`tokenizer` 记录 class、Transformers/tokenizers 版本、fast tokenizer hash 和模板 hash；v3 另有 renderer/stop/template identity。
- `planned_sampling` / `actual_sampling`：按源 `unique_loss_tokens` 为已选视图唯一有效 token 基数 `U`（去重复目标由 release 门禁负责），`requested_loss_tokens` 是目标份额对应的预算，`sampled_loss_tokens` 是该报告中计划／实际成功消费的 token 数。`actual_share=p`，`effective_epochs = T*p/U`。另记录 `share_error`、独立事件组数、样本重复次数/最大 repeats、按目标 source 的最大 repeats、正文/EOS/分隔/输入 token、最后一窗、partial 窗、预算余量、排除计数和停止原因。CPT 与 SFT 分阶段独立计数；同一 release 中的跨目标类型 overlap 仍由 schema 报告。两个独立 release 不把全局重合数当作零：应核对审批的 source index、manifest/group 账本；持久化 checkpoint 内有按源的去标识计数，公共聚合不含正文和明文源 ID。

## 生命周期与兼容

| 模式 | HF 语义 |
|---|---|
| `new` | 本地基座按内容指纹校验；实际目标模块、PEFT rank/alpha/dropout/bias 和 A/B dtype 校验；staged HF 仅允许 FP32 的 LoRA A/B 可训练，base 冻结。 |
| `continue_stage` | 只加载已核验的 inference final；原 LoRA 结构、权重可训练，不 merge、不重注入，不继承优化器；本阶段独立 seed、采样、scheduler、预算。输出不能覆盖父运行，保存前复验父和基座。 |
| `resume` | checkpoint 包含原 adapter、optimizer、scheduler、AMP scaler、Python/NumPy/CPU/CUDA RNG、成功更新游标及窗口日志；重新验证 base/tokenizer/policy/renderer、数据/采样、源代码依赖环境和 schedule。最终发布仍与 checkpoint 不同类型。 |

`lm_trainer.adapter-artifact.v1` 仍兼容旧推理产物，沿用 `verify_artifact`、`base_fingerprint`、`tokenizer_hash`、`policy_identity`、文件 inventory 和 registry fixture 门禁。新增可选 `data_lineage`（携带 release ID、parent version、manifest hash）；拒绝已知同谱系的不同版本（同 ID 改 hash、直接继承和共同非空根）。旧产物缺完整谱系不做无证据的跨源判断，不代替正式版本台账。v3 adapter 继续绑定任务 A `conversation_renderer`。训练 `state` payload 私有版本升级至 2；旧源码/状态的 checkpoint 不跨版本 resume，旧 verified final 不受此变化影响；需用原代码/环境独立恢复旧 checkpoint，不能降级为只加载权重的 resume。

## HF / Unsloth Capability Matrix（本轮）

| 能力 | HF | Unsloth 独立锁环境 |
|---|---|---|
| 既有 legacy / v1 `new` | CPU tiny 和原回归通过 | 保留旧入口，依赖导入检查通过；本轮未验证真实 loss/梯度/吞吐，不作为任务 B 正确性路径 |
| v3 多气泡 new | CPU tiny 模型 + 本地真实 tokenizer 分别验证 | **拒绝**（未证明 parity） |
| `continue_stage` / `resume` | CPU FP32 验证；真实 CUDA FP16/NF4 本轮未运行 | **拒绝**，未移除任何 guard |
| GPU 吞吐、显存、低精度 parity | NOT RUN（未授权） | NOT RUN（未授权） |

实测性能报告区分全程 wall time、每次尝试预热后的同步窗口 wall time、有效 loss/input token/s、主进程 process CPU 时间（不含 loader worker CPU）、loader 可见等待（含首次迭代/组窗开销），以及 CUDA 峰值 allocated/reserved（CPU 为 null）。记录单次模型加载、tokenizer 预处理、量化/精度、gradient checkpointing、microbatch/accumulation、KV cache 状态和成功步数；已有 dev hook 的分词仅在本阶段内存中做一次，不写持久缓存。未做任何 fused/packing 重写或凭旧 GPU 数据宣称提速。
