# Task A：训练数据契约 v1

## 数据入口

| `data_mode` | 配置 | 门禁 |
|---|---|---|
| `legacy`（旧配置默认） | `data_file_path`，每条 `{ "text": "…" }`，JSON/JSONL 单文件 | 不接受带结构化语料信封的记录；保留 tokenizer 自带特殊符和旧长度截断。缺失授权的结构化数据绝不会降级到此模式。 |
| `fixture` | `manifest_path` + 明确 `objective` | 清单 `scope=fixture, fixture_only=true, training_authorized=false, review_status=approved`；仍逐条校验 `permissions.train=true` 与 `split=train`；产物打 `fixture_only: true`。 |
| `release` | `manifest_path` + 明确 `objective` | 清单 `scope=release, fixture_only=false, training_authorized=true, review_status=approved`，并有非空 `approval_ref`；其余完整性、闭包及逐条门禁与 fixture 一致。 |

- **记录 JSONL**：`schema=lm_trainer.corpus.v1`；`corpus_schema_allowlist` 可显式接受同字段语义的 `.v1` 别名，如 `corpus.v1`，不转换未知主版本。共同必填：`id`（包内唯一）、`kind`、`target_person`（全包一致）、`event_group`、`source_index_ref`（不解引用的审计指针）、`permissions.{local_review,train,external}`（独立布尔值）、`review_status=approved`、`split=train`。`causal_text` 对应 `kind=cpt`、`text`、`source_ids`；`assistant_only_messages` 对应 `kind=sft`、`messages[{speaker,role,content,source_ids}]` 和等长的严格布尔 `target_mask`。`True` 只允许归属 `target_person` 的非空 assistant 回复。
- **清单 JSON**：`schema=lm_trainer.release.v1`、`release_id`、`target_person`、`parent_version`、上述授权字段，`files[{path,sha256,kind,source,split,records}]`、`source_group_manifest:{path,sha256}`、可选 `source_index:{path,sha256}`。全部引用须为清单目录内的字面相对路径；拒绝遍历、glob、符号链接和重复路径，以通过哈希的同一份字节解析训练文件。`--asset-root` 下清单应位于 `data/train/`。
- **闭包 JSONL**：`lm_trainer.source-groups.v1` 每行 `{schema,id,kind,target_person,event_group,split,source_ids}`。须覆盖训练记录，且可有其他视角或 `dev`/`test` 的审计条目；后者不在 `files` 中。无论视角，`event_group` 或任一 `source_id` 不得跨 split；CPT/SFT 有意复用只计数，SFT 重复目标拒绝。可选 `lm_trainer.source-index.v1` 行 `{schema,source_id,disposition,review_status}`；提供时，每个 SFT 目标须有获批的 `S` 或 `F` 裁决。训练端验证包内声明，不替代发布方对闭包完整性、批准引用真实性的保证。

## Task B：stage 与批次

- `preflight_data(runtime) -> ValidatedData(plan, samples)`（无权重/无 tokenizer）。`plan` 记录 `objective`/`sources`/`target_loss_token_share`/`token_budget`/`seed`/`template_policy`/数据 SHA 与逐源统计；CLI `--validate-release` 输出隐去系统指令正文的可机读计划（分词前无真实 token 数）。`--dry-run` 只查配置路径；`--simulate-only` 另外做本地分词与建批。分词不落磁盘缓存。
- CPT 不 packing：`add_bos=false` / `add_eos=true` 默认可显式修改；按 `labels[1:] != -100` 分开计正文与 EOS 目标。仅 EOS 的样本默认拒绝（`allow_eos_only` 可诊断）。两种结构化目标都只支持 `overlength_policy=reject`，从不裁剪目标正文。
- `load_training_dataset(..., validated=...)` 和 `build_data_loader(...)` 接收通过的方案。`CausalBatch` 的字典键仅 `input_ids/attention_mask/labels`；`.trace` 提供样本/事件/来源、输入与正文/EOS 目标计数，不送入前向。右 padding 位置才置 `-100`，真实 EOS 不因 token ID 等于 PAD 而被屏蔽。
- `natural` 无预算/上限时每轮遍历一次；配比或预算/上限启用时构造**一次性**阶段计划。按位移后有效 token 成本安排完整样本，`max_effective_epochs` 按源的累计目标 / 唯一目标定义，`max_sample_repeats` 按样本；不指定上限时从 `epochs` 派生。报告计划/消费份额、偏差与停止原因；不输出虚构逐源 loss。
- `loader.batch_sampler.state_dict()` v1：`{version,plan_key,epoch,next_batch}`；同数据、同设置的采样器可 `load_state_dict()` 后继续，此处游标是**已派发**批次，不是已提交优化器更新。需要精确断点的后续编排应在更新边界记录消费位置；`num_workers=0` 避免预取领先。本任务不保存优化器、调度器或 RNG，不恢复整个训练运行。

## Task C：共享原生模板策略

`lm_trainer.template_policy` 不导入 torch/Transformers。用 `resolve_policy(tokenizer, rule)` 获取原生模板 SHA-256、版本、角色映射、可见回复策略与 `stop_tokens/stop_token_ids`，并可用 `template_hash` 锁定版本。`chat_template_kwargs(..., for_training=True/False)` 分别设置 `add_generation_prompt=False/True`；若模板支持则显式 `enable_thinking: false`。`system_prompt` 只取显式应用配置。SFT 用官方 generation mask（如有）交叉验证逐轮对齐；以原生完整渲染和分段 token 逐位相等为放行依据。只监督选中正文与可选 `supervise_target_eos` 的已验证结束符；控制符伪造、不可对齐模板、多人、工具、非空推理块拒绝。新模板/版本应先 `--simulate-only` 验证，不使用近似 mask。
