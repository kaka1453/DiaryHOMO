# 阶段二任务 B · Markdown 合成验证 / A-B 配置示例

从仓库根运行。配置默认 **fixture + 本地随机 tiny Qwen2 + CPU**，不表示模型质量。没有正式训练、发布或线上机器人行为。不要把这里的生成脚本用于生产数据。

```bash
export CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
PY=.venv/bin/python
MD='examples/1/Intelligent_Support_Groupchat_Data_Test.md'
D="$PWD/.cache/phase2-b-replay-$(date +%s)"  # 必须不存在
$PY -B -m examples.phase2_task_b.prepare "$D" --markdown "$MD" --with-tiny-base

# 静态验证；也可换成 --dry-run 打印同一组阶段数据计划。
$PY -B -m lm_trainer --config "$D/direct_sft.yaml" --model-config "$D/model.yaml" --validate-release
$PY -B -m lm_trainer --config "$D/mixed_cpt_sft.yaml" --model-config "$D/model.yaml" --validate-release
$PY -B -m lm_trainer --config "$D/direct_sft.yaml" --model-config "$D/model.yaml" \
  --manifest-path "$D/primary/manifest.json" --validate-release

# 可选：已有真实本地 tokenizer 才运行。--simulate-only 不加载权重。
Q="${LOCAL_QWEN_MODEL_DIR:-/path/to/models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}"
$PY -B -m lm_trainer --config "$D/qwen_direct_sft.yaml" --model-config "$D/model.yaml" \
  --model-name-or-path "$Q" --simulate-only
$PY -B -m lm_trainer --config "$D/qwen_direct_sft.yaml" --model-config "$D/model.yaml" \
  --model-name-or-path "$Q" --manifest-path "$D/primary/manifest.json" --simulate-only
$PY -B -m lm_trainer --config "$D/qwen_mixed_cpt_sft.yaml" --model-config "$D/model.yaml" \
  --model-name-or-path "$Q" --simulate-only

# BASE 不训练；可先生成 base-only 推理对照注册表。
$PY -B -m examples.phase2_task_b.model_registry "$D"
# DIRECT_SFT：自然分布，初始化同结构的新 LoRA。
$PY -B -m lm_trainer --config "$D/direct_sft.yaml" --model-config "$D/model.yaml"
# MIXED_CPT_SFT：CPT 60/40 是目标份额，实际受完整文本和最大重复限制；继承同一 adapter。
$PY -B -m lm_trainer --config "$D/mixed_cpt_sft.yaml" --model-config "$D/model.yaml"
$PY -B -m examples.phase2_task_b.model_registry "$D" --include-adapters

# 验证模拟输出和盲评管线；不会据此给模型打效果分。
$PY -B -m lm_trainer.eval validate --cases "$D/eval-dev.json"
printf '%s\n' '{"BASE":"请先明确故障现象。","DIRECT_SFT":"请报告设备编号。","MIXED_CPT_SFT":"先记录现象；还不能确认根因。"}' > "$D/mock.json"
$PY -B -m lm_trainer.eval run --cases "$D/eval-dev.json" --roles "$D/roles.json" \
  --mock --model BASE --model DIRECT_SFT --model MIXED_CPT_SFT --mock-responder-file "$D/mock.json" \
  --memory-conditions M0,M1,M2,M3,M4 --cards examples/vision_support/task_d_eval/cards_m0_m4.json \
  --output "$D/eval-mock"
$PY -B -m lm_trainer.eval blind --run "$D/eval-mock/run.json" --pair DIRECT_SFT MIXED_CPT_SFT \
  --output "$D/blind"
```

在 `blind_pack.json` 选 `item_id`，不查看私有 `blind-map.json`，在新文件 `review.json` 按下面格式填写，然后调用 `python -m lm_trainer.eval review --pack ... --reviews ... --entry ...`：

```json
{"item_id":"替换为实际 item_id","choice":"tie","scores":{"participant_grounding":3},"dimension_reasons":{"participant_grounding":"两边均未分清引用对象。"}}
```

`reveal --pack "$D/blind/blind_pack.json" --map "$D/blind/blind-map.json" --reviews "$D/reviews" --output "$D/reveal.json"` 生成含分维理由和场景的私有揭盲结果；`report --run "$D/eval-mock/run.json" --reveal "$D/reveal.json" --output "$D/report"` 输出分场景统计。对三组做独立成对比较（同一 seed、三组各一份私有 map），不以 loss 代替盲评。也可用 `eval run --models "$D/models-ab.json" --roles "$D/roles.json" --cases "$D/eval-dev.json" --allow-fixture --device cpu --output "$D/eval-tiny"` 验证本地随机权重推理及 adapter 重载；随机 tiny 输出不作客服质量依据。

正式 A/B 设计须先固定一份批准的本地中文 Instruct 基座 revision/字节、tokenizer、原生模板/renderer 和短系统提示卡。SFT 两臂固定相同数据、路由、LoRA 结构、seed、派发次序及有效 SFT token 预算；CPT 的来源、有效 token、耗时单列。可从 60/40 开始并按可用量调整，不能靠小源反复过采样凑比例。已准备的例子可检查两臂的 `batch_tensors_sha256` 与 `batch_order_sha256`。4B/14B、r16/alpha32、CPT 2e-5 / SFT 5e-5、1024/2048+ 仅是候选起点，不代表显存容量已验证。

评测 dev 内容独立编造，和训练、系统提示及上下文卡没有同源组依赖。多人引用样例使用任务 A renderer 的已观察群聊 user turn；旧多轮评测仍采用固定 transcript/rollout，**不是**结构化生产环境的等价替代。正式 dev/盲测须用独立审批的全局 train/dev/test source/event 台账，禁止将评测原文或 rubric 放进训练/提示卡；现有独立评测工具仍标记为 `fixture_only`，没有开放正式或生产数据路径。
