# Task B｜验证与 GPU 工程基准

**范围：仅脚本生成或仓库已有的虚构机器视觉客服数据。** 未使用真实工单、未下载模型、未修改依赖锁或驱动。模型输出、loss 和以下吞吐只说明工程链路，不说明业务效果。

## 可复跑命令

从仓库根目录运行。所有输出写到新目录；复跑请新建 `w`。如已有环境，始终用 `uv ... --no-sync`。

```sh
mkdir -p .cache
w=$(mktemp -d "$PWD/.cache/task-b-repro.XXXXXX")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' uv run --no-sync -m examples.vision_support.task_b_workload --output "$w/example"
export CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true
uv run --no-sync -m pytest -q lm_trainer/tests
for mode in --dry-run --validate-release --simulate-only; do
  uv run --no-sync -m lm_trainer --config "$w/example/stages.yaml" --model-config "$w/example/model.yaml" "$mode"
done
uv run --no-sync -m lm_trainer --config "$w/example/stages.yaml" --model-config "$w/example/model.yaml"
uv run --no-sync -m lm_trainer --config "$w/example/sft_new.yaml" --model-config "$w/example/model.yaml"
uv run --no-sync -m lm_trainer --config "$w/example/cpt.yaml" --model-config "$w/example/model.yaml" \
    --output-dir "$w/paused" --stop-after-updates 1
uv run --no-sync -m lm_trainer --config "$w/example/cpt.yaml" --model-config "$w/example/model.yaml" \
    --output-dir "$w/resumed" --init-adapter resume --resume-from "$w/paused/checkpoint_1"
unset CUDA_VISIBLE_DEVICES ACCELERATE_USE_CPU
```

RTX 3090 冒烟（独立解释器、独立缓存；`--legacy-baseline-root` 可选，指向 Task B 开始前的 Task A 快照，复原方式见[变更报告](TASK_B_CHANGES.md)）：

```sh
LM_TRAINER_CUDA_SMOKE=1 uv run --project envs/unsloth --no-sync -m pytest -q envs/unsloth/tests/test_task_a_loss.py
uv run --no-sync -m examples.vision_support.task_b_benchmark --level L2 \
  --work-dir "$w/rtx3090" --lengths 512 1024 2048 --steps 4 --warmup 1 --repeats 1 \
  --unsloth-python "$PWD/envs/unsloth/.venv/bin/python" \
  --legacy-baseline-root "$task_a_baseline"
```

有合法的本地未量化基座时，才执行以下 L3/L4 命令。由 `--local-base` 明确选择现有模型（不拉取权重）；可补 `--stop-token`、`--disable-thinking`、`--rank`、`--target-modules` 以匹配原生模板/架构。先不传 `--unsloth-python` 验证 HF，再用新的工作目录做双后端对比。L4 应选单张 RTX 3090（工具拒绝尺寸不符合对应级别的权重）。

```sh
uv run --no-sync -m examples.vision_support.task_b_benchmark --level L3 \
  --work-dir "$w/l3-hf" --local-base "$local_4b_or_7b" \
  --lengths 512 1024 2048 --quantization 4bit --warmup 4 --steps 20 --repeats 3
uv run --no-sync -m examples.vision_support.task_b_benchmark --level L4 \
  --work-dir "$w/l4-hf" --local-base "$local_14b" \
  --lengths 512 1024 2048 --quantization 4bit --warmup 4 --steps 20 --repeats 3
```

## 验收矩阵（正确性优先）

| 项目 | 状态 | 命令/观察 |
|---|---|---|
| B01 单阶段 CPT / SFT | PASS | CPU pytest `test_b01_b03…`：两目标都跑真实 Qwen2 forward/backward；loss 有限、LoRA 更新、冻结基座指纹不变；独立 dev 按来源统计。 |
| B02 CPT→SFT | PASS | CPU pytest `test_b02…` + 两阶段 CLI；单 adapter、LoRA 起点等于父产物，独立 optimizer 第一步和 scheduler/warmup，父 SHA 不变。另实测 BF16 基座不舍入 FP32 LoRA。 |
| B03 直接 SFT | PASS | CPU pytest 的同一个 `train()`；GPU 矩阵 `direct_sft`。 |
| B04 同阶段 resume | PASS | `test_b04…`：两 worker、dropout、暂停于 epoch 中/尾，连续 7 步 vs 暂停恢复；真实批次/游标、optimizer、scheduler、RNG 和最终权重逐元素精确 (`atol=rtol=0`)。另以新解释器 CLI `1+3` 步验证。 |
| B05 重载与原子发布 | PASS | `test_b05…`：重载前后 logits 精确；暂存完毕后模拟异常也不发布 final；已发布目录不受影响。提供的两份 fixture 也实际重载过。 |
| B06 拒绝不合法输入 | PASS | `test_b06…`：基座实际权重、tokenizer、模板、LoRA、量化、损坏父文件、权限/预算、换超参恢复、已存在输出、非有限梯度与 test split 的负例。 |
| B07 跨后端 | PASS | 独立进程 CUDA `test_task_a_loss.py` 两个 fused 路径都与 causal-shift 参考一致；GPU 矩阵另外逐 update 比对实际 ids/labels/masks、采样顺序和 token 预算。Unsloth `continue_stage/resume` 明确不支持，HF 从父阶段开始可用。 |
| B08 回归 | PASS | 根环境 `uv run --no-sync -m pytest -q lm_trainer/tests`；独立 Unsloth import smoke 与两种 CUDA loss 路径；旧 CLI/配置/可运行单适配器路径。 |

## 口径与边界

- 机器：RTX 3090 24 GiB，驱动 580.178.04、CUDA 13.0；GPU 同时承载桌面，未抢占现有进程。HF：PyTorch 2.14.0+cu130、Transformers 5.17.0、PEFT 0.21.1；Unsloth：2026.9.12、Torch 2.12.1+cu130、Transformers 5.5.0；两套 Python 3.13.15。完整包/代码/lock SHA 在机器报告。小矩阵 `microbatch=1/accumulation=2`、rank 8、AdamW、standard gradient checkpointing，同一个离线基座和显式序列化 fast tokenizer；HF 使用确定性算法，Unsloth 内核、实现及依赖不同。旧入口对照控制每个后端内的前后版本，不把旧入口 HF/Unsloth 的 tokenizer 实现差异当纯后端差异。
- 记录 `sum(actual)/sum(measured_seconds)` 的有效输入 `attention_mask`、causal shift 后目标 token、样本各自吞吐；每 update 包含 DataLoader 等待、forward/backward/optimizer 及 CUDA 同步，排除模型加载、预处理、保存和 dev；这些分别记录。每阶段重置显存峰值，allocated / reserved 分列。随机 Qwen2 过小、一次重复且仅三次计入测量的 update；**不是稳态、容量边界或加速排名**。冷进程编译、共享 GPU 负载造成波动；不据此认定或排除 >10% 回归。若目标模型稳态复跑出现 >10% 差值，先按预处理、batch 等待、CUDA 内核、保存分别定位，不修改 mask 或 loss 归一化。未测字段为 `null`。
- CPT 长短样本、变长排查笔记；SFT 有稀疏和密集目标轮，定长上限以真实 tokenizer 的 token 数校准，无人造 padding 对齐吞吐。真正训练逐源实际长度和统计见聚合 JSON。14B 的 512 诊断不是业务语境下的容量证明；基准剪短或 model/tool 不兼容时不会静默截掉 SFT 目标。

## 分级实测结果

<!-- TASK_B_RESULTS -->

| 层级 | 状态 | 范围 |
|---|---|---|
| L1 | **PASS** | CPU 全量 77 通过，两项 CUDA opt-in 在此命令中跳过。Task A 基线为 65 通过。 |
| L2 | **PASS（冒烟）** | CPU 两阶段、直接 SFT、精确 resume/重载；RTX 3090 的 19 个矩阵用例通过，另有两项 CUDA FP16/GradScaler 精确恢复测试通过（含 NF4 QLoRA）。随机 Qwen2 序列化参数约 0.36M。 |
| L3 | **NOT RUN** | 未发现合格的本地 4B/7B 档模型，不下载。 |
| L4 | **NOT RUN** | 未发现合格的本地 14B 模型；27B 不纳入。因此容量、显存上限与 OOM 边界为未测。 |

下表为 L2 单次冒烟（BF16、未量化，1 次预热 + 3 次测量）。全部 19 个用例、每步 token/损耗、分来源实际量、显存与各阶段时间见[机器可读 JSON](benchmark_results.json)。

| 长度上限 | 负载 | 后端 | 输入 token/s | 目标 token/s | 样本/s | alloc / reserved MiB |
|---|---|---|---:|---:|---:|---:|
| 512 | 混合 CPT | hf | 3,423 | 3,407 | 15.4 | 77.0 / 82.0 |
| 512 | 混合 CPT | unsloth | 2,313 | 2,303 | 10.4 | 20.8 / 26.0 |
| 512 | 多轮 SFT | hf | 5,338 | 1,379 | 16.0 | 77.1 / 90.0 |
| 512 | 多轮 SFT | unsloth | 3,512 | 907 | 10.5 | 20.8 / 26.0 |
| 1024 | 混合 CPT | hf | 4,468 | 4,458 | 10.2 | 77.0 / 94.0 |
| 1024 | 混合 CPT | unsloth | 15,853 | 15,816 | 36.3 | 23.6 / 28.0 |
| 1024 | 多轮 SFT | hf | 9,986 | 2,508 | 14.7 | 89.4 / 108.0 |
| 1024 | 多轮 SFT | unsloth | 7,518 | 1,888 | 11.1 | 23.6 / 28.0 |
| 2048 | 混合 CPT | hf | 10,033 | 10,019 | 14.4 | 89.3 / 100.0 |
| 2048 | 混合 CPT | unsloth | 25,395 | 25,358 | 36.4 | 29.2 / 34.0 |
| 2048 | 多轮 SFT | hf | 20,678 | 5,212 | 15.2 | 115.0 / 164.0 |
| 2048 | 多轮 SFT | unsloth | 13,835 | 3,487 | 10.2 | 29.2 / 34.0 |

- **混合覆盖**：初次短跑在 1024/2048 的 chat 比例 0.35 时没有消费长 notes，未将该组认作混合成绩。表中混合和两阶段来自各长度相同的 0.05/0.95 复跑，查验两个来源均实际消费。小型 tokenizer 探针确认 notes 样本可达到 496 / 1009 / 2035 输入 token。基准只测频次和时延，不推荐此比例用于业务调优。
- **跨后端（B07 PASS）**：6 对 CPT/SFT 用例实际 ids/labels/attention mask、采样顺序和 causal 目标 token 完全一致；第一步加权 loss 的最大绝对差 `0.000141`（阈值 `0.03`），独立 fused NLL 参考测试低于其独立容差。初测见到依赖版本所用 tokenizer 默认规则不同，复跑已显式固定序列化 fast tokenizer；未将依赖、优化 kernel 或确定性设置的差异算作纯后端加速。
- **Legacy 512 同后端基线 → Task B**：hf 1,377 → 1,377 输入 token/s（+0.0%）；unsloth 1,069 → 1,089 输入 token/s（+1.9%）。仅三个测量窗口，不作稳态回归定论；legacy 不进入 Task B 推理注册表。
- **CUDA 扩展正确性**：FP16 未量化及 NF4 双量化在 RTX 3090 实测 scaler、RNG、采样器、optimizer/scheduler、最终 LoRA 均精确（`rtol=atol=0`），NF4 通过续训后的 CPT→SFT 继承。首次 NF4 检查捕捉到 bitsandbytes 首次 forward 会改变冻结 bias dtype；修正为在初始化时固定 compute dtype，再取权重/量化状态指纹，重新验证无训练期变动。复跑：`LM_TRAINER_TASK_B_CUDA=1 uv run --no-sync -m pytest -q lm_trainer/tests/test_task_b.py -k cuda_fp16_scaler_resume`（确保 `CUDA_VISIBLE_DEVICES` 未设为空，并取消 `ACCELERATE_USE_CPU`）。

## 未验证/不支持

- L3/L4 对应的本地未量化中文 Instruct 权重未发现：本地 Hugging Face 缓存仅可用微型模型及排除的 27B 所在目录；常用离线模型目录还有不在目标档位的模型。未下载、未改驱动，不跑 27B。**没有 14B OOM 边界、显存容量或业务效果结论。**
- 未做三重复稳态验收、真实大模型原生模板和大模型量化/继承评估、8-bit 精确 resume、跨硬件 Bitwise 保证、业务推理运行时或前端。锁定版 Unsloth 不支持 Task B 的跨阶段继承与完整 checkpoint resume，未把仅权重加载标作 resume。
