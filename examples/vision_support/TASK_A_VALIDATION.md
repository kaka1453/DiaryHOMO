# Task A：变更与离线验证

## 基线与变更

开始时工作树干净，`HEAD` 为 `2ae5bc2223f5a142375aacb32d96ecf0b4d97311`。核对了 HF/Unsloth 环境及原有 CLI、训练、路径、模型加载、样例；`pyproject.toml`、`uv.lock`、`lm_trainer/requirements.txt` 均未改动。修改前按任务书运行 CPU 离线基线：**7 passed in 71.22s**（`uv run --no-sync` 使用现有环境）。未修改原始待审批候选。

| 文件 | 目的 |
|---|---|
| `lm_trainer/__main__.py`, `lm_trainer/config/train_config.py`, `lm_trainer/train/io.py` | 模式及路径选择、无模型数据预检、非训练命令互斥；保留旧 CLI、asset-root 约束及 Unsloth 分进程。 |
| `lm_trainer/train/schema.py` | 版本化门禁、完整性、来源闭包与旁路诊断。 |
| `lm_trainer/template_policy.py`, `lm_trainer/train/encoding.py` | 共享原生模板策略、CPT 边界、SFT 精确监督与拒绝。 |
| `lm_trainer/train/data.py`, `lm_trainer/train/sampling.py` | 预构造标签、右 padding、无元数据前向、批采样器及按有效 token 预算/曝光统计。 |
| `lm_trainer/train/engine.py`, `lm_trainer/train/schedule.py`, `lm_trainer/train/performance.py` | 使用已校验数据、更新窗与计划驱动、权重更新数学保护、报告及 fixture 标记。 |
| `lm_trainer/model/loader.py`, `lm_trainer/train/modeling.py` | 结构化入口使用本地 tokenizer 和原生模板；保持 legacy/Unsloth 导入顺序隔离。 |
| `lm_trainer/tests/test_task_a.py`, `envs/unsloth/tests/test_task_a_loss.py` | 离线 CPU 验收、真实 CUDA Unsloth fused loss 最小验证。 |
| `examples/vision_support/make_fixture.py`, `task_a_fixture/{manifest.json,groups.jsonl,index.jsonl,chat.jsonl,notes.jsonl,dyadic.jsonl}` | 可重建、哈希锁定的虚构清单、闭包、裁决和训练记录；生成器另支持负例与模拟正式发布门禁。 |
| `examples/vision_support/{train_cpt.yaml,train_sft.yaml,audit_tokens.py}` | 阶段配置与真实 tokenizer 逐位审计。旧单文件配置沿用 `examples/industrial_support/train.yaml`。 |
| `lm_trainer/README.md`, `lm_trainer/TASK_A_CONTRACT.md`, `examples/vision_support/{README.md,TASK_A_VALIDATION.md}` | 使用说明、后续 Task B/C 契约、验证证据。 |

回滚：若需要回退，先保存当前工作树补丁；根据本表仅把 Task A 修改的既有文件恢复到上述 `HEAD`，新增文件移至回收站。不要用 `git clean` 或全树 `reset --hard`；尤其不要删除已有资产、缓存或训练输出。本次不提交 Git 历史、不推送。

最终 `git diff --stat`（临时索引，相对起始 HEAD，含新增文件）：

```text

 envs/unsloth/tests/test_task_a_loss.py               |  82 ++++
 examples/vision_support/README.md                    |   4 +
 examples/vision_support/TASK_A_VALIDATION.md         | 151 +++++++
 examples/vision_support/audit_tokens.py              |  96 +++++
 examples/vision_support/make_fixture.py              | 175 ++++++++
 examples/vision_support/task_a_fixture/chat.jsonl    |   2 +
 examples/vision_support/task_a_fixture/dyadic.jsonl  |   1 +
 examples/vision_support/task_a_fixture/groups.jsonl  |   4 +
 examples/vision_support/task_a_fixture/index.jsonl   |   4 +
 examples/vision_support/task_a_fixture/manifest.json |  45 ++
 examples/vision_support/task_a_fixture/notes.jsonl   |   1 +
 examples/vision_support/train_cpt.yaml               |  28 ++
 examples/vision_support/train_sft.yaml               |  25 ++
 lm_trainer/README.md                                 |   6 +-
 lm_trainer/TASK_A_CONTRACT.md                        |  25 ++
 lm_trainer/__main__.py                               |  30 +-
 lm_trainer/config/train_config.py                    |  52 ++-
 lm_trainer/model/loader.py                           |  11 +-
 lm_trainer/template_policy.py                        | 137 ++++++
 lm_trainer/tests/test_task_a.py                      | 868 +++++++++++++++++++++++++++++++++++++++
 lm_trainer/train/data.py                             | 239 +++++++----
 lm_trainer/train/encoding.py                         | 241 +++++++++++
 lm_trainer/train/engine.py                           | 106 ++++-
 lm_trainer/train/io.py                               |  22 +-
 lm_trainer/train/modeling.py                         |   9 +-
 lm_trainer/train/performance.py                      |   4 +
 lm_trainer/train/sampling.py                         | 280 +++++++++++++
 lm_trainer/train/schedule.py                         |   8 +-
 lm_trainer/train/schema.py                           | 717 ++++++++++++++++++++++++++++++++
 29 files changed, 3253 insertions(+), 120 deletions(-)

```

## 复跑命令

从仓库根目录执行。`--no-sync` 避免安装依赖或改动环境；所有样例都是虚构数据。

```bash
export CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --no-sync python -B -m pytest -p no:cacheprovider -q lm_trainer/tests

# --validate-release 无需模型目录；可将 cpt 换成 sft。
uv run --no-sync python -B -m lm_trainer \
  --config examples/vision_support/train_cpt.yaml \
  --model-config examples/industrial_support/model.yaml --validate-release

# 只打开本地真实 tokenizer 文件，不打开权重。替换为你的本地目录。
P=/path/to/models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16
mkdir -p .cache
out=$(mktemp -d "$PWD/.cache/task-a-evidence.XXXXXX")
for stage in cpt sft; do
  uv run --no-sync python -B -m lm_trainer \
    --config "examples/vision_support/train_$stage.yaml" \
    --model-config examples/industrial_support/model.yaml \
    --model-name-or-path "$P" --checkpoint-dir "$out/$stage" --simulate-only
done
uv run --no-sync python -B -m examples.vision_support.audit_tokens --tokenizer "$P" > "$out/real-eos.md"
uv run --no-sync python -B -m examples.vision_support.audit_tokens --tokenizer "$P" --without-eos > "$out/real-no-eos.md"

# 另生成正、反例；要求目标子目录不存在。
uv run --no-sync python -B -m examples.vision_support.make_fixture "$out/mock-release" --mock-release
uv run --no-sync python -B -m examples.vision_support.make_fixture "$out/bad-hash" --negative hash
check=(uv run --no-sync python -B -m lm_trainer --config examples/vision_support/train_cpt.yaml
       --model-config examples/industrial_support/model.yaml --validate-release)
"${check[@]}" --data-mode release --manifest-path "$out/mock-release/manifest.json"
"${check[@]}" --manifest-path "$out/bad-hash/manifest.json" # 预期退出 2，SHA-256 mismatch
```

可选独立 CUDA 归约烟测（新解释器，无权重加载，仅几组张量及本地 kernel 编译）：

```bash
unset CUDA_VISIBLE_DEVICES ACCELERATE_USE_CPU
cache=$(mktemp -d "$PWD/.cache/task-a-unsloth.XXXXXX")
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  UNSLOTH_DISABLE_STATISTICS=1 LM_TRAINER_CUDA_SMOKE=1 HF_HOME="$cache/hf" \
  TRITON_CACHE_DIR="$cache/triton" MPLCONFIGDIR="$cache/matplotlib" \
  UNSLOTH_COMPILE_LOCATION="$cache/unsloth" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" \
  uv run --project envs/unsloth --no-sync python -B -m pytest -p no:cacheprovider -q \
    envs/unsloth/tests/test_task_a_loss.py
```

## 验收矩阵

| 编号 | 结果 | 证据 |
|---|---|---|
| A01 | PASS | `test_a01_*`：fixture / 虚构 release 可通；错误哈希、审批/角色/split、空源、逃逸与 legacy 夹带拒绝；真实微型 HF/LoRA 单步的中间与最终产物、性能报告均打 fixture 标记，重复输出拒绝。 |
| A02 | PASS | `test_a02_*`：跨视角事件/来源跨 split、重复目标、空来源及裁决冲突拒绝；允许 2 个跨阶段来源 ID 重叠并单列。 |
| A03 | PASS | `test_a03_*`：独立样本、原始换行及 Unicode、显式 BOS/EOS、移位后正文/EOS 数及最短样本防护；真 EOS 在 PAD=EOS 下保留。 |
| A04 | PASS | `test_a04_*`：逐位置 golden labels，官方 generation mask 与历史选择交叉验证；两种目标视角、部分/全选、连续同角色、重复正文、空白/Markdown/emoji、非法 mask、伪造控制符。多人默认拒绝。 |
| A05 | PASS | 真实本地 **Qwen3.8-27B-Quark-AWQ-INT4-W4A16** tokenizer，训练模板 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`。163 tokens、3 个被选中的回复，分别解码并与 fixture 正文比对；EOS 开/关 `59 + 3` / `59 + 0` 个目标。 |
| A06 | PASS | `test_a06_*`：多样本、非 8 倍数上限、右 pad 且保留已有 labels；拒绝 left padding，跨 DataLoader worker 旁路不丢。 |
| A07 | PASS | `test_a03_*`、`test_a07_*`：零目标、超长、模板正文变换、跨界 token 合并明确拒绝，不静默裁剪。 |
| A08 | PASS | `test_a08_*`：随机初始化微型 Qwen2 CPU，独立交叉熵对照 NLL、梯度、更新后参数（含 3+2 微批尾窗），零目标不更新；独立 RTX 3090 执行安装版 Unsloth logits 与 chunked fused loss 两条路径的默认有效目标数归约，对照 NLL 和一步参数更新。 |
| A09 | PASS | `test_a09_*`：seed、尾批边界、JSON sampler state 中途恢复、多源统计、上限/预算停机；见下方真实 tokenizer 投影。 |
| A10 | PASS | 原 `test_contract.py` / `test_hf.py` 原样通过；`test_a10_*` 覆盖带 Unsloth 环境变量的数据预检不调度、不读模型、asset-root；实跑旧配置 `--dry-run`/`--simulate-only`。 |

实测：CPU **65 passed in 72.36s**，CUDA Unsloth loss **2 passed in 23.99s**；CPT / SFT / legacy 三种本地 tokenizer 模拟通过；虚构 `release` 预检通过、哈希负例退出码 2。重新生成的默认 fixture 与仓库文件逐字节一致。

HF 环境：PyTorch `2.14.0+cu130`（CPU 执行）、Transformers `5.17.0`、Accelerate `1.15.0`、PEFT `0.21.1`。Unsloth 环境：Unsloth `2026.9.12`、PyTorch `2.12.1+cu130`。

### A05 逐 token 表节选

`token_str` 为 tokenizer 原始词表 token（byte-level 中文显示为字节映射）；完整表由上方 `audit_tokens` 命令导出。

| index | token_id | token_str | label（EOS 开） | label（EOS 关） | role |
|---:|---:|---|---:|---:|---|
| 0 | 248045 | `<\|im_start\|>` | -100 | -100 | template |
| 7 | 104563 | `çĽ¸æľº` | -100 | -100 | assistant（未选历史） |
| 41 | 248046 | `<\|im_end\|>` | -100 | -100 | user_stop |
| 50 | 96218 | `åħĪ` | 96218 | 96218 | assistant_target |
| 70 | 15 | `0` | -100 | -100 | user |
| 100 | 198 | `Ċ` | 198 | 198 | assistant_target（正文换行） |
| 161 | 248046 | `<\|im_end\|>` | 248046 | -100 | assistant_stop（目标轮） |
| 162 | 198 | `Ċ` | -100 | -100 | template |

### A09 多源样例（真实 tokenizer，模拟消费投影）

| 来源 | 样本/事件 | 字符 | 唯一正文+EOS | 输入 tokens（唯一 / 已计划） | 曝光 | 计划份额 → 批计划份额 | 有效 epochs |
|---|---:|---:|---:|---:|---:|---|---:|
| chat | 2 / 2 | 81 | 48 + 2 | 52 / 83 | 3 | 60.00% → 56.34% (`80`) | 1.60 |
| notes | 1 / 1 | 48 | 30 + 1 | 32 / 64 | 2 | 40.00% → 43.66% (`62`) | 2.00 |

预算 160，选用 142（正文 137 + EOS 5），剩 18；两次更新窗（后一窗不足），无截断。`stop_reason=no_eligible_complete_sample`：达到两次重复上限的候选 2 条，预算不足候选 1 条。真实训练性能文件另报告已成功更新的实际消费，不把投影当作训练结果。

**NOT RUN**：真实基座权重、CUDA adapter/4-bit 端到端训练、其他真实模型模板、真实发布语料；未安装新依赖、未下载模型。CPT 保真切块、SFT `trim_turns`、多人序列化、完整断点、阶段继承和推理运行时均不在此实现范围。v1 校验和编码驻内存，不适用于超大流式语料；清单是可信发布方提供的授权声明，不是数字签名。模板不能原样渲染或 token 边界无法证明时会拒绝（例如原生模板修剪了正文边界空白），不会修补正文。
