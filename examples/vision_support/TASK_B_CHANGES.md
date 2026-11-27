# Task B｜基线、变更和回滚

## 开始状态

- 起始 `HEAD`：`2ae5bc2223f5a142375aacb32d96ecf0b4d97311`，`main`。工作树已有 **Task A 未暂存修改和未跟踪文件**，因此仅回到 `HEAD` 会丢掉 Task A。未切换分支、未提交、未推送，未修改 `pyproject.toml`、`uv.lock` 或依赖列表。
- 修改前保存了 `git status`、`git diff --binary` 和 Task A 原有未跟踪文件归档。入口在 `.cache/task-b-baseline-path`；完整本地执行日志在 `.cache/task-b-evidence-path` 指定的目录。归档和日志不纳入交付报告；交付 JSON 不含原文或敏感绝对路径。
- 基线命令：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 uv run --no-sync -m pytest -q lm_trainer/tests`，**65 passed**。HF 和 `envs/unsloth` 的既有 venv 可用；基础版本和实测设备见[基准报告](BENCHMARK_REPORT.md)。
- 本地模型盘点范围：用户 Hugging Face 缓存和已有模型目录。微型缓存 `models--sshleifer--tiny-gpt2` 未用于新阶段；测试从零生成 Qwen2。已存在的 27B AWQ/GGUF 与其他不在档位的模型明确排除。未发现符合 L3/L4 模型档位的基础权重。

## 文件改动说明

| 文件 | Task B 改动 |
|---|---|
| `lm_trainer/__main__.py` | 同一 CLI 预查/模拟和执行完整阶段计划；保持延迟导入与进程隔离。 |
| `lm_trainer/config/train_config.py` | 严格解析单/多阶段兼容入口，约束超参，补生命周期 CLI 覆盖。 |
| `lm_trainer/config/stages.py` | 阶段独立默认值、显式父引用/恢复和输出不重叠校验。 |
| `lm_trainer/train/engine.py` | 在 Task A 的 update window 上接入三种生命周期、真实 batch 审计、原子发布、阶段/dev 监控。 |
| `lm_trainer/train/modeling.py` | 检查 LoRA 目标和冻结范围；按 dtype/量化加载基座，正确 HF/PEFT 父权重加载和字节校验；隔离 Unsloth。 |
| `lm_trainer/train/artifacts.py` | Base 实际权重和 tokenizer 指纹、签入清单/完整相对文件库存、fixture 默认拒绝、PEFT 配置兼容校验。 |
| `lm_trainer/train/checkpoints.py` | 安全序列化 optimizer/scheduler/scaler/RNG/曝光和已消费游标；恢复前检查身份与计划。 |
| `lm_trainer/train/stages.py` | 绑定阶段身份及 lineage、冻结基座和训练有效性审计、独立 dev 输入检查。 |
| `lm_trainer/train/dev.py` | 按真实目标 token 加权的 per-source dev、定点虚构对话 probe。 |
| `lm_trainer/train/io.py` | Linux `renameat2(RENAME_NOREPLACE)` 原子发布；不替换已有目标。 |
| `lm_trainer/train/data.py` | 记录截断、每 loader 隔离 generator，预取或重建迭代器不扰动训练 RNG。 |
| `lm_trainer/train/sampling.py` | 追踪实际 unique/累计曝光和持久化状态，保留 Task A 的计划与完整样本预算。 |
| `lm_trainer/train/schedule.py` | 用真实批次数（含尾批）限制阶段 warmup。 |
| `lm_trainer/train/schema.py` | 将显式 CPT 下游模板绑定、dev 独立 split 校验纳入 Task A 门禁，先验证声明再打开载荷。 |
| `lm_trainer/train/reproducibility.py` | Python/NumPy/CPU/CUDA seed、HF 确定性运行。 |
| `lm_trainer/train/performance.py` | 共享文件/批次 SHA、环境和按实际分子/分母的可聚合吞吐。 |
| `lm_trainer/tests/test_task_b.py` | B01–B08；虚构 tiny Qwen2、精确恢复、原子故障、负例、CUDA/AMP opt-in。 |
| `lm_trainer/README.md`、`lm_trainer/TASK_B_CONTRACT.md` | 入口与 Task C/D 消费约定。 |
| `examples/vision_support/task_b_workload.py` | 可重建纯虚构负载、真实 tokenizer 校准、无下载微型基座。 |
| `examples/vision_support/task_b_config/{stages,model}.yaml` | 可直接复制到生成目录的两阶段场景。 |
| `examples/vision_support/task_b_benchmark.py` | 多级 CUDA 矩阵、独立解释器、等价性和聚合证据、按本地权重尺寸拒绝 27B。 |
| `examples/vision_support/task_b_fixture_adapters/` | 两份小型 canary adapter（safetensors、config、tokenizer、manifest），以及 pin 齐的索引/重建说明；无 checkpoint 和基座大文件。 |
| `examples/vision_support/{README,BENCHMARK_REPORT,TASK_B_CHANGES}.md`、`benchmark_results.json` | 入口、报告、验收、无正文的机器证据。 |

`lm_trainer/model/loader.py` 在开工前已因 Task A 修改，Task B 没有再改它。其他 Task A 原始未跟踪材料保持原样。

## 安全回滚（仅在用户决定放弃 Task B 时）

先保存当前改动，尤其是 Task B 完成之后可能新增的用户编辑；不要用 `reset --hard`、`git clean` 或对整仓操作。在**新目录**重建原始 Task A 工作树以供对照：

```sh
baseline=$(realpath "$(cat .cache/task-b-baseline-path)")
task_a_baseline=$(mktemp -d "$PWD/.cache/task-a-for-comparison.XXXXXX")
git archive "$(cat "$baseline/head.txt")" | tar -x -C "$task_a_baseline"
(cd "$task_a_baseline" && patch --batch -p1 < "$baseline/task-a.patch")
tar -xf "$baseline/task-a-untracked.tar" -C "$task_a_baseline"
```

逐文件 `diff -u`/审阅后，只从该快照恢复上表中**开工时已存在**的条目（包括 Task A 当时未跟踪的 `schema.py`/`sampling.py`）。保留用户后来编辑。仅 Task B 新增的代码、配置、报告和 fixture 目录，审阅后用 `gio trash -- PATH` 逐项移入回收站；不要清空回收站。已有训练目录不覆盖也不清理。

## 最终 Git 统计

下方是相对起始 Git `HEAD` 的 tracked `git diff --stat`，含 Task A 和 B；从未暂存加入新文件，因此 Task B 新增文件另见上表和 `git status --short`。

<!-- TASK_B_DIFFSTAT -->

```text
 examples/vision_support/README.md   |   8 +
 lm_trainer/README.md                |   6 +-
 lm_trainer/__main__.py              | 113 +++++-
 lm_trainer/config/train_config.py   | 166 ++++++--
 lm_trainer/model/loader.py          |  11 +-
 lm_trainer/train/data.py            | 246 ++++++++----
 lm_trainer/train/engine.py          | 746 ++++++++++++++++++++++++++++++------
 lm_trainer/train/io.py              |  48 ++-
 lm_trainer/train/modeling.py        | 211 +++++++++-
 lm_trainer/train/performance.py     |  31 +-
 lm_trainer/train/reproducibility.py |   9 +-
 lm_trainer/train/schedule.py        |  10 +-
 12 files changed, 1354 insertions(+), 251 deletions(-)
```
