# Task C · 变更与验收

## 基线、范围与回滚

- 起始 HEAD：`2ae5bc2223f5a142375aacb32d96ecf0b4d97311`。Task A/B 已存在于工作树（含未跟踪实现与产物），全部保留。动手前离线 CPU 基线：**77 passed, 2 skipped**；两项是现有 Task B 的 CUDA 测试。
- Python `3.13.15`、uv `0.12.6`、Transformers `5.17.0`、Torch `2.14.0+cu130`、Accelerate `1.15.0`、PEFT `0.21.1`、bitsandbytes `0.50.2`、pytest `9.1.1`。GPU 为 RTX 3090 24,576 MiB，驱动 `580.178.04`。本地验证使用 Task B 指纹锁定的随机 tiny Qwen2 和 CPT/SFT safetensors fixture；从固定 seed 重建基座，不下载权重、不读取客户数据。没有用无关的模型缓存。
- 默认 `git diff --stat` 仍只显示既有变更（`12 files changed, 1354 insertions(+), 251 deletions(-)`）。本任务新增文件未暂存，因此按下面的清单，用只读 no-index diff 计算本次 stat：**`16 files changed, 3522 insertions(+)`**。没有改训练核心、依赖文件或锁文件，无提交、无推送。

| 新增文件 | 用途 |
|---|---|
| `lm_trainer/chat/__init__.py` | Task D 公开导出，轻量导入 |
| `lm_trainer/chat/types.py` | 会话、设置、结果、事件、错误和审计类型 |
| `lm_trainer/chat/registry.py` | 本地注册、配置、Task A/B 验证、不可用原因 |
| `lm_trainer/chat/backend.py` | 单设备 HF/PEFT 状态、流式解码、取消、指标与资源释放 |
| `lm_trainer/chat/context.py` | 原生模板、保留符防护、卡片和完整轮预算 |
| `lm_trainer/chat/memory.py` | 两种 MemoryProvider、作用域/版本/ACL 及参考区 |
| `lm_trainer/chat/runtime.py` | 结构化会话、事务式提交、互斥、无依赖 Mock、显式私有导出 |
| `lm_trainer/chat/__main__.py` | 离线、无 UI 调试入口（默认仅打印元数据，不打印原文） |
| `lm_trainer/tests/test_task_c.py` | C01–C08 自动化回归 |
| `lm_trainer/TASK_C_CONTRACT.md` | Task D 冻结契约 |
| `examples/vision_support/task_c_roles.json` | 虚构 CS/L2 角色配置 |
| `examples/vision_support/task_c_cards.json` | 虚构工单/现场排查卡片 |
| `examples/vision_support/task_c_models.json` | Base/CPT/SFT、原生模板与 Task B fixture manifest SHA pins |
| `examples/vision_support/task_c_benchmark.py` | 短/典型/长历史 3090 基准与显存诊断 |
| `examples/vision_support/task_c_benchmark_results.json` | 真实测量 JSON，没有 prompt 或输出正文 |
| `examples/vision_support/TASK_C_VALIDATION.md` | 本报告 |

安全回滚（如需要）：先用 `git status --short` 确认以上仍是本次新增项，并备份其后个人修改；只将上表文件移到回收站（如 `gio trash -- <选定文件>`）。不要 `git reset --hard`、`git clean` 或触碰 Task A/B 文件。以下生成物在新建的项目 `.cache` 子目录中，可保留，或单独识别该目录并移到回收站；不清空回收站。

## C01–C08 验收

| 编号 | 结果 | 验收证据 |
|---|---|---|
| C01 注册/Manifest | **PASS** | 本地预检三个版本，缺权重、篡改文件/摘要、指纹、tokenizer、版本、fixture 标志与量化/dtype 错配拒绝；导入不拉起模型库。 |
| C02 单活动模型 | **PASS** | CPU 与 3090：Base → CPT → SFT → Base，验证旧模型弱引用消失、下一模型加载前 allocated 回到工作区基线、Base 不走 PEFT、PEFT 仅一条未合并 adapter、同进程第二个 runtime 的 CUDA 激活被拒。 |
| C03 原生模板/多轮 | **PASS** | Token IDs 与 Task A 原生 `apply_chat_template(..., add_generation_prompt=True)` 相等；保留结构与格式；裁剪旧完整轮、底线溢出报错、拒绝伪角色和不安全推理前缀。 |
| C04 卡片门禁 | **PASS** | 严格 namespace/角色/会话/版本/撤销/read ACL；检查最终 prompt、默认无记忆以及预算审计。 |
| C05 真流/取消/停止 | **PASS** | 实际 HF forward 在第 2 步用屏障等待：第 1 个 delta 已交付，取消确实在第 2 步停止；排队请求可独立取消，切换等待生成锁。另用真实 tokenizer 检查跨 token 的隐藏协议、stop 与 Unicode/换行/代码格式。 |
| C06 事务与隐私 | **PASS** | 后端异常/OOM 释放并可重试，已发 delta 在 error 终态撤回；无半截历史、无异常明文；显式目录/文件权限验证。 |
| C07 RNG 和实测指标 | **PASS** | 真模型同步/流式输出相同且 seeded 请求不污染 Torch RNG；验证 TTFT/吞吐/实际 token、历史增长与 CPU 显存为 null，并另跑下方 3090 基准。 |
| C08 Task D 接口 | **PASS** | CLI 本地离线入口与 Mock 会话/切换/终态测试；Mock 无伪 delta/伪 token 指标；原 `test_contract.py` 随全套回归运行。 |

最终全套结果：离线 CPU **87 passed, 3 skipped**（CUDA 测试未启用）；RTX 3090 环境 **88 passed, 2 skipped**（跳过的 Task B opt-in AMP 用例随后显式开启，单跑 **2 passed**）；Task C 在 3090 环境 **11/11 passed**；Ruff 与两轮 CLI 冒烟通过。未做发布模型质量评估、生产规模吞吐、量化 adapter GPU 兼容的端到端基准（已测量化错配拒绝）或不在本任务内的 Unsloth 训练复测。

### 复跑（仓库根目录，使用现有锁定的 uv 环境）

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 TZ=Asia/Shanghai
mkdir -p "$PWD/.cache"
work=$(mktemp -d "$PWD/.cache/task-c.XXXXXX")
CUDA_VISIBLE_DEVICES='' uv run --no-sync --locked python -m pytest -q lm_trainer/tests --basetemp "$work/cpu"
uv run --no-sync --locked python -m pytest -q lm_trainer/tests --basetemp "$work/gpu"
LM_TRAINER_TASK_B_CUDA=1 uv run --no-sync --locked python -m pytest -q \
  lm_trainer/tests/test_task_b.py -k cuda_fp16 --basetemp "$work/task-b-amp"
uv run --no-sync --locked ruff check lm_trainer/chat lm_trainer/tests/test_task_c.py examples/vision_support/task_c_benchmark.py

uv run --no-sync --locked -m examples.vision_support.task_b_workload --output "$work/prepared"
uv run --no-sync --locked -m lm_trainer.chat --models examples/vision_support/task_c_models.json \
  --roles examples/vision_support/task_c_roles.json --base "$work/prepared/base" \
  --allow-fixture --model cpt_stage1 --role CS_AGENT --cards examples/vision_support/task_c_cards.json \
  --stream --question 'Fictional IRIS camera: which exposure setting should we check?'
uv run --no-sync --locked -m examples.vision_support.task_c_benchmark \
  --base "$work/prepared/base" --output "$work/benchmark.json"
```

## 3090 实测（2026-11-26，Asia/Shanghai）

**Fixture only，无业务质量含义。** 真权重/真 adapter/真 kernel，FP32 无量化，greedy，16 个新 token。短历史 0 轮、典型 2 轮加 1 张授权卡（另一角色卡拒绝）、长历史 14 轮留最近 9 完整轮。12 个请求均完整提交、`finish_reason=length`。单次快照并非统计置信区间；对随机 byte 级小模型，无可发布的 UTF‑8 增量时会等到可见文本或 decoder flush，所以一些 TTFT 接近完成时间；CPT 的短/典型请求分别有 4/8 个真实 delta，C05/C07 另外验证在未完成时有真实 delta。

| 激活顺序 | 加载+前置校验 ms | 首次预热 ms | 激活后 allocated MiB |
|---|---:|---:|---:|
| base（进程冷启动） | 4525.4 | 1645.2 | 9.51 |
| cpt_stage1 | 2015.8 | 8.1 | 17.65 |
| sft_final | 377.5 | 10.4 | 17.65 |
| base（回切） | 351.1 | 8.8 | 17.64 |

| 模型 | 历史 | 输入/输出 tokens | 裁剪轮 | delta | TTFT ms | E2E ms | token/s | 峰值 allocated MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | short | 163 / 16 | 0 | 1 | 99.5 | 100.1 | 166.0 | 18.68 |
| base | typical | 941 / 16 | 0 | 1 | 77.4 | 78.0 | 227.8 | 23.64 |
| base | long | 1905 / 16 | 5 | 1 | 85.2 | 85.6 | 237.0 | 29.82 |
| cpt_stage1 | short | 163 / 16 | 0 | 4 | 16.1 | 78.9 | 211.3 | 18.69 |
| cpt_stage1 | typical | 941 / 16 | 0 | 8 | 21.5 | 88.9 | 196.1 | 23.65 |
| cpt_stage1 | long | 1905 / 16 | 5 | 1 | 92.4 | 93.0 | 210.6 | 29.83 |
| sft_final | short | 163 / 16 | 0 | 1 | 105.9 | 106.5 | 155.3 | 18.69 |
| sft_final | typical | 941 / 16 | 0 | 1 | 98.6 | 99.2 | 175.3 | 23.65 |
| sft_final | long | 1905 / 16 | 5 | 1 | 93.3 | 93.9 | 209.5 | 29.83 |
| base（回切） | short | 163 / 16 | 0 | 1 | 84.6 | 85.2 | 195.7 | 18.68 |
| base（回切） | typical | 941 / 16 | 0 | 1 | 74.0 | 74.5 | 237.4 | 23.64 |
| base（回切） | long | 1905 / 16 | 5 | 1 | 85.9 | 86.4 | 229.7 | 29.82 |

**释放证据。** Torch 进程初始 allocated/reserved 为 `0/0 MiB`；`unload()` 后 `16.25/20 MiB`。独立基准进程只清 cuBLAS 工作区和缓存即可回到 `0/0 MiB`，表明留下的不是模型、adapter 或 KV 张量。生产 `ChatRuntime` 不碰进程全局 cuBLAS 工作区，避免干扰同进程其他程序。C02 在下一次模型分配前验证只剩已预热工作区，并检查旧模型弱引用；真实切换的驻留量也未按版本累积。数据不含其他进程显存及驱动上下文，[原始 JSON](task_c_benchmark_results.json) 同时保存 allocated/reserved 和每次请求的峰值。
