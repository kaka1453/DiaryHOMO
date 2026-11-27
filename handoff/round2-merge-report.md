# DiaryHOMO Round 2 合并实施报告

> 日期：2026-11-27｜执行主机：`new3090`｜审阅分支：`merge/company-round2`｜状态：**代码合并与离线 CPU 合成验收 PASS（未动 `develop`，未运行真实训练或 GPU）**

## 1. 仓库与版本证据

| 角色 | 路径 / 分支 | 基线 / 目标 Commit |
|---|---|---|
| 上游源仓（只读） | `/home/jutze/ZY_backup/locall_train` (`main`) | `edf7ce8daff8dd8330346abdd0f182ecf5475878` + Round 2 未提交增量（77 文件全部逐字节匹配方案第 12 节 SHA-256） |
| 本地主仓（未改动） | `/home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo` (`develop`) | `2996ccd87392e660abe51b987b19036d7db4b2e6`（工作区干净） |
| 隔离审阅工作树 | `/home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo_wt_round2` (`merge/company-round2`) | 基于 `2996ccd` 新建分支，包含 4 个原子提交 |

### 审阅分支提交清单

1. `45b2998` `sync(lm_trainer): import Round 2 shared core (Tasks A-D)` — 精确导入 24 个变更 + 23 个新增运行/契约/测试文件（共 47 文件，`lm_trainer/` 合计 86 文件，逐字节匹配上游）。
2. `67bd509` `chore(sync): upgrade core_snapshot/ledger to 86 files and import Round 2 fixtures & env locks` — 升级 `tools/core_snapshot.py`（86 文件 `CORE`、`TEST_SUPPORT`、`BINARY_FIXTURES`、`ENV_LOCKS` 及 `--allow-new` 原子新增支持）、更新 `handoff/core-ledger.json`、归档 4 份上游报告至 `handoff/upstream_round2/`、导入合成测试资源（并对 `examples/phase2_task_b/README.md` 脱敏绝对路径）、导入根及 `envs/unsloth` 锁文件、并集合并 `.gitignore`、生成 `handoff/round2-merge-manifest.json`。
3. `a1d217b` `fix(compat): guard --validate-release and staged configs in legacy entrypoints` — 修复 `diary_core/config/train_config.py` 多阶段资产路径校验；修复 `train/train.py` 与 `diary_core/train/engine.py` 的 `--validate-release` 误触发训练风险及多阶段快速拒绝，并补充 `tests/test_round2_local_compat.py`。
4. 本提交：`docs(handoff): record Round 2 merge verification report`。

## 2. 合并清单与差异处理

- 完整 78 项（26 同路径变更 + 52 新增）前后 SHA-256 与处理动作见 `handoff/round2-merge-manifest.json`。
- 删除文件：0。本地独有模块（`diary_core/`、`train/`、`generate/`、`webui/`、`data/`、`product_assets/{private,example,user}/`、`tests/`、`tools/`、`handoff/`）全部完整保留。
- 私有资产完整性：`git ls-files product_assets/private` 返回空；`git check-ignore -v product_assets/private/fictional_test.jsonl` 命中 `.gitignore:34:/product_assets/private/`。
- 脱敏与归档：
  - 4 份上游报告归档至 `handoff/upstream_round2/PHASE2_TASK_{A,B,C,D}_REPORT.md`，未放进 `lm_trainer/`。
  - `examples/phase2_task_b/README.md` 中 `/home/jutze/...` 已替换为仓库相对路径与 `${LOCAL_QWEN_MODEL_DIR:-...}`。
  - `examples/vision_support/TASK_A_VALIDATION.md` 接收上游已脱敏版本。

## 3. 执行环境与离线 CPU 实测结果

- 解释器：`/home/jutze/ZY_backup/locall_train/.venv/bin/python`（Python 3.13.15）
- 关键依赖：`torch 2.14.0+cu130`、`transformers 5.17.0`、`peft 0.21.1`、`accelerate 1.15.0`、`datasets 5.0.1`、`bitsandbytes 0.50.2`、`pytest 9.1.1`、`fastapi 0.142.1`、`httpx 0.28.1`、`uvicorn 0.54.0`、`gradio 6.29.0`。
- 运行模式：纯离线、纯 CPU、纯合成 fixture（`CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`），未读取真实私有语料，未下载权重，未启动正式训练或对外服务。

| 检查项 | 命令摘要 | 结果 | 退出码 |
|---|---|---|---:|
| Diff 格式与空白检查 | `git diff --check 2996ccd..HEAD` | PASS | 0 |
| 私有目录忽略与跟踪 | `git check-ignore -v ...` & `git ls-files product_assets/private` | PASS（未跟踪，已忽略） | 0 |
| 合成 Token Trace 校验 | `python -B -m examples.conversation_renderer.trace --check examples/conversation_renderer/synthetic_token_trace.json` | PASS（`conversation trace matches`） | 0 |
| 本地产品/同步/兼容回归 | `python -B -m pytest -p no:cacheprovider -q -rs tests`（含修复后 `test_round2_local_compat.py` 3/3） | PASS（50 passed, 0 failed） | 0 |
| 共享核心全套回归 | `python -B -m pytest -p no:cacheprovider -q -rs lm_trainer/tests` | PASS（253 passed, 10 skipped, 0 failed） | 0 |
| Phase III 合成 Tiny E2E | `examples.phase2_task_b.prepare --with-tiny-base` -> `direct_sft` & `mixed_cpt_sft` 的 `--validate-release` / `--simulate-only` / CPU tiny 训练 / `model_registry --include-adapters` | PASS（全部阶段与继承通过） | 0 |
| 同步工具幂等预览 | `python -B -m tools.core_snapshot preview --source /home/jutze/ZY_backup/locall_train --target .` | 86/86 `identical`，`unexpected.target=[]`，`unexpected.source` 仅含 4 份已归档上游报告 | 2（预期） |

### 10 个 Skip 原因说明（均属预期 opt-in / 未授权项）

1. `test_conversation_renderer.py:803`（1）：未设置 `CONVERSATION_REAL_TOKENIZER`。
2. `test_inference_service.py:1082, 1120`（2）：未设置 `CONVERSATION_REAL_TOKENIZER` / `MODEL_ARTIFACT_PENDING`。
3. `test_memory_rag.py:570`（1）：`MODEL_ARTIFACT_PENDING`。
4. `test_task_b.py:461`（2）：CUDA/AMP opt-in 检查点测试。
5. `test_task_c.py:165`（1）：CUDA 显存生命周期检查。
6. `test_task_d.py:659, 694, 723`（3）：未设置 `LM_TRAINER_TASK_D_REAL=1`。

## 4. 遗留项与放行状态

- **M0 / M1 / M2（共享核心、同步账本、环境锁、旧入口兼容、CPU 合成回归）**：**PASS**。
- **M3（BOA Corpus Stage3 → `release.v2` 私有适配与全局分区）**：**NOT RUN / 待后续独立任务**。
- **M4（BOA 角色中立 renderer profile、`BOA_V3_PROFILE_PENDING` 与正式评测门禁）**：**NOT RUN / 正式群聊 SFT 前必做**。
- **M5（真实本地 tokenizer 前缀验证与 3090 GPU 工程冒烟/正式训练）**：**NOT RUN / 需单独授权**（`MODEL_ARTIFACT_PENDING`）。
- **M6（HTTP 服务 / RAG 运行时启用）**：代码已合入并默认关闭，未监听端口，未创建 BOA 记忆库。

## 5. 合入 `develop` 与回滚方法

- 若审阅满意，可在主仓 `/home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo` 上执行快进合并：
  ```sh
  git -C /home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo merge --ff-only merge/company-round2
  ```
- 若需放弃本次合并，直接移除隔离工作树与分支即可，原 `develop`（`2996ccd`）完全未受影响：
  ```sh
  git -C /home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo worktree remove /home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo_wt_round2
  git -C /home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo branch -D merge/company-round2
  ```
