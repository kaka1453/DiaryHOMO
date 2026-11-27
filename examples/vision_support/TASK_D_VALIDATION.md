# Task D · 评测 Runner、匿名盲评与本地 Gradio 调试台

## 基线、范围与回滚

- 起始 HEAD：`2ae5bc2223f5a142375aacb32d96ecf0b4d97311`（Task C 已就绪）。动手前离线 CPU 基线：**88 passed, 2 skipped**。
- 本任务只新增文件，未修改训练核心（`lm_trainer/train/*`）、推理内核（`lm_trainer/chat/*`）、依赖或锁文件。`git diff --stat` 仍只显示 Task A/B/C 的既有变更（`12 files changed, 1354 insertions(+), 251 deletions(-)`）。
- 新增代码 **`16 files, 4668 lines`**（见下表）。无提交、无推送。

| 新增文件 | 用途 |
|---|---|
| `lm_trainer/eval/__init__.py` | Task D 公开导出；导入不拉起 torch/transformers/peft/gradio |
| `lm_trainer/eval/schema.py` | `lm_trainer.eval.v1` 校验、manifest SHA-256、`test` 集锁、`fixture_only` 强制 |
| `lm_trainer/eval/cards.py` | M0–M4 卡片条件装配；仅准备候选快照，注入判定全在 Task C |
| `lm_trainer/eval/provider.py` | **唯一依赖注入点**：`MockRuntimeProvider` ↔ `RegistryRuntimeProvider` |
| `lm_trainer/eval/runner.py` | Headless runner：单卡串行、断点续跑、fixed-history / rollout 分开统计 |
| `lm_trainer/eval/blind.py` | 匿名盲评打包、稳定左右映射、评审落盘、揭盲统计、泄露检测 |
| `lm_trainer/eval/report.py` | JSON/JSONL/CSV/Markdown/HTML 报告；缺失项 `N/A` 或 `NOT RUN` |
| `lm_trainer/eval/__main__.py` | `python -m lm_trainer.eval {validate,run,cards,blind,reveal,report}` |
| `lm_trainer/webui/app.py` | 双 Tab Gradio 工作台（对话调试 + 盲评/报告） |
| `lm_trainer/webui/state.py` | 每浏览器会话状态；只持有 ID 与界面选项 |
| `lm_trainer/webui/markdown_safe.py` | Markdown/HTML/URL scheme 安全渲染 |
| `lm_trainer/webui/__main__.py` | `python -m lm_trainer.webui` |
| `examples/vision_support/task_d_eval/cases_dev.json` | 10 道虚构评测题（6 fixed_history + 4 rollout），dev split |
| `examples/vision_support/task_d_eval/cases_test.json` | 2 道 test split 门禁样例 |
| `examples/vision_support/task_d_eval/cards_m0_m4.json` | M0–M4 工单卡 fixture（含跨 namespace/拒读/过期卡） |
| `examples/vision_support/task_d_eval/mock_responders.json` | 离线 Mock 虚构回复（明确标注非模型输出） |
| `lm_trainer/tests/test_task_d.py` | D01–D08 离线 + 可选真实集成测试 |
| `lm_trainer/tests/test_task_d_ui.py` | 驱动真实 Gradio app 的端到端测试 |

**安全回滚**：先 `git status --short` 确认上表仍是本次新增项，备份个人修改后，仅将上表文件移到回收站（如 `gio trash -- <选定文件>`）。**不要** `git reset --hard`、`git clean` 或触碰 Task A/B/C 文件。`.cache/task-d-*` 为生成物，可单独识别后移到回收站；不清空回收站。

## D01–D08 验收

| 编号 | 结果 | 验收证据 |
|---|---|---|
| **D01** Gradio Mock 端到端 | **PASS** | `test_task_d_ui.py` 12 项：双 Tab、模型列表含 `unavailable` 禁用项、真实流式帧、取消、重试分支、新会话、参数校验、卡片审计（注入/拒载原因）、盲评提交与揭盲、Task B 基准读取 |
| **D02** C API 真实集成 | **PASS** | `test_d02_real_task_c_runtime_end_to_end`：真实 tiny Qwen2 + fixture adapter 完成对话，`input_tokens>0`、`output_tokens>0`，历史与锁全由 C 管理，第二会话历史为空 |
| **D03** 会话与隐私隔离 | **PASS** | 双浏览器会话交错聊天无串话（`B-leak: False`）；`launch_args()` 拒绝 `0.0.0.0` 与 `share=True`，环境变量 `GRADIO_SHARE=true` / `GRADIO_SERVER_NAME=0.0.0.0` 同样被拒；未勾选保存时无正文落盘 |
| **D04** Eval Schema 与门禁 | **PASS** | 非法 schema、篡改 case（`HASH_MISMATCH`）、`fixture_only=false`、未锁定 `test`（`TEST_SPLIT_LOCKED`）全部拒绝；`reference_only`/`rubric` 结构上无法进入 prompt（`test_d04_rubric_and_reference_never_enter_prompt`） |
| **D05** Fixed History & Rollout | **PASS** | 同题所有模型 `prompt_sha256` 完全一致；每题独立会话；缺权重模型 `SKIP` 不中断；续跑不重复、配置变更拒绝（`RESUME_MISMATCH`）；rollout 每 case 一个会话并逐轮记录 |
| **D06** 匿名盲评与揭盲 | **PASS** | 左右映射由 `sha256(seed‖case‖event)` 派生，同 seed 完全可复现；盲评包无模型名（`assert_no_leak` + 泄露即拒载）；平局/均失败/跳过计数准确；揭盲统计含独立 case/event、各维度分布与缺失值 |
| **D07** M0–M4 卡片对照 | **PASS** | 五种条件完整跑通；C 返回的实际注入/拒载 ID/版本/原因如实记录：M1 `injected`、M2 `injected`、M3 `stale`+`injected`、M4 `namespace`/`read_denied`/`superseded`+`injected`；报告不输出卡片明文 |
| **D08** 报告回读与回归 | **PASS** | JSON/JSONL/CSV 行数与 summary 一致；Markdown/HTML 安全转义（注入 `<script>` 被转义）；Mock 缺失指标显示 `N/A`、Task B 缺失项显示 `NOT RUN`；Task A/B/C 与旧测试无回归 |

**最终测试**：离线 CPU **132 passed, 5 skipped**（基线 88 + 新增 44）；真实集成 **3 passed**（`LM_TRAINER_TASK_D_REAL=1`）；`ruff check` 全绿。

## 可复跑命令

```sh
# 1. 离线 CPU 全套（含 Task A/B/C 回归）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  uv run --no-sync -m pytest -q lm_trainer/tests

# 2. 真实 Task C 集成（需先生成 tiny base，见下）
LM_TRAINER_TASK_D_REAL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  uv run --no-sync -m pytest -q lm_trainer/tests/test_task_d.py -k "d02 or d07_real"

# 3. 数据校验
uv run --no-sync -m lm_trainer.eval validate \
  --cases examples/vision_support/task_d_eval/cases_dev.json

# 4. Mock 评测（无 GPU、无权重）
uv run --no-sync -m lm_trainer.eval run \
  --cases examples/vision_support/task_d_eval/cases_dev.json \
  --mock --mock-responder-file examples/vision_support/task_d_eval/mock_responders.json \
  --model base --model sft_final --max-new-tokens 64 --seed 1234 \
  --output .cache/task-d/mock-run

# 5. M0–M4 卡片对照（Mock）
uv run --no-sync -m lm_trainer.eval cards \
  --cases examples/vision_support/task_d_eval/cases_dev.json \
  --cards examples/vision_support/task_d_eval/cards_m0_m4.json \
  --mock --mock-responder-file examples/vision_support/task_d_eval/mock_responders.json \
  --model sft_final --output .cache/task-d/mock-cards

# 6. 匿名盲评 + 评审 + 揭盲
uv run --no-sync -m lm_trainer.eval blind \
  --run .cache/task-d/mock-run/run.json --output .cache/task-d/blind --seed 2026
#   评审后：
uv run --no-sync -m lm_trainer.eval reveal \
  --pack .cache/task-d/blind/blind_pack.json \
  --map  .cache/task-d/blind/blind-map.json \
  --reviews .cache/task-d/blind/reviews \
  --output .cache/task-d/blind/reveal.json

# 7. 报告导出
uv run --no-sync -m lm_trainer.eval report \
  --run .cache/task-d/mock-run/run.json --output .cache/task-d/reports \
  --reveal .cache/task-d/blind/reveal.json

# 8. 本地 UI（默认 127.0.0.1，share=False）
uv run --no-sync python -m lm_trainer.webui \
  --mock --cards examples/vision_support/task_d_eval/cards_m0_m4.json \
  --benchmark examples/vision_support/benchmark_results.json --port 7861

# 9. 真实推理（RTX 3090，本地 tiny Qwen2 fixture）
#    先生成基座：
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 uv run --no-sync \
  -m examples.vision_support.task_b_workload --output .cache/task-d-prepared
#    再跑评测：
uv run --no-sync -m lm_trainer.eval run \
  --cases examples/vision_support/task_d_eval/cases_dev.json \
  --models examples/vision_support/task_c_models.json \
  --roles examples/vision_support/task_c_roles.json \
  --allow-fixture --base .cache/task-d-prepared/base \
  --model base --model sft_final --max-new-tokens 16 --seed 7 \
  --output .cache/task-d/real-run
```

## 架构与红线落实

**只消费 C API**：`RuntimeProvider` 是唯一后端接缝。`MockRuntimeProvider` 包装 Task C 的 `MockChatRuntime`；`RegistryRuntimeProvider` 调用 `load_registry`/`load_roles`/`ChatRuntime`。UI 与 runner 从不导入 `PeftModel`、从不渲染 chat template、从不检查卡片权限。

**单卡串行**：`run_evaluation` 逐模型 `build()` → 跑完该模型全部 (case, condition, turn) → `unload()` → 下一个模型。两个大模型不会同时驻留。

**rubric/reference 隔离**：`EvalCase.prompt_messages()` 只从 `messages` 构造输入。runner 与 blind pack 都不含 `rubric`/`reference_only` 字段；D04 测试直接断言这些文本不出现在 prompt 中。

**卡片门禁**：`ManualCardProvider` 由 Task C 提供，namespace / role scope / session scope / 版本 / 撤销 / read ACL 全在 C 内判定。D 只记录 C 返回的 `card_audit`（ID、版本、原因），绝不自行判断或补造。

**隐私**：`launch_args()` 在启动前拒绝非 loopback 绑定与 `share=True`，且忽略环境变量绕过。会话保存需**两步独立授权**（勾选 + 提供 `0700` 私有目录），实际写入由 C 的 `save_session` 完成（`0600` 文件）。盲评映射 `blind-map.json` 权限 `0600`。默认不写盘。

**盲评**：左右位置由 `sha256(seed‖case_id‖event_group)` 决定，同 seed 完全可复现；真实映射只存于私有映射文件。`load_blind_pack()` 会扫描 `model_id`/`stage`/`vram`/`adapter`/`base` 等暗示词，命中即 `BLIND_LEAK` 拒载。

**安全渲染**：`safe_markdown()` 转义 HTML 标签（保留代码块与行内代码原文）、中和 `javascript:`/`data:`/`vbscript:` scheme（含 HTML 属性内），保留换行与 emoji。

## 已知限制

- `fixed_history` 通过把固定 transcript 放入当前 user 轮实现（`history_mode="fixed_transcript"`），因为 Task C 只提交模型生成的 assistant 轮，无法注入字面历史。所有对比模型收到**逐字节相同**的 prompt（`prompt_sha256` 校验）。
- Mock 后端无真实 token 计数，Task C 会把选中但无法验证 token 的卡降级为 `not_sent`；真实后端返回 `injected`。两者都表示卡片已通过 ACL/作用域/版本门禁。
- 单个 `MockRuntimeProvider` 共享一个 runtime，适合单操作员交互或 headless 串行评测；真实后端每个模型独立 runtime，支持多浏览器会话。
- 真实 tiny 模型为随机初始化，输出无语义质量；部分 case 因乱码触发 Task C 的 `UNSAFE_OUTPUT`（这是**预期的安全行为**，报告如实记录）。
