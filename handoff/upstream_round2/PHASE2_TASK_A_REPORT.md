# 阶段二 · 任务 A 实施报告

验证日期：2026-11-27（Asia/Shanghai）。基线 `main` / `edf7ce8`，开始时工作区干净。未改动任务 A/B/C 旧契约及既有测试断言。用户粘贴文本为任务，参考 Markdown 的指令未扩大范围；只读解析并用独立合成源事实核对了其中 `win-joint-main` 的 JSON 投影，得到 `joint-m02`、`joint-m04` 两个目标。

## 交付与版本

- `lm_trainer/conversation/{__init__,contract,renderer}.py`：源事实 `lm_trainer.conversation-source.v1`、显式候选 `conversation.stage3.structured.v3`、时点观察及 `lm_trainer.conversation-renderer.v1`。
- `lm_trainer/train/{schema,encoding,data}.py`：`lm_trainer.release.v2` 预检、逐 token 归属与监督、保留每个生成机会的真实统计；旧 v1 与 legacy 校验保留。`train/{stages,engine}.py` 仅接入分区核对和阶段/产物统计元数据，不改训练循环、采样器或后端生命周期。
- `lm_trainer/chat/context.py`：新增共享直接推理前缀入口，旧双人状态 API 不变。
- `CONVERSATION_RENDERER_CONTRACT.md`、`tests/test_conversation_renderer.py`、`examples/conversation_renderer/{fixture.py,fixture_train.yaml,trace.py,synthetic_token_trace.json}`：接口契约、正反例、虚构 fixture 及可复验 `lm_trainer.conversation-token-trace.v1`。trace 是合成 HF BPE 的实际输入/监督数组，带源气泡 spans、stop、独立推理前缀及比较结果；不是模型或训练批准文件。

## 执行记录

| 类型 | 实际验证 |
|---|---|
| 原有 CPU 基线 | `CUDA_VISIBLE_DEVICES="" ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q lm_trainer/tests` → **131 passed, 6 skipped**（129.23s）。含已存在的微型 HF/LoRA、冻结基座、继承、断点与保存重载回归。 |
| 最终全套 CPU 回归 | 下方离线复跑命令（含 `CONVERSATION_REAL_TOKENIZER`）→ **185 passed, 6 skipped, 109 warnings in 133.65s**；新增 54 项均通过。 |
| 静态 | 修改范围内 `ruff check` → `All checks passed!`；`git diff --check` → 无输出。 |
| Mock tokenizer / CPU | 字符 mock 逐位复核 role/alias、气泡、可见历史、EOS 与前缀；故意构造非法模板和跨边界 merge。CPU CrossEntropy 验证 causal shift 之后仅监督 token 得到梯度，即使 PAD=EOS，padding 仍为 `-100`。 |
| 合成 HF BPE | 用仓库自带的合成 HF tokenizer 调用原生 `apply_chat_template`；`trace --check` → `conversation trace matches`。它是合成词表，不冒充基座模型 tokenizer。 |
| 本地真实 tokenizer | 仅加载已有 Qwen3.8 tokenizer（248,044 基础词表条目）；未加载 Qwen 权重。8 个生成机会完成原生训练／推理前缀逐位比较和各正文 span 解码复核，并验证真实 BPE 会合并跨气泡换行时拒绝、长段落仍可复原。 |
| CLI fixture | `--dry-run`、`--validate-release`、`--simulate-only` 均成功。结构预检显示 `lm_trainer.release.v2`、5 窗口/5 组/165 个字符；真实分词模拟为 `fixture_only=true`，首批形状 `[2, 88]`、计划 2 步，输出目录未创建。计划步数不是实际训练。 |

Qwen 实测原生模板 SHA-256：`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`；stop 绑定 SHA-256：`2e7a278144110c206e349e27f4e9c13b8cbb510c11e36f06139617612c59e896`。

## Token 与隔离统计

下表每格为「本地真实 Qwen / 合成 HF BPE」。`前缀`含原生 assistant generation prompt；`语境标记`是别名、间隔、媒体及可见引用/@ 的实际 token；它们、历史、气泡分隔都不进入 loss。分隔列仅计目标之间的换行。各组实际预算以 shifted loss token 计，不按窗数。

| 场景 | 机会 / 唯一目标 | 输入 | 前缀 | 正文监督 | EOS 监督 | 分隔 | 语境标记 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 1 / 2 | 35 / 103 | 16 / 39 | 16 / 62 | 1 / 1 | 1 / 1 | 0 / 0 |
| joint | 2 / 2 | 90 / 258 | 76 / 199 | 10 / 57 | 2 / 2 | 0 / 0 | 18 / 27 |
| group | 1 / 1 | 54 / 129 | 50 / 116 | 2 / 12 | 1 / 1 | 0 / 0 | 12 / 15 |
| media | 2 / 2 | 100 / 240 | 90 / 196 | 6 / 42 | 2 / 2 | 0 / 0 | 38 / 75 |
| anchor | 2 / 2 | 106 / 244 | 97 / 212 | 5 / 30 | 2 / 2 | 0 / 0 | 35 / 56 |
| 总计 | **8 / 9** | **385 / 974** | **329 / 762** | **39 / 203** | **8 / 8** | **1 / 1** | **103 / 173** |

真实 Qwen：direct 输入 35 / loss 17；group 输入 350 / loss 30。默认 fixture 编译无截断、无仅 EOS、无拒绝。非默认分布如下：2 个 primary 替代视图涉及 2 个已由 joint 覆盖的目标（不额外监督）；1 个 review 视图涉及 1 个与 group 测试视图重合的目标（review 记录本身未批准）；opening 与 continuation 各 hold 1 个独立目标。缺 `available_for_training` 的库存反例涉及 1 个被篡改目标，拒绝整个输入而不默许恢复。其他反例独立在各自临时目录运行，不与默认路由混算：身份/来源/span/边界、受保护或未选字段恢复、未来 owner 或见证、重复路由、跨分区/闭包、超预算、模板与控制符、哈希/审批/fixture 发布门禁分别有静态诊断。预检遇坏记录失败关闭，不跳过后静默训练。

## 离线复跑

从仓库根目录运行（本地已有 `.venv`，不安装或联网）：

```bash
export CUDA_VISIBLE_DEVICES="" ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CONVERSATION_REAL_TOKENIZER=/home/jutze/ZY_backup/llm/models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16
.venv/bin/python -B -m pytest -p no:cacheprovider -q -rs lm_trainer/tests
.venv/bin/python -B -m examples.conversation_renderer.trace \
  --check "$PWD/examples/conversation_renderer/synthetic_token_trace.json"
D=$(mktemp -d "$PWD/.cache/conversation-renderer.XXXXXX")
.venv/bin/python -B -m examples.conversation_renderer.fixture "$D/fixture"
cli=(.venv/bin/python -B -m lm_trainer \
  --config "$PWD/examples/conversation_renderer/fixture_train.yaml" \
  --model-config "$PWD/examples/vision_support/task_b_config/model.yaml" \
  --manifest-path "$D/fixture/manifest.json" --model-name-or-path "$CONVERSATION_REAL_TOKENIZER" \
  --checkpoint-dir "$D/never-train")
"${cli[@]}" --dry-run
"${cli[@]}" --validate-release
"${cli[@]}" --simulate-only
.venv/bin/python -B -m examples.conversation_renderer.trace \
  --tokenizer "$CONVERSATION_REAL_TOKENIZER" --local-native --output "$D/qwen-token-trace.json"
```

## BLOCKED / NOT RUN 与回滚

- **BLOCKED：** 当前提供的附件只有任务文本和参考 Markdown，未找到文档提及的 `Intelligent_Support_Groupchat_Fixtures.zip` / `build_support_fixture.py`。已验证 Markdown 的 joint 投影和自建的独立合成场景，未声称复验原 ZIP 的全部记录或其清单哈希。若提供 ZIP，应在独立目录先核验原始字节，按契约生成只读 source facts 并重跑候选/路由检查。
- **NOT RUN：** 未正式训练、未制作或发布新的生产 adapter，未部署机器人，未下载权重；生产 v2 仅做拒绝/门禁验证。Unsloth parity 无实测，v2 仍硬性禁止 Unsloth。CUDA/真实 Task D opt-in 用例按现有条件跳过，不属于这次 Qwen tokenizer 验证。
- 严格模板和逐段 token 对齐可能隔离部分真实模型模板或文本。v2 不自动裁剪长历史，不猜测缺失引用、不绕过超预算；业务 ACL、可信分区和不可变批准仍由发布方承担。分区目录必须物理隔离正文，全局 ledger/index 只含审批与分组元数据。当前只接入无状态结构化直接推理前缀，既有有状态双人 API 不会自动转群聊。
- 回滚：停用结构化 v3 allowlist，继续用未改变的 v1 release/legacy 配置；无需数据迁移。若撤回代码，应用本次增量的反向补丁即可，不需要切分支、reset、clean 或删除工作区文件。
