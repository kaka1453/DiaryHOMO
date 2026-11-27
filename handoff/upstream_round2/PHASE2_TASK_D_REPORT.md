# Phase 2 · Task D 实施报告 · 2026-11-27

## 结论与实施前差异

基线分支 `main`，HEAD `edf7ce8daff8dd8330346abdd0f182ecf5475878`（`feat: initialize clean lm_trainer snapshot for round 2`）。起始已经有任务 A/B/C 的未提交内容及用户参考数据；测试基线是该**实际工作区**，不是干净 HEAD。未切分支、reset、clean、push；保留其他任务的改动。原数据参考 `examples/1/rag参考文件/Intelligent_Support_RAG_Fixtures.zip`，仅使用虚构记录，不调用任何远程服务、不下载模型/依赖，不把 SQLite 写入仓库。

| 范围 | 起始能力 | 本次补齐 | 环境 / 剩余门槛 |
|---|---|---|---|
| memory / ACL | 手工候选卡按 namespace/scope 与传入版本过滤；无全局点查 | SQLite 授权 + 版本/冲突/来源墓碑；倒排/结构化检索、二次校验、撤销/删除/重建/审计 CLI | `.venv` Python 3.13.15 / SQLite 3.53.1，实测可创建 FTS5 trigram 表；实际索引使用普通倒排表，不依赖 FTS5 |
| 任务 A / C | 已有可见事件投影、服务端可信 binding、原生 tokenizer、ContextBuilder | 无状态 hook 注入 + 构造 prompt 前权威核验；Task A 前缀不截断、不增加会话历史 | Python 装配 opt-in；未部署机器人或 HTTP 服务；不更改路由及训练循环 |
| eval | M0–M4 及旧报表 | 独立 M5 命令输出结构化查询与最终注入指标，不更改旧格式 | 正式模型 M1/M5：`MODEL_ARTIFACT_PENDING` / `NOT RUN`；生产知识库/真实数据：未获授权、`NOT RUN` |

## 变更清单

- `lm_trainer/chat/memory.py`：保持旧卡/Provider，新增三种 kind；`chat/context.py` / `chat/inference.py` / `chat/backend.py`：仅 Task C 注入钩子、二次门禁及真实参考 token 审计。
- `lm_trainer/memory/{__init__,schema,store,provider,__main__}.py` / `import-v1.schema.json`：检索/记录 API、私有 SQLite、持久撤销及管理 CLI、导入结构规范。
- `lm_trainer/eval/memory_rag.py`：独立 M5 离线评测/JSON 指标；`examples/memory_rag/{README.md,fixture.py}`：适配原始数据、词表和引用说明。
- `lm_trainer/tests/test_memory_rag.py`：权限、重放、冲突、语义、真实预算、注入、CLI、Task A/C 模型输入回归，以及 opt-in 的正式模型对照。
- `lm_trainer/MEMORY_RAG_CONTRACT.md` 与本报告：数据、授权、运维和回滚契约。

## 离线测试与静态检查

```sh
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q lm_trainer/tests
.venv/bin/ruff check --no-cache lm_trainer/chat/{memory,context,backend,inference}.py \
  lm_trainer/memory lm_trainer/eval/memory_rag.py lm_trainer/tests/test_memory_rag.py examples/memory_rag
git diff --check
```

- 实施前 CPU 基线：`230 passed, 9 skipped`（136.61 s）。新增专项：`23 passed, 1 skipped`（13.71 s，跳过正式产物运行）。验收观察到真实 HF fixture 的 `model.generate` 输入：正常注入；召回后立即撤销时，下一次构造输入精确等于无卡前缀。模型输出被测试固定为合成短文，用于验证输入而不是生成效果。
- 实施后 CPU 全量：`253 passed, 10 skipped`（141.09 s；保留基线的 110 条环境告警）。
- 变更范围 Ruff 0.16.9：通过；`git diff --check`：通过。全 `lm_trainer` Ruff 另有 4 条未修改文件的问题：`config/common.py` 的 TRY004、`tests/test_contract.py` 的 I001 和两条 C408；未顺手改动。导入 schema 和转换后的 JSON 另用已安装 `jsonschema` 校验通过（运行时不需它）。

## M5 工程实测（虚构数据，不是模型效果）

7 个 scope、22 个原始事件版本（1 个待审核候选不导入；1 个有权威撤销墓碑；入库含 19 个当前 active 家族），24 个有标注查询，17 条必需证据。术语表按该虚构语料校准，非独立盲评；历史知识截止使用参考包提供的查询线索，不作为历史授权。`recall@3` 在注入前按所有明确 required 证据 micro 计算；alternative 计为相关，但不抬高 required 召回；无关率按最终注入材料计算；违规计数覆盖召回和注入并集，快照攻击另有专项断言。实际 token 是仓库既有合成 HF tokenizer 原生模板编码前后之差，不是字符估算。耗时仅检索（含 SQLite/排序），单次本机离线采样，不代表负载 SLA。

| 增量预算 / 最大卡数 | recall@3 | 越权命中 | 撤销或旧版本命中 | 实际注入卡数 | 无关插入率 | 总增量 token | 检索 p50 / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 400 / 3（既有开发默认） | 17/17（100%） | 0 | 0 | 1 | 0/1 | 399 | 1.796 / 2.655 ms |
| 1000 / 3（预算对照，未改默认） | 17/17（100%） | 0 | 0 | 17 | 1/17（5.88%） | 6,761 | 1.854 / 2.864 ms |

| 查询（1000 token 对照） | 参考预期 | 权限后待排序事件数 → 实际召回 → 最终注入 |
|---|---|---|
| `q01_current` 当前维修状态 | 新结论 + 别名；不能返回旧状态/跨群 | 3 → `incident_a_status@2, alias_a_device@1` → 二者，696 token |
| `q04_same_nick` 群乙同昵称 | 群乙扫码事件 | 1 → `incident_b_scan@1` → 该卡，437 token |
| `q06_private_denied` / `q07_private_allowed` | 群聊不得读；授权私会话可读 | 0 → 空 → 空 / 1 → `private_a_address@1` → 该卡，420 token |
| `q10_revoked` / `q12_nodisclose` / `q13_holdout_denied` | 撤销、不可披露或非该 purpose 不得返回 | 1 / 2 / 2 → 均空 → 均空；这些数字只含当下 scope 的词法近似候选，不包含被权限排除的卡 |
| `q21_injection` 恶意资料 | 资料内伪指令不能进入 prompt | 4 → `injection_note@1` → `reserved_content`，0 token（检索命中，不视为已注入） |
| `q23_early_cutoff` 早期历史 | 仅当时的两条观察 | 3 → `chat_a_light@1, incident_a_initial@1, alias_a_device@1` → 三张，935 token；别名卡为额外无关插入 |
| `q24_unknown_scope` / `q19_unrelated` / `q20_ambiguous` | 不可返回资料 | 都是 0 → 空 → 空 |

工程取舍：不能用 100% 召回掩盖 400 token 下实际证据覆盖很低；合成 tokenizer 的中文编码较贵。预算只能按获准正式 tokenizer 实测后选择，不应单靠调大预算覆盖问题。`q23` 显示放宽预算后别名卡可能占用容量。数据规模、不同文本风格和载荷下效果未验证；查询在 scope 过滤后扫描有界候选，不承诺大库表现。未验证真实模型是否自然引用、能区分假设与结论、证据不足时是否主动澄清。

复跑全部 24 条及逐条候选/召回/注入审计（每次使用新目录）：

```sh
D=$(mktemp -d /tmp/lm-rag-m5.XXXXXXXX)
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m lm_trainer.eval.memory_rag --data-dir "$D" --k 3 --card-tokens 1000
# 默认基线把 1000 改成 400，并换一个私有目录。
```

## 正式模型 M1 / M5：`MODEL_ARTIFACT_PENDING` / `NOT RUN`

获授权的本地正式 Task C config 与凭证就绪之后，运行共享一个冻结模型的合成 Task A 场景冒烟（相同设置、人工选卡 vs 自动召回）；可经显式开关查看两条合成回答，仍需由人工按事实准确、自然及不硬插等评分。没有执行正式模型生成或打分，也无真实业务集准确度结论：

```sh
INFERENCE_FORMAL_CONFIG=/approved/local/service.json INFERENCE_FORMAL_SCOPE=fixture.acceptance.group \
  RAG_PRINT_SYNTHETIC_GENERATION=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q -s \
  lm_trainer/tests/test_memory_rag.py -k formal_m1_m5_generation
```

若 M1 已失败，先修参考注入、词表/预算或正式模板适配，不转向复杂索引。关闭 provider 可安全回到无记忆；不要退回不经实时 authority 的手工卡做线上授权。保留最新 SQLite authority/墓碑，只重建派生倒排；缺失最新 authority 时离线停用，不用旧库替代。文件系统/备份保留和撤销时的在途请求处理边界见 [契约](MEMORY_RAG_CONTRACT.md)。
