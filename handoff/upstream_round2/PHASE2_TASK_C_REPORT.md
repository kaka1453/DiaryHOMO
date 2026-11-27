# 2026-11-27 · 阶段二任务 C 实施报告

结论：无状态 Python 契约、独立 HTTP/SSE 适配、单模型门禁与离线验证已实现；**不等同于正式模型验收**。正式模型 `MODEL_ARTIFACT_PENDING`。未启动任务 C 的监听或对外服务；旧 Gradio 回归仅在测试内短暂使用回环端口。没有下载权重、使用真实工单/客户材料或改动依赖/锁文件。

## 基线与差异

- 分支 `main`，HEAD `edf7ce8daff8dd8330346abdd0f182ecf5475878`（`feat: initialize clean lm_trainer snapshot for round 2`）；`git log -n 5 --oneline` 仅有该提交。开工前 `git status --short` 已有任务 A、B 的修改和未跟踪文件（包括 `chat/context.py` 中 Task A 的 `build_conversation`）；本轮未 reset/clean/checkout/切分支。旧 Task A/B/C 契约、训练门禁、旧测试断言未改动。
- `.venv`: fastapi 0.142.1、starlette 1.7.0、uvicorn 0.54.0、httpx 0.28.1、pydantic 2.13.5、gradio 6.29.0、aiohttp 3.14.3、ruff 0.16.9；`vllm`、`llama_cpp`、`openai` Python 包未装，vLLM、`llama-server`、Ollama 命令均未发现。

| 已有实现 | 本轮补齐 | 环境/授权阻塞 |
|---|---|---|
| 有状态 v1、Mock、Gradio、eval；HF 激活/模板/stop/可见流 | 同 runtime/model lock 的无状态完整历史和观察；固定模型 lease、FIFO/取消/超时，不读写旧会话 | 正式模型未选择：`MODEL_ARTIFACT_PENDING` |
| Task A `render_observation`/native token 证明；Task B verified final | 可信角色与按 scope 授权资料的实际 tokenizer 预算；组聊 manifest renderer/stop 强校验；三路实测 | 本地真实 tokenizer 已测，仅 tokenizer，不加载其关联权重 |
| `.venv` 已有 HTTP 依赖，原工程未有该 HTTP 入口 | 可选 FastAPI、Bearer/API key、受控 scope、体积/生成限制、SSE/审计/启动检查 | 三方推理引擎 `NOT INSTALLED / NOT RUN`；正式 GPU 并发/吞吐/显存 `NOT RUN` |

## 改动文件

| 文件 | 内容 |
|---|---|
| `lm_trainer/chat/inference.py` | v1 Python request/result/event、可信绑定与 provider、单模型执行与入场、取消/截止、结构化审计 |
| `lm_trainer/chat/context.py` | 保留 A 的入口，新增真实 tokenizer 的完整输入/可选卡预算、保留标记过滤 |
| `lm_trainer/chat/{backend,runtime,registry,types}.py` | 共用模型锁和 HF 生成/VisibleStreamer；运行中固定模型；按真实 resolved policy 校验 manifest renderer identity；固定错误码和 Mock 适配 |
| `lm_trainer/serve/{__init__,app,__main__}.py` | 独立 HTTP 依赖、鉴权/结构化请求/响应、SSE、断连和取消、单 worker 回环启动器与 `--check` |
| `lm_trainer/tests/test_inference_service.py` | 虚构 Task A 样例、CPU HF fixture 与 Mock、ASGI 断连模拟、实测 token 断言、正式产物复跑入口 |
| `lm_trainer/{INFERENCE_SERVICE_CONTRACT,PHASE2_TASK_C_REPORT}.md` | 契约、配置和兼容矩阵、执行和回滚证据 |

## 实际验证

```bash
# 改动前 CPU 基线
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q lm_trainer/tests

# 本地真实 tokenizer（只读 tokenizer）；后续命令在此变量有效的条件下执行。
export CONVERSATION_REAL_TOKENIZER=/home/jutze/ZY_backup/llm/models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q -rs lm_trainer/tests/test_inference_service.py
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q -rs lm_trainer/tests
# 加开旧 eval + 小型 HF 合成权重集成用例
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 LM_TRAINER_TASK_D_REAL=1 \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q -rs lm_trainer/tests
```

| 验证 | 结果 |
|---|---|
| 改动前 CPU 基线 | **192 passed, 7 skipped** / 132.19 s；没有删改失败断言 |
| 初次 C 专项（未设可选本地 tokenizer） | **37 passed, 1 skipped** / 13.92 s |
| C 专项（最终代码、含本地真实 tokenizer 和正式产物 opt-in 门） | **39 passed, 1 skipped** / 15.57 s；跳过正式权重 |
| 全套 CPU（含本地真实 tokenizer；正式 opt-in 测试尚未加入该次收集） | **231 passed, 6 skipped** / 136.45 s；跳过 CUDA ×3、旧 eval 的 opt-in 合成 HF 集成 ×3 |
| 全套 CPU（含旧 eval 小型 HF 集成；末次采样配置回归加入前） | **234 passed, 4 skipped** / 140.26 s；跳过正式模型 ×1、CUDA ×3 |
| 最终全套 CPU（含旧 eval 小型 HF 集成） | **235 passed, 4 skipped** / 139.19 s；跳过正式模型 ×1、CUDA ×3，无失败 |
| C 改动范围 `ruff check` / `ruff format --check`，`git diff --check` | **通过**；新增文件另用 `git diff --no-index --check` 检查无空白诊断 |

逐项断言：

1. **逐 token 前缀**：任务 A 同一合成 group 样例的训练编码独立生成；拦截 Python 请求、HTTP 非流式、HTTP SSE **实际传给 HF `model.generate` 的 `input_ids`**，对照训练监督正文起点之前的前缀，三者完全相同。测试使用已有虚构 adapter/合成 byte-level tokenizer，以及本地生成的极小随机 CPU 基座；正反例的回复只在测试中设定 logits，仍由实际 HF 生成与 VisibleStreamer 经过原生 stop 路径。另用本地真实 Qwen tokenizer 直接证明新 ContextBuilder 的训练/服务前缀一致；这不是其模型推理结果。
2. **隔离与权限**：不同授权、角色和完整历史依次/交错提交，不污染旧 session；错误模型/未授权 scope、结构化对话归属、角色覆盖、无 fixture 双开关、错误 manifest SHA 或 renderer/stop、未知工具/推理字段、伪造角色/控制 token、超量正文/上下文/生成被拒绝。资料按请求 scope、namespace 和 read ACL 验证，实际 tokenizer 预算只丢弃可选资料，不裁剪任务观察；未观察正文不提供给 provider。
3. **生命周期**：真 SSE 的 role/delta/finish/`[DONE]` 顺序和 TTFT；mock 仅缓冲成功终态而不伪造时间。工具伪造输出、非成功 SSE、HTTP 授权取消、不同 principal 取消失败、ASGI 2.4 双路断连、满队列、排队/生成超时以及模拟 OOM 均覆盖。运行中超时不抢占工作槽；下一次生成保持原始固定模型且不复用历史。启动 `--check` 用环境密钥，测试断言不打开端口。
4. **日志/懒加载**：捕获日志验证不含虚构问题、回复、卡正文、卡来源路径或凭据；子进程验证导入 `lm_trainer.chat` 并跑 Mock 无状态服务不会导入 torch、transformers、peft、unsloth 或 HTTP 框架。CPU/mock 没有冒报 GPU 指标。
5. **回归**：全套继续包含旧 Task C、WebUI 与 eval；未修改旧断言。当前依赖的 Starlette/HTTPX、SWIG/Gradio、fork 有已知兼容/弃用警告，本轮未升级它们。

```bash
.venv/bin/ruff check --no-cache lm_trainer/chat lm_trainer/serve lm_trainer/tests/test_inference_service.py
.venv/bin/ruff format --check --no-cache \
  lm_trainer/chat/{backend,context,inference,registry,runtime,types}.py \
  lm_trainer/serve lm_trainer/tests/test_inference_service.py
git diff --check
```

附加全目录 `ruff check --no-cache lm_trainer examples/conversation_renderer examples/phase2_task_b` 报 4 条既有问题：未改文件 `lm_trainer/config/common.py:103` (`TRY004`)、`lm_trainer/tests/test_contract.py:1/57/80` (`I001/C408/C408`)；未替其他任务修改。

## 正式阻塞与复跑

- `MODEL_ARTIFACT_PENDING`：未提供已正式选定且获授权的 registry、与之匹配的 verified `inference_adapter`、manifest SHA 和角色/场景配置。没有用测试 fixture 充当正式产物。待就绪用[契约中的同一 opt-in 复跑入口](INFERENCE_SERVICE_CONTRACT.md#启动停止与复跑)：先 `lm_trainer.serve --check`，再 `INFERENCE_FORMAL_CONFIG=/path/to/service.json ... -k formal_model_acceptance`，只消费同一虚构 group，走进程内 HTTP 并对比正式模型的三路前缀。
- `NOT RUN`：正式 GPU 延迟/吞吐/显存/并发，及 CUDA kernel 时间和可取消边界；需要获批的固定设备、模型版本、请求集和压测方案。先执行上面的正式 ASGI 复跑，再独立制定容量压测；不把 CPU/mock 指标或参数量推测当作容量证据。
- `NOT INSTALLED / NOT RUN`：vLLM、llama.cpp server、Ollama。后续按契约表执行独立后端引入验证；不做 LoRA merge/量化/GGUF 转换，不删除或替换原产物。

## 失败回滚

停止本地 launcher（Ctrl+C/SIGTERM），停止将客户端流量路由到新 HTTP 入口；旧 headless/Gradio/eval 不需要新服务配置即可继续运行。未写客户端会话、模型、adapter 或发布数据，无数据迁移。若需源码撤回，先保存现有 diff，再人工确认并反向应用仅本任务的补丁：`chat/context.py` **只撤回** `check_inference_text` / `INFERENCE_CONTROLS` / `_encode_rendered` 提取 / `build_inference_v1` 及其 import，保留任务 A 的 `build_conversation` 和原生渲染改动；其他文件也只反向移除本报告列出的 C 修改。新文件如需移走应移至回收站，不执行 `reset/clean/checkout` 或永久删除，不动其他任务和系统内容。
