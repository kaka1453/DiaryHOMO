# 阶段二 · 任务 B 实施报告

日期：2026-11-27（Asia/Shanghai）。工作分支 `main`，基线 HEAD `edf7ce8daff8dd8330346abdd0f182ecf5475878`（唯一近期提交：`feat: initialize clean lm_trainer snapshot for round 2`）。开工时有任务 A 的未提交改动；保留其 `conversation/`、`train/schema.py`、`train/encoding.py`、旧契约及测试，未 reset、clean、切分支、commit 或 push。

Markdown `Intelligent_Support_Groupchat_Data_Test.md`（SHA-256 `6247c2a4807ebd2f1b0f852fb7a3769699eb59caed9c5b259111d56978143093`）作为**合成数据**只读：读取 JSON 投影 `win-joint-main`，逆核对既有独立合成源事实和目标 `joint-m02`、`joint-m04`。没有 ZIP；下文的「全量」仅指本轮由 Markdown 投影 + 现有虚构同伴场景生成的各个 fixture release（每条都实际编码），不是未知原始交付包，更不是正式语料。默认 joint 与替代 primary 在不同 fixture release，不增曝同源目标。CPT 另建完全虚构的作者历史与值班笔记。

## 审计与变更

| 范围 | 已有能力 | 本轮补齐 |
|---|---|---|
| 数据/模板 | 任务 A v3、精确 span、路由及分区审批 | 下游逐位独立核验、真实 tokenizer 证据与结构/真实 token 统计分离；超长按源计数后拒绝阶段；Markdown 专用可复现 fixture 工具 |
| 优化/采样 | HF/Unsloth native mean、有效 token 归约、source sampler、游标 checkpoint | 真实 HF 手工 FP32 梯度对照；AMP skip 不提交；v3 显式来源权重、严格预算/占比约束、独立事件/重复/尾窗报告 |
| LoRA | HF new/continue/resume、阶段产物/registry 门禁 | 有效 LoRA 结构、FP32 dtype 校验；继承版本谱系保护；对真实 span 的完整阶段、分批、暂停/恢复验证 |
| 性能 | 同步窗口/输入有效 token 统计 | 每次尝试独立排除预热；loader wait、主进程 CPU 时间和运行设置；已编译 dev 本阶段复用，不写缓存 |
| 评测 | Task D fixture、provider、盲评、报告 | 场景分类、三臂可选两两匿名比较、每维理由、虚构 M0–M4 开发回放与 tiny 核验 |

核心文件：`lm_trainer/__main__.py`、`config/stages.py`、`train/{data,dev,token_audit,engine,sampling,stages,modeling,artifacts,checkpoints,performance}.py`；回归 `tests/test_trainer_phase2_b.py`；`eval/{__main__,schema,runner,blind,report}.py` 新增可选场景及理由字段。实现说明和后端矩阵见 [TRAINER_OPTIMIZATION_CONTRACT.md](TRAINER_OPTIMIZATION_CONTRACT.md)，BASE / DIRECT_SFT / MIXED_CPT_SFT 可复跑命令见 [示例 README](../examples/phase2_task_b/README.md)。原有 `--asset-root`、通用入口与工业/视觉应用资产未改。

## 真实执行与输出

- **基线**：`CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q lm_trainer/tests` → **184 passed, 7 skipped, 109 warnings / 130.13s**。
- **第 1 层 / 静态**：`git diff --check` 无输出；针对变更的 `.venv/bin/ruff check lm_trainer/__main__.py lm_trainer/train lm_trainer/config/stages.py lm_trainer/eval lm_trainer/tests/test_trainer_phase2_b.py examples/phase2_task_b` → **All checks passed**。README 中 direct、mixed 的 `--dry-run` / `--validate-release` 及 primary `--validate-release` 均退出 0；结构输出 `token_counts_measured=false`。不使用额外全局源或升级依赖；使用 uv 已管理的现有解释器。
- **第 2/3 层回归**：在基线命令上加 `OMP_NUM_THREADS=2 CONVERSATION_REAL_TOKENIZER=/home/jutze/ZY_backup/llm/models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` 与 `-rs` → **193 passed, 6 skipped, 109 warnings / 130.00s**。8 个新增回归整体通过。末次盲评计数微调后，重跑 `lm_trainer/tests/test_task_d.py lm_trainer/tests/test_trainer_phase2_b.py` → **40 passed, 3 skipped / 20.93s**。旧阶段继续比较 optimizer/scheduler/scaler/RNG、顺序和逐位权重、保存重载推理；新 structured 路径用 [3, 3, 2] microbatch 窗比对手工 NLL/梯度、参数更新和跳步；在 step 1 暂停/恢复的测试与连续训练有相同顺序、实际 token、scheduler、游标与 final 权重，父 adapter 文件库存不变。真实 CUDA/AMP 跳步和正式模型不在此次验证范围，AMP overflow 用 CPU GradScaler 机制验证。
- **真实 tokenizer CLI**：README 三个 `--simulate-only` 均退出 0，`backend_model_validated=false`、`token_counts_measured=true`、无训练输出目录。Transformers **5.17.0**、tokenizers **0.23.2**；fast tokenizer identity `7fe1e0a3f34c707ba2f34f4e36b1e62de36c57cd71750292253369e697365caa`、native 模板 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`。这是本机已有 Qwen3.8 tokenizer；未加载其模型权重。另有 Task A 中文/边界/stop/独立推理前缀逐位回归。
- **纯虚构 CPU tiny 训练**：README 中不带 `--simulate-only` 的两个命令只用刚生成的随机基座。DIRECT_SFT **3** 步，CPT **2** 步，继承 SFT **3** 步；两个 SFT 的逐步输入/标签与采样顺序 hash 一致。SFT 每臂输入 **1,526**、有效 **211**；CPT 输入 **234**、有效 **229**；CPT 额外开销单列如下。未用真实数据或正式权重。
- **独立 Unsloth**：GPU 隐藏时整组命令得到 1 failed / 2 skipped（依赖 smoke 断言 `torch.cuda.is_available()`）；恢复仅设备可见性、不设置 `LM_TRAINER_CUDA_SMOKE`，运行 `env -u CUDA_VISIBLE_DEVICES -u LM_TRAINER_CUDA_SMOKE HF_HUB_OFFLINE=1 envs/unsloth/.venv/bin/python -B -m pytest -p no:cacheprovider -q -rs envs/unsloth/tests` → **1 passed, 2 skipped / 9.84s**。版本逐项符合锁：unsloth 2026.9.12 / zoo 2026.9.8 / Torch 2.12.1+cu130 / Transformers 5.5.0 / Accelerate 1.15.0 / PEFT 0.21.1 / BnB 0.50.2 / datasets 4.3.0 / xformers 0.0.35 / torchvision 0.27.1+cu130。只检查依赖导入和设备可见性，没有运行 Unsloth 训练或 parity。模型加载与 PEFT 仍使用隔离入口和原 guard；本轮不放行新能力。

### 本地真实 token 预检统计（非训练消费）

| 路由 / 来源 | 分词前窗口或文档 | 生成机会/文档 | 独立 target ID / event | Input | Body | EOS | 未监督分隔 | 有效 token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 共同 single_person | 1 | 1 | 2 / 1 | 52 | 16 | 1 | 1 | 17 |
| 共同 group_continuous | 3 | 5 | 5 / 3 | 345 | 13 | 5 | 0 | 18 |
| 默认 group_joint | 1 | 2 | 2 / 1 | 124 | 10 | 2 | 0 | 12 |
| 替代 group_primary（替换上行） | 2 | 2 | 2 / 1 | 124 | 10 | 2 | 0 | 12 |
| **任一完整 SFT 路由** | **5 / 6** | **8** | **9 / 5** | **521** | **39** | **8** | **1** | **47** |
| CPT support_history | 3 | 3 | 4 / 1 | 26 | 20 | 3 | 0 | 23 |
| CPT shift_notes | 2 | 2 | 2 / 2 | 29 | 25 | 2 | 0 | 27 |
| **CPT 合计** | **5** | **5** | **6 / 3** | **55** | **45** | **5** | **0** | **50** |

两套 SFT 规划均为 3 个更新、1 个两 microbatch 的尾窗；每个已选样本/目标最大 repeats=1，超长隔离=0，重复曝光=0。CPT 为 2 个规划更新、1 个尾窗；50 个有效 token 的实际投影为 46% 作者历史 / 54% 值班笔记，不伪称 60/40 达标；配重是目标，fixture 每样本上限 1，严格配重不满足时应用配置拒绝。最大长度 8 的负例拒绝整个计划；预算差 1 token 的严格负例也拒绝，不切片。

### 纯虚构 CPU 性能与产物证据

| 运行 | 输入 / 有效训练 tokens | 成功窗口 | 全程 wall 秒 | 预热后窗口数 | input/s | loss/s |
|---|---:|---:|---:|---:|---:|---:|
| DIRECT_SFT | 1526 / 211 | 3 | 12.721 | 1 | 5472.9 | 815.4 |
| CPT | 234 / 229 | 2 | 10.920 | 0 | N/A | N/A |
| MIXED 的 SFT | 1526 / 211 | 3 | 0.748 | 1 | 4544.9 | 677.1 |

CPU FP32 / 无量化 / 标准 gradient checkpointing / microbatch 1 / accumulation 3 / KV cache 禁用 / 预热 2。不同进程的冷启动不同，且仅一个稳态窗口，**不是性能优劣结论**。显存 allocated/reserved 均 null。证据文件在工作区 `.cache/phase2-b-delivery-1795756793/`（ignored；可由 README 重新物化）：`qwen-{joint,primary,mixed}.json`、`runs/*/performance.json`、`tiny-*.log`。已校验 fixture final manifest SHA-256：

- DIRECT_SFT：`45bdcf290a185bbc0d584d8f7c4d29914ac32a6a6b9fdd06949725ce5e5aaa60`
- CPT parent：`4e159f666e9e54469941e9e36583b28c05105a4357f57caa46d9c7991293f29b`
- MIXED 的 SFT final：`b523e8d4d60a5ebf4ebe7954fedd27e387a3b2516d882ae6cb3071627db2b161`

CPT 父权重指纹 `62f75fc580dbb2ca9bb5393666f98e7cecb4c31f7d7c4328939537edb613766c` 与继承 SFT 的起始指纹相同；继承 SFT final 权重指纹变为 `e125267ee209d0a1e1b1ea30f66cbbfb9120f31d0c27f12226dc91ffc149c4af`。正式 artifact 验证拒绝 fixture，checkpoint 也不能作 final。合成训练／评测完成后 Markdown 原始文件 hash 未变。

### 独立评测实测

`eval validate` 通过； mock 三臂 × 8 场景 × M0–M4 共 **120** 条，DIRECT / MIXED 匿名盲包 **40** 条。提交一条**虚构流程评分**（含维度理由），`reveal`、JSONL/CSV/Markdown/HTML 报告命令通过。也用已验证 tiny 注册表在 CPU 跑了真实重载推理（8 场景 × 3 臂、M0、生成上限 8）：24 条中 22 条提交（16 length、6 EOS），2 条随机输出被原运行时 `UNSAFE_OUTPUT` 门禁拒绝，无模型缺失。这不是客服质量评分，也不放宽输出门禁。独立开发评测和提示、训练/卡片不混放；当前 fixture-only 的 runner 不作为正式盲测入口。

## 阻断项、复跑与回滚

- **NOT RUN：**真实本地基座模型训练、GPU 容量/吞吐、真实 CUDA FP16/NF4 checkpoint、Unsloth 对同一 workload 的 loss/gradient/行为 parity。原因是本轮没有权重加载／训练授权。RTX 3090 设备可见性与依赖导入不证明模型容量。禁止引用旧基准作为本轮结果。
- **BLOCKED：**正式真实工单、独立正式评测分区与审批、真实模型对照质量结论，以及缺失原始 ZIP 流水线的全量验证。只有用户后续明确授权并提供固定本地 base/revision、批准语料和设备预算才进入下一层。应先用上述 `--simulate-only`，再在独立已锁环境用相同样本/张量 hash、step、dtype、预热、loss/gradient 容差跑 GPU 验证；Unsloth 源码提供的 `FastLanguageModel.from_pretrained/get_peft_model/for_training` 并不等价于可验证的同阶段恢复，不开放 `continue_stage/resume`。
- **失败回退：**保留父产物、已核验 checkpoint 及其 manifest hash，用同源码/锁环境、同 stage 配置、`init_adapter: resume` / `resume_from: .../checkpoint_N`、新的 `output_dir` 复跑，并去掉 `parent` 和旧暂停设置。只在成功终点发布 final。旧训练状态须用写出它的旧源码/锁环境恢复；旧 inference final 仍沿用 v1 验证。撤回本轮实现时仅反向应用任务 B 的改动，不能覆盖本来存在的 Task A 修改；不删除用户输入、现有缓存或产物，清理交由用户。
