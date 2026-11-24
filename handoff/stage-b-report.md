# BOA 阶段 B 私有验收与运行说明

2026-11-24，目标为 `new3090` 的 `develop`。阶段 A 基线为 `62cae87ce60a90e91aa7333342e17769f08d6694`。本次实现分支为 `refactor/product-boundary-20261124`。训练与产品边界提交为 `4dc8797`，快照工具与私有台账提交为 `7baad60`。本报告所在的交付提交可用 `git log -1 --format=%H -- handoff/stage-b-report.md` 定位。

## 如何运行

在 3090 私有主仓根目录，原命令 `python train/train.py`、`python generate/predict.py`、`python webui/webui_8bithf.py` 继续使用各自的产品配置。`--dry-run` 可检查配置。WebUI 共用已加载的模型，每个浏览器会话分别持有历史、审计及 Markdown 输出。清空按钮只清当前会话。卸载模型会释放共用资源并使正在生成的旧请求失效。

在独立交付包的解压根目录，可直接运行以下命令。共享训练入口要求两份 YAML。相对的基座、数据和输出路径以训练 YAML 所在目录为准。私有旧入口的相对路径仍以项目根为准。

```sh
python -m lm_trainer --help
python -m lm_trainer --config examples/industrial_support/train.yaml --model-config examples/industrial_support/model.yaml --dry-run
python -m examples.industrial_support XR-17 --revision B
```

合成客服示例从本产品的 `manuals.json` 精确匹配型号和硬件版本。版本 A、B 分别引用 `XR17-A-1 §2`、`XR17-B-2 §4`。缺版本先澄清，缺依据或有歧义则提示核对或转人工。样例训练配置指向本产品自己的基座和 `questions.jsonl`。包中只有源码、合成样例及许可证。

## B0 基线与现有资产

| 位置 | 实际版本与范围 |
| --- | --- |
| Mac `DiaryHOMO` | `RAG记忆升级` 在 `62cae87`。仅用来核对结构。 |
| 3090 私有主仓 `/home/jutze/ZY_backup/LEDtest/semi_minile/man/docker_test/diary_homo` | 开工时 `develop` 在 `62cae87`。`924cdc7` 引入按有效目标数更新、尾窗与最终 adapter，`62cae87` 引入隔离的可选后端及计量。有关训练文件没有待合并的 A 改动。原工作树已删除 `scripts/eval_diary_validation.py`。 |
| 3090 公司工程 `/home/jutze/ZY_backup/locall_train` | `main` 在 `9c8014035b5b7c6f00f889776cf94cc390a1f0e0`。五个已跟踪训练文件有工作树改动，另外两个后端文件尚未跟踪。观察到训练配置与合成测试有本地变动。只为七个指定通用源文件记录了实际字节指纹。 |

私有资产差异按文件存在性及目录的直接条目数核对，含占位项。条目数不用于判断两端内容是否相同。

| 类别 | Mac | 3090 |
| --- | ---: | ---: |
| `config/diary_profile.yaml` | 缺失 | 存在 |
| `data/dataset/json/` | 1 | 4 |
| `data/val/` | 1 | 1 |
| `checkpoint/` | 1 | 5 |
| `generate/output/` | 1 | 4 |
| `debug/` | 缺失 | 27 |

主仓原有发布依赖记录在 `requirements.txt`。仓根未发现 Dockerfile、Compose 文件或 Python 项目打包配置。共享交付采用可原样运行的源码快照。若作为容器构建上下文，使用白名单导出目录。私有 checkout、历史、角色卡、缓存、运行输出和两端的环境锁均在交付包之外。保留两份适用的 MIT 通知。

## B1–B2 结构及文件级改动

| 边界 | 文件与行为 |
| --- | --- |
| 训练入口与配置 | `lm_trainer/__main__.py`、`config/{common,train_config}.py`（相对 `lm_trainer/`）负责显式配置、参数校验及后端选择。保留 HF、`fp16`、梯度检查点、旧 `load_in_8bit` 映射与既有参数覆盖顺序。共享入口把默认缓存放在显式输出目录。`diary_core/config/{common,train_config}.py` 保留私有项目根和旧默认 YAML。`train/train.py` 保留原调用形式及延迟导出。 |
| 单一共享实现 | `lm_trainer/train/{bootstrap,data,engine,io,metrics,modeling,performance,reproducibility,schedule}.py`，`lm_trainer/model/{loader,quantization}.py`。同名 `diary_core` 文件保留兼容接口和原有模块检查入口。通用模型加载器不导入私有推理配置。Unsloth 可显式指定 `LM_TRAINER_UNSLOTH_PYTHON`，旧入口继续识别 `BOA_UNSLOTH_PYTHON`。 |
| 数据与产物 | `load_training_dataset` 仍读取 `text`，collator 屏蔽 padding；保留零目标窗口拒绝更新、尾窗更新和原样的 adapter 产物。`step_*` 和 `final` 都是推理 adapter。A 当前没有可续训的优化器、调度器和 RNG 状态。`performance.json` 的源码指纹现在指向实际执行的 `lm_trainer/` 文件，旧入口继续支持原有锁文件指纹。运行快照、计量和图表留在显式训练输出目录。 |
| 私有推理与会话 | `lm_trainer/generation.py` 负责解码新增 token。`diary_core/infer/{generation,diary_runtime}.py` 复用它，保留消息组装、角色、最近六条历史的附件处理、停止条件、后处理、Guard 和重试。`diary_profile`、`prompt_*`、`audit`、`eval` 和 WebUI 均由产品持有。未配置外部检索服务。`diary_core/webui/{app,handlers,state}.py` 使用每会话的 Gradio State、历史版本和独立输出目录；清空或卸载后，迟到的生成结果不会提交。 |
| 客服示例、依赖和测试 | `examples/industrial_support/` 包含独立命令、两版虚构手册、示例训练配置和合成文本。`lm_trainer/requirements.txt` 记录参考 HF 依赖。`lm_trainer/tests/{test_contract,test_hf}.py` 验证独立快照。`tests/test_product_boundary.py` 验证私有入口和原交互行为。 |

选用中性包名 `lm_trainer`，因为旧命名空间同时含私有推理和产品默认路径，直接把它作为交付根会将两者混在一起。旧包路径继续作为兼容入口。客服使用独立的配置和证据库，训练器无需识别产品名。对话 `messages` SFT 与 assistant-only loss 留在后续阶段。

## B3 同步

`tools/core_snapshot.py` 只主动装入清单中的 25 个核心文件，可另外导出 7 个合成示例文件。预览按文件列出来源、双方 SHA-256 和权限、复制、相同、上游未变、冲突和待复核。对未知文件只盘点共享目录内的名称。常规文件及其父目录均拒绝符号链接。历史、日志、索引、权重、源端补丁和私有台账不会随快照导出。

`handoff/core-ledger.json` 记录了目标实际字节，以及七个旧布局公司文件的来源路径、实际 SHA-256、权限和观察快照 `3e9a17bcd4535ae8f3b2d06b8d8f72acf3c4b0ca9ac2d8654e9b468bcef23869`。当前公司布局预览为 **25 项待复核、0 项复制**。历史观察没有自动授予复制资格。首次采用同构目录时，逐项审查源内容，批准文件使用 JSON 对象记录相对路径及对应 SHA-256。两端字节已相同且经批准时，显式执行带 `--establish` 的预览和候选操作即可登记首版。曾适配且内容仍不同的项继续人工复核。

以下变量表示准备好的来件、私有隔离 worktree 和来自本次审查的批准 JSON。预览本身只读。不提供批准文件也可预览。

```sh
python tools/core_snapshot.py --source "$incoming" --target "$review_tree" --approval "$approved"
python tools/core_snapshot.py candidate --source "$incoming" --target "$review_tree" --approval "$approved" --expect "$fingerprint"
```

第二条命令使用预览返回的完整 `fingerprint`。它在新的源码目录跑 CPU 套件，复验源、旧目标、批准记录和实测字节，用独立 Git 索引把核心改动和台账放在同一个候选提交中。整组有一项阻断时不会生成候选。成功时输出 `review/core-…` 分支，原工作树仍是原版。审阅候选后，可在隔离树用 `git merge --ff-only "$candidate_branch"` 接受。相同来件再次导入返回 `no changes`。已接受的同步可以用 `git revert <候选提交>` 将文件和台账一并回退。

默认继续使用经过批准的工作树快照。双方将来均获准发布并长期维护同一中性包、有共同版本流程后，再评估独立核心仓与固定版本依赖。

## B4 验收与复跑

| 验证 | 实际结果 |
| --- | --- |
| A 原测试 | 基线和拆分后各通过 20 项。覆盖不同监督长度与大批梯度等价、真实 dataloader 尾窗、零目标、定期和最终保存、独立进程重载、后端默认项及 Unsloth 工厂接线。 |
| 私有产品 | 从 A 提交取得合成消息、两次渲染、模型请求参数、Guard 判定、选中草稿与后处理的联合 SHA-256，B 与 `5f9e4d83c39967bb3f7db32bbdf79e954f89fa1b7b685939f074d0520e45497e` 一致。实测交错两会话、清空、迟到回复的历史及审计、Gradio 会话状态独立、显式模型选择和旧 CLI。缺少启用的人格文件时，旧入口报错。 |
| 干净源码 | 对最终交付压缩包在两仓之外的 `/tmp` 解压，使用空 HOME、离线 HF、禁用 CUDA，未注入原仓 `PYTHONPATH`。加载的通用模块均来自解压目录。7 项通过。普通 CLI 在 CPU 训练随机微型 Qwen2，三个合成样本形成 2+1 微批，两次更新后重载 `final`，验证 adapter 关闭、开启、再次关闭的数值结果。 |
| 同步、隔离与回退 | 合成来源与接收工作树覆盖未变、同内容、单侧/双侧漂移、旧适配、未批准、增删改名、源与目标链接、FIFO 和模式变化。测试中分别在预览后和测试中改源、改目标或撤销批准。失败保留原提交与账本。实际完整 25 文件组也经过默认 CPU 验收门，第一次只改获批文件和账本，第二次零改动。合成回退后 Git 树与上一批准版相同。向未列入清单的位置放入合成隐私哨兵不会被导出，向已列文件植入哨兵会阻断导出。 |
| 交付物 | 私有套件 `tests/` 共 44 项通过，包含 A 原测试 20 项、产品 4 项、同步与客服 20 项。交付包另测 7 项。压缩包为 32 个常规文件，成员名、模式与内容已复验，归档无用户、组、时间或私有 Git 元数据。SHA-256 为 `f6825817c5cb04f9b38f8de8c06b7c58d6dfb7be07ca63b4ee5e591e6d2bb749`。 |

在 3090 现有 HF 环境中复跑。各依赖从已有解释器读取，测试使用新临时目录和 CPU。实测环境是 Python 3.13.15、torch 2.14.0+cu130、Transformers 5.17.0、PEFT 0.21.1、Accelerate 1.15.0，HF 环境中未安装 Unsloth。项目里记录的旧版精确依赖组合未另建环境验证。

```sh
p=/home/jutze/ZY_backup/locall_train/.venv/bin/python
qa_dir=$(mktemp -d)
export CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 GRADIO_ANALYTICS_ENABLED=false
export HOME="$qa_dir" TMPDIR="$qa_dir" HF_HOME="$qa_dir/hf" MPLCONFIGDIR="$qa_dir/mpl"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
"$p" -B -m pytest -p no:cacheprovider -q tests
"$p" -B tools/core_snapshot.py export "$qa_dir/source" --with-example
(cd "$qa_dir/source" && "$p" -B -m pytest -p no:cacheprovider -q lm_trainer/tests)
```

回退本次功能时，在基于当前 `develop` 的新私有 worktree 逆序执行 `git revert --no-commit 7baad60 4dc8797`，复跑原 A 套件后再提交回退。原工作树既有的评测脚本删除不在这两个提交的改动范围。交付说明可以单独保留。自动生成的审阅提交使用命令级署名，不修改 Git 用户配置。

本轮没有启动 GPU、真实日记训练、真实客服服务或公共发布。WebUI 验证覆盖 Gradio 事件接线、状态和合成后端，未用浏览器加载真实模型或做线上并发压测。真实 Unsloth 运行、GPU 数值与性能未测。是否让公司正式采用中性目录，交由用户审阅独立草稿后决定。
