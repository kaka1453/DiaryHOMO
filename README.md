# DiaryHOMO

基于qwen小模型loar微调训练的AI包包日记

@boa你最爱的，训不训罢

## 产品资产目录

所有 BOA 命令都显式接收 `--asset-root`。私有根为 `product_assets/private/`，训练、推理和 WebUI 的 YAML 在 `config/`。原始数据在 `data/raw/`，训练和验证分别在 `data/train/` 和 `data/eval/`。人格、提示词、检查规则分别在 `assets/persona/`、`assets/prompts/` 和 `assets/guard/`。适配器在 `checkpoints/`，生成、评测、会话、调试和缓存写入 `runs/`。配置中的产品路径相对于资产根。外部公共基座用 `--external-base-model` 指定本地绝对目录。

```bash
python -m lm_trainer init-assets --asset-root /absolute/path/product_assets/private
python /absolute/path/diary_homo/train/train.py --asset-root /absolute/path/product_assets/private --external-base-model /absolute/path/base --dry-run
python /absolute/path/diary_homo/generate/predict.py --asset-root /absolute/path/product_assets/private --external-base-model /absolute/path/base
python /absolute/path/diary_homo/webui/webui_8bithf.py --asset-root /absolute/path/product_assets/private --external-base-model /absolute/path/base --init-only
python -B -m diary_core.eval.run --asset-root /absolute/path/product_assets/private --external-base-model /absolute/path/base --dry-run
```

从仓库外执行 `python -B -m diary_core.eval.run` 时先安装项目或把仓库加入 `PYTHONPATH`。预处理脚本位于 `data/dataset/md_preprocess/`，同样接受 `--asset-root` 和根内相对 `--input-file` 与 `--output-file`。分词脚本另需 `--external-base-model`。验证集由 `--val-file` 选择，默认 `data/eval/diary_validation_prompts.jsonl`。

`product_assets/example/` 提供独立编写的虚构样板。训练配置可用 `python -m lm_trainer --asset-root /absolute/path/product_assets/example --dry-run` 验证。推理示例使用本地基座和兼容的合成 adapter，评测可传 `--val-file data/eval/fictional_observatory.jsonl`。自行添加的基座、权重和运行结果在示例目录中也受 Git 忽略。`product_assets/user/` 提供调用方的目录位置。

`lm_trainer/` 的通用训练契约见 [共享训练器说明](lm_trainer/README.md)。公司侧通过 `tools/core_snapshot.py export` 获取白名单中的共享文件和可选的独立合成训练示例。
