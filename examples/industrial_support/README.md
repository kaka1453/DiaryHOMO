# 合成设备客服示例

在源码快照根目录运行 `python -m examples.industrial_support XR-17 --revision A`。省略版本会收到澄清问题。`--manuals` 可以显式指定同结构的 JSON 证据库。条目按型号和硬件版本精确匹配，并带手册版本和章节。缺少或重复证据会转人工。

训练接入使用 `python -m lm_trainer --config examples/industrial_support/train.yaml --model-config examples/industrial_support/model.yaml --dry-run`。正式调用需显式准备本产品自己的本地基座并选择新输出目录。`questions.jsonl` 仅含合成问答方式。设备事实通过手册查询提供。
