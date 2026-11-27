# Task D 虚构数据适配

直接读取 `examples/1/rag参考文件/Intelligent_Support_RAG_Fixtures.zip`，不解包执行里面的脚本，也不修改原始附件。`fixture.py` 只消费经 SHA-256 核查的表；校验 `.jsonl` 数量、源哈希和原文字符范围，再产出 D v1 短事件、精确 scope 和撤销 ID。拒绝候选事件/别名/关联，交集事件、证据片段和原文的 ACL。语料、授权、24 条 query 和 required/alternative/forbidden 均来自原包；`TERMS` 是对虚构语料的人工检索词校准，不是独立盲评。

```sh
D=$(mktemp -d /tmp/lm-rag-m5.XXXXXXXX)
CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 \
  .venv/bin/python -B -m lm_trainer.eval.memory_rag --data-dir "$D" --k 3 --card-tokens 400
```

如需观察预算影响，换新的私有目录、使用 `--card-tokens 1000`。CLI 只输出虚构 query ID、证据 ID、固定诊断、token、耗时和指标；不输出问题或卡正文，不生成回答。模型测试只可由获准操作者配置正式产物后复跑。数据库不落仓库。
