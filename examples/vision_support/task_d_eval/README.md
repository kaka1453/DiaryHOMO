# Task D · 虚构机器视觉客服评测集与工单卡

全部内容为从零编写的虚构 fixture，仅用于工程链路验证。`fixture_only: true`，不对应任何真实客户、真实工单或真实设备。

## 文件

| 文件 | 说明 |
|---|---|
| `cases_dev.json` | 10 道虚构评测题（6 道 `fixed_history`，4 道 `multi_turn_rollout`），`split=dev`，含 manifest 与 SHA-256 |
| `cases_test.json` | 2 道 `split=test` 的门禁样例，仅用于验证未解锁时被拒绝，**不要用于日常评测** |
| `cards_m0_m4.json` | M0–M4 条件卡：正确相关卡、无关产线卡、冲突/更新卡、指代不清卡，另含跨 namespace 与拒读卡用于验证 C 拦截 |
| `mock_responders.json` | 离线 Mock 的虚构回复（明确标注非模型输出） |

## 复跑

```sh
# 1. 校验数据与 manifest
uv run --no-sync -m lm_trainer.eval validate \
  --cases examples/vision_support/task_d_eval/cases_dev.json

# 2. 离线 Mock 跑完整评测（无 GPU、无权重）
uv run --no-sync -m lm_trainer.eval run \
  --cases examples/vision_support/task_d_eval/cases_dev.json \
  --mock --mock-responder-file examples/vision_support/task_d_eval/mock_responders.json \
  --model base --model sft_final \
  --output .cache/task-d-eval/mock-run

# 3. M0–M4 卡片对照（Mock）
uv run --no-sync -m lm_trainer.eval cards \
  --cases examples/vision_support/task_d_eval/cases_dev.json \
  --cards examples/vision_support/task_d_eval/cards_m0_m4.json \
  --mock --mock-responder-file examples/vision_support/task_d_eval/mock_responders.json \
  --model sft_final \
  --output .cache/task-d-eval/mock-cards

# 4. 打匿名盲评包
uv run --no-sync -m lm_trainer.eval blind \
  --run .cache/task-d-eval/mock-run/run.json \
  --output .cache/task-d-eval/mock-blind --seed 2026
```

真实推理（RTX 3090，本地 tiny Qwen2 fixture）见仓库根 `TASK_D_VALIDATION.md`。
