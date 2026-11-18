# 从零训练 VLM 引导的锤击 Teacher

**准备场景 → VLM 生成奖励 → 从零训练 Teacher → 按反馈修订奖励并续训 → 独立测试。**

VLM 负责定奖励。Teacher 用 PPO 学习如何微调 AnyGrasp 给出的初始抓姿，每个回合只决策一次，随后由 MuJoCo 执行固定轨迹。目标是三次有效命中且结束时仍持有锤子，同时尽可能提高每一击的平均力。

## 1. 准备数据和 VLM 配置

在同一个终端按顺序执行。已有的训练场景缓存会直接复用，人手映射 JSON 已在仓库中。

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
export MUJOCO_GL=glfw
export MPLBACKEND=Agg

bank=weight/hammer_rl/reset_bank_v1_256.npz
mapping=data/human_grasp/processed/hammer_mapping.json
run_dir="weight/hammer_rl/runs/teacher/vlm_$(date +%Y%m%d_%H%M%S)"
scene="$run_dir/scene.json"
recipe="$run_dir/reward_v1.json"

if [ ! -f "$bank" ]; then
  python -m hammer_rl.prepare --output "$bank"
fi

python -m hammer_rl.prepare \
  --data-file "$bank" --scene-index 0 --scene-context "$scene"
```

设置 `OPENAI_API_KEY`。使用当前配置里的 `qwen4-max` 时，把 `OPENAI_BASE_URL` 设为提供该模型的 Responses API 基地址。所用服务需支持图像输入和 JSON Schema 输出。 [模型配置](../vlm_reward/configs/vlm.yaml)

## 2. 生成初始奖励并从零训练 Teacher

第一条命令用任务描述、RGB-D 和人手映射生成奖励 JSON。第二条启动全新的 PPO 训练，直接使用这份配方。 [初始提示词](../vlm_reward/prompts/reward_design.md)

```bash
python -m vlm_reward.reward_designer --task hammer \
  --mapping "$mapping" --scene-context "$scene" \
  --description "保持三次有效命中和终局持有，尽可能提高每一击的平均力" \
  --output "$recipe"

python -m hammer_rl.train --stage teacher \
  --data-file "$bank" --reward-recipe "$recipe" \
  --num-envs 512 --max-iterations 5000 --seed 42 \
  --log-dir "$run_dir/initial"
```

初始训练的 checkpoint 保存在 `$run_dir/initial/`，下一步用其中按训练奖励选出的 `model_best.pt`。

## 3. 按实测表现修订奖励并续训

每轮先评估 Teacher，让 VLM 根据失败阶段和力度提出新配方，再从同一个 checkpoint 分别用旧、新配方做 100 次 PPO 更新，保留实测表现更好的分支。当前按完整成功率、挥击前持有率、命中比例、较低掉落率、平均敲击力依次比较。 [修订提示词](../vlm_reward/prompts/reward_revision.md) · [分支比较](../vlm_reward/adaptive_train.py#L36)

```bash
python -m vlm_reward.adaptive_train \
  --data-file "$bank" --mapping "$mapping" --scene-context "$scene" \
  --initial-checkpoint "$run_dir/initial/model_best.pt" --initial-recipe "$recipe" \
  --rounds 3 --train-iters-per-round 100 --num-envs 512 --seed 42 \
  --output-dir "$run_dir/adaptive"
```

最终选中的权重和奖励配方在 `$run_dir/adaptive/rounds.jsonl` 的最后一行，字段分别是 `selected_checkpoint` 和 `selected_recipe`。

## 4. 在独立测试场景评估 Teacher

```bash
test_bank=weight/hammer_rl/test_bank_v1_128.npz
if [ ! -f "$test_bank" ]; then
  python -m hammer_rl.prepare --test-set --output "$test_bank"
fi

teacher_ckpt=$(tail -n 1 "$run_dir/adaptive/rounds.jsonl" | \
  python -c 'import json, sys; print(json.load(sys.stdin)["selected_checkpoint"])')

python -m hammer_rl.evaluate --policy trained --selection sdk \
  --checkpoint "$teacher_ckpt" --data-file "$test_bank" \
  --split eval --episodes 0 --headless --output "$run_dir/test.npz"
```

终端输出里看 `task_success` 的均值（完整成功率），以及 `mean_force` 和 `strike_force_1` 到 `strike_force_3`（平均力度和各次敲击力度，单位 N）。
