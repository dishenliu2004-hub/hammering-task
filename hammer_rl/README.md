# 从零训练 VLM 引导的锤击 Teacher

**准备场景 → 生成抓姿预览与奖励 → 确认后从零训练 Teacher → 独立测试。**

VLM 负责定奖励。Teacher 用 PPO 学习如何微调 AnyGrasp 给出的初始抓姿，每个回合只决策一次，随后由 MuJoCo 执行固定轨迹。目标是三次有效命中且结束时仍持有锤子，同时尽可能提高每一击的平均力。

## 1. 激活环境并输入 VLM 地址和密钥

打开 Bash 终端，按顺序执行下面的命令。后续步骤都在这个终端里运行。

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
export MUJOCO_GL=osmesa
export MPLBACKEND=Agg
```

当前模型名是 `qwen4-max`。准备能访问这个模型的服务地址和密钥。服务需支持 Responses API、图像输入和 JSON Schema 输出。使用其他模型时，修改 `model:` 字段。 [模型配置](../vlm_reward/configs/vlm.yaml)

将下面整段复制到终端。出现第一个提示时，粘贴服务商给出的 **API 基地址**并回车，SDK 会自动追加 `/responses`。出现第二个提示时，粘贴 **API Key** 并回车。输入密钥时终端不会显示字符。

```bash
{
  read -r -p '请输入 VLM API 基地址：' OPENAI_BASE_URL
  read -r -s -p '请输入 VLM API Key：' OPENAI_API_KEY
  printf '\n'
  export OPENAI_BASE_URL OPENAI_API_KEY
}
```

## 2. 准备训练数据

使用已有的 256 场景训练缓存。映射图复用工作区的 ContactPose 锤击样例。 [ContactPose 数据说明](人工奖励函数运行说明.md#历史-contactpose-实验)

```bash
bank=weight/hammer_rl/reset_bank_v1_256.npz
run_dir="weight/hammer_rl/runs/teacher/vlm_$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$bank" ]; then
  python -m hammer_rl.prepare --output "$bank"
fi
```

## 3. 查看抓姿预览并确认训练

将 `hand_image` 改为实际的人手抓锤照片路径。`mapping` 将 ContactPose 抓持先验映射到 OmniPicker 并生成对照图。VLM 读取映射图和人手照片，为整个训练缓存生成一份奖励配方。 [奖励提示词](../vlm_reward/prompts/reward_design.md)

```bash
hand_image="/home/ubuntu20/下载/1.png"
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy \
    -u ALL_PROXY -u all_proxy \
  python -m vlm_reward.reward_designer \
  --hand-image "$hand_image" --data-file "$bank" \
  --description "保持三次有效命中和终局持有，尽可能提高每一击的平均力" \
  --num-envs 2048 \
  --output-dir "$run_dir"
```

命令保存 `mapping.json`、`grasp_preview.png` 和 `reward_v1.json` 后暂停。打开预览核对人手骨架、接触区域和夹爪开爪位姿，然后回到原终端输入 `yes` 启动训练。直接回车可取消。

预览使用缓存中锤柄尺寸的均值。Teacher 从全部 256 个场景采样，沿用各场景的 AnyGrasp 候选和固定锤击轨迹，全程使用同一份奖励配方。

`--num-envs` 设置同时运行的独立 MuJoCo 物理世界数。例如改成 `--num-envs 256`，Teacher 就并行运行 256 个世界，每个世界在重置时从训练缓存中抽取场景。从 checkpoint 续训时，在 `hammer_rl.train --resume` 命令中传入所需的 `--num-envs`。

默认使用 512 个并行世界、5000 次 PPO 更新、种子 42 和 `cuda:0`。训练 checkpoint 保存在 `$run_dir/teacher/`。

## 4. 在独立测试场景评估 Teacher

```bash
test_bank=weight/hammer_rl/test_bank_v1_128.npz
if [ ! -f "$test_bank" ]; then
  python -m hammer_rl.prepare --test-set --output "$test_bank"
fi

python -m hammer_rl.evaluate --policy trained --selection sdk \
  --checkpoint "$run_dir/teacher/model_best.pt" --data-file "$test_bank" \
  --split eval --episodes 0 --headless --output "$run_dir/test.npz"
```

终端输出里看 `task_success` 的均值（完整成功率），以及 `mean_force` 和 `strike_force_1` 到 `strike_force_3`（平均力度和各次敲击力度，单位 N）。
