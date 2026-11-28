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

使用已有的 256 场景训练缓存。映射编码复用工作区的 ContactPose 锤击样例。 [ContactPose 数据说明](人工奖励函数运行说明.md#历史-contactpose-实验)

```bash
bank=weight/hammer_rl/reset_bank_direct_256.npz
run_dir="weight/hammer_rl/runs/teacher/vlm_$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$bank" ]; then
  python -m hammer_rl.prepare --output "$bank"
fi
```

## 3. 查看抓姿预览并确认训练

将 `hand_image` 改为人手抓锤照片文件或仅包含照片的目录路径。目录中的照片按文件名排序读取，初始奖励生成和每轮反思均读取全部照片。`mapping` 将 ContactPose 抓持先验映射到 OmniPicker 并导出结构化 latent。VLM 读取 latent 和人手照片，为整个训练缓存生成一份奖励配方，并选择关键点对照图要展示的候选位姿。 [奖励提示词](../vlm_reward/prompts/reward_design.md)

```bash
bank="/home/ubuntu20/CodeMujoco/hammering-task/weight/hammer_rl/reset_bank_v1_256.npz"
run_dir="weight/hammer_rl/runs/teacher/vlm_$(date +%Y%m%d_%H%M%S)"
hand_image="data/grasp_picture"
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy \
    -u ALL_PROXY -u all_proxy \
  python -m vlm_reward.reward_designer \
  --hand-image "$hand_image" --data-file "$bank" \
  --description "保持三次有效命中和终局持有，尽可能提高每一击的平均力" \
  --num-envs 2048 \
  --total-iterations 900 --max-iterations 1000 \
  --output-dir "$run_dir"
```

这是可视化查看立体3维人手和夹爪关键点对应的指令
```bash
python -m visualization.keypoint_preview \
  --mapping weight/hammer_rl/runs/teacher/vlm_20261123_195647/mapping.json \
  --candidate 0 --show
```

命令保存 `mapping.json`、`grasp_preview.png` 和 `reward_v1.json` 后暂停。核对同色指尖的对应关系及夹爪开爪位姿后，回到原终端输入 `yes` 启动训练。直接回车可取消。

在另一个已激活 `hammer` 环境的终端，从项目根目录运行下列命令打开交互窗口。把 `run_dir` 设为要查看的运行目录，`--candidate` 可取 `0` 或 `1`，VLM 选中的下标见 `reward_v1.json` 的 `generation.preview_candidate`。左键拖动任一视图，两侧会同步旋转。

```bash
run_dir=weight/hammer_rl/runs/teacher/vlm_20261123_195647
python -m visualization.keypoint_preview --mapping "$run_dir/mapping.json" \
  --candidate 0 --show
```

要导出静态图，可使用 `--output`。

```bash
python -m visualization.keypoint_preview --mapping "$run_dir/mapping.json" \
  --candidate 0 --output "$run_dir/grasp_preview_0.png"
```

预览使用缓存中锤柄尺寸的均值。Teacher 从全部 256 个场景采样，沿用各场景的 AnyGrasp 候选和固定锤击轨迹。每轮内循环使用同一份奖励配方，轮间由 VLM 根据评估反馈修正配方并续训。

`--num-envs` 设置同时运行的独立 MuJoCo 物理世界数。例如改成 `--num-envs 256`，Teacher 就并行运行 256 个世界，每个世界在重置时从训练缓存中抽取场景。从 checkpoint 续训时，在 `hammer_rl.train --resume` 命令中传入所需的 `--num-envs`。

`--total-iterations` 指定总训练 PPO 更新数，默认 10000；`--max-iterations` 指定每轮内循环的 PPO 更新数，默认 5000。外循环次数由两者相除向上取整，末轮只训练剩余次数。例如总计 12000、每轮 5000 时，依次训练 5000、5000、2000 次，进行两次奖励修正。总数不超过内循环轮次时，只训练和评估 V1，不修正奖励。

默认使用 512 个并行世界、种子 42 和 `cuda:0`。训练 checkpoint 保存在 `$run_dir/teacher_vN/`，默认两轮，最终阶段为 `teacher_v2`。下方评估命令按默认两轮填写，调整轮次后相应修改阶段目录。

## 4. 在独立测试场景评估 Teacher

```bash
test_bank=weight/hammer_rl/test_bank_v1_128.npz
if [ ! -f "$test_bank" ]; then
  python -m hammer_rl.prepare --test-set --output "$test_bank"
fi

python -m hammer_rl.evaluate --policy trained --selection sdk \
  --checkpoint "$run_dir/teacher_v2/model_best.pt" --data-file "$test_bank" \
  --split eval --episodes 0 --headless --output "$run_dir/test.npz"
```

终端输出里看 `task_success` 的均值（完整成功率），以及 `mean_force` 和 `strike_force_1` 到 `strike_force_3`（平均力度和各次敲击力度，单位 N）。
