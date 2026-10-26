# 锤击任务强化学习操作指南

训练流程：**离线准备数据 → 教师 PPO → 教师向学生蒸馏 → 学生 PPO 微调**。

所有命令均在项目根目录 `/home/ubuntu20/CodeMujoco/hammering-task` 执行。
请使用 `python -m hammer_rl.train` 等模块命令，不要直接执行 `python hammer_rl/train.py`。
各阶段按顺序执行，等待上一阶段成功完成后再启动下一阶段。

## 1. 激活运行环境

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer

export MUJOCO_GL=glfw
export MPLBACKEND=Agg
```

本机 Python 路径：`/home/ubuntu20/anaconda3/envs/hammer/bin/python`。
已核对的环境版本：mjlab 1.6.0、MuJoCo / MuJoCo Warp 3.11.0、
Torch 2.9.0+cu128、rsl-rl-lib 5.4.2。

项目已有验证记录使用 GLFW，EGL 初始化曾失败。准备数据时需要可用的图形显示环境；
评估的 `--headless` 参数只表示不打开回放窗口。

## 2. 准备训练数据

现有 `weight/hammer_rl/reset_bank.npz` 是流程验证缓存，仅含 8 个场景，
训练集和评估集各 4 个。以下另建数据文件，供较大规模训练使用。

```bash
python -m hammer_rl.prepare \
  --output weight/hammer_rl/reset_bank_256.npz \
  --num-scenes 256 \
  --candidates-per-scene 32 \
  --seed 42 \
  --workers 4
```

- 四类初始摆放按顺序轮流生成；每类内部按场景划分约 80% 训练集、20% 评估集。
- `--workers` 默认为 `1`（串行）；上例以两个独立进程并行生成场景，结果按场景编号汇总。
- 每个场景使用由 `--seed` 和场景编号派生的独立随机种子，初始采样不受进程数影响；
  相同种子的采样结果与旧版共享随机数生成器的方式不同。
- 每个场景最多保留 32 个候选，实际数量取决于 AnyGrasp 的输出。
- 此步骤包含落稳仿真、深度图采集、AnyGrasp 推理和抓取开度标定，完成后再开始训练。
- 依赖 `weight/local_runtime/` 下的 AnyGrasp 运行库和授权，以及
  `weight/pretrained_weights/anygrasp/checkpoint_detection.tar`。
- 重复执行会覆盖同名数据文件；训练及后续评估应使用同一份缓存。

## 3. 教师 PPO 训练

教师 actor 和 critic 使用特权观测。默认课程里程碑为 500、1200 次环境步；
当前每次更新采集一个环境步，因此也对应更新进度。

```bash
python -m hammer_rl.train \
  --stage teacher \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --num-envs 512 \
  --max-iterations 3000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/teacher/formal_01
```

从头成功完成 3000 次更新后，最终权重为：

```text
weight/hammer_rl/runs/teacher/formal_01/model_2999.pt
```

## 4. 教师向学生蒸馏

学生使用深度图和公开抓取候选信息，学习教师输出。此阶段默认开启物理随机化和深度噪声。

```bash
python -m hammer_rl.train \
  --stage distill \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --teacher-checkpoint weight/hammer_rl/runs/teacher/formal_01/model_2999.pt \
  --num-envs 128 \
  --max-iterations 1000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/distill/formal_01
```

从头成功完成 1000 次蒸馏更新后，最终权重为：

```text
weight/hammer_rl/runs/distill/formal_01/model_999.pt
```

## 5. 学生 PPO 微调

从蒸馏权重初始化学生 actor，再进行 PPO 训练。学生 actor 使用深度图和公开抓取候选信息，
critic 仍使用特权观测；此阶段默认开启物理随机化和深度噪声。

```bash
python -m hammer_rl.train \
  --stage student \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --init-from weight/hammer_rl/runs/distill/formal_01/model_999.pt \
  --num-envs 128 \
  --max-iterations 3000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/student/formal_01
```

从头成功完成 3000 次更新后，最终权重为：

```text
weight/hammer_rl/runs/student/formal_01/model_2999.pt
```

以上 3000 / 1000 / 3000 是项目提供的初始训练规模，不保证收敛。
若调整迭代数或使用续训，请以程序结束时打印的 `checkpoint:` 路径为准，
并同步修改下一阶段的输入权重路径。

## 6. 查看训练曲线

在另一个终端执行：

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
tensorboard --logdir weight/hammer_rl/runs --port 6006
```

本机浏览器打开 <http://localhost:6006>。

训练目录包含 TensorBoard 日志、`model_*.pt` 权重和 `run_config.json` 配置。
默认每 50 次更新保存权重，正常结束时也会保存最终权重。
每次新实验使用新的 `--log-dir`（例如 `formal_02`）；省略该参数时会自动创建时间戳目录。

## 7. 中断后继续训练

同一阶段继续训练使用 `--resume`。例如教师已经保存 `model_1499.pt`，
追加训练 1500 次更新：

```bash
python -m hammer_rl.train \
  --stage teacher \
  --resume weight/hammer_rl/runs/teacher/formal_01/model_1499.pt \
  --num-envs 128 \
  --max-iterations 1500 \
  --device cuda:0 \
  --seed 42
```

- 将 `--resume` 替换为实际存在的 checkpoint 路径；蒸馏和学生阶段也可用相同方式续训。
- `--max-iterations` 始终表示**本次追加的更新次数**，不是累计目标次数。
- 续训会在原权重目录写入结果，并从该目录的 `run_config.json` 恢复阶段、
  数据路径、锤头额外质量、课程里程碑和算法配置。保留权重旁的配置文件。
- `--num-envs`、`--device`、`--seed` 仍取本次命令参数，建议与原训练保持一致。
- `--teacher-checkpoint` 用于首次启动蒸馏，`--init-from` 用于首次启动学生 PPO；
  同阶段恢复时使用 `--resume`。

## 8. 评估训练效果

先运行零残差基线，再用相同数据、划分、样本数和随机种子评估学习策略。

```bash
# 零残差基线
python -m hammer_rl.evaluate \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --split eval \
  --policy zero \
  --episodes 128 \
  --num-envs 16 \
  --device cuda:0 \
  --seed 42 \
  --headless \
  --output weight/hammer_rl/eval_zero_formal_01.npz

# 教师模型
python -m hammer_rl.evaluate \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --split eval \
  --policy trained \
  --checkpoint weight/hammer_rl/runs/teacher/formal_01/model_2999.pt \
  --episodes 128 \
  --num-envs 16 \
  --device cuda:0 \
  --seed 42 \
  --headless \
  --output weight/hammer_rl/eval_teacher_formal_01.npz

# 最终学生模型
python -m hammer_rl.evaluate \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --split eval \
  --policy trained \
  --checkpoint weight/hammer_rl/runs/student/formal_01/model_2999.pt \
  --episodes 128 \
  --num-envs 16 \
  --device cuda:0 \
  --seed 42 \
  --headless \
  --output weight/hammer_rl/eval_student_formal_01.npz
```

终端会打印奖励及过程指标的均值、分位数和最大值；`--output` 保存逐样本原始指标。
评估默认关闭物理随机化和深度噪声；若要评估深度噪声，对各策略统一添加 `--noise`。
评估时务必显式指定训练使用的数据文件：当前评估入口不会自动从 run 配置恢复数据路径。

## 9. 可视化回放

在具有图形显示环境的终端执行：

```bash
python -m hammer_rl.evaluate \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --split eval \
  --policy trained \
  --checkpoint weight/hammer_rl/runs/student/formal_01/model_2999.pt \
  --episodes 1 \
  --device cuda:0 \
  --viewer native
```

## 10. 常用参数与操作提示

| 参数 | 含义 |
| --- | --- |
| `--num-envs 128` | 并行环境数；显存不足时可先降至 32 或 16 |
| `--max-iterations 3000` | 本次追加的更新次数 |
| `--device cuda:0` | 使用第 0 张 GPU |
| `--seed 42` | 随机种子 |
| `--log-dir 路径` | 新训练的日志及权重目录 |
| `--curriculum-milestones 600 1800` | 教师课程的两个切换里程碑 |
| `--hammer-head-extra-mass 0.0` | 锤头额外质量，单位 kg；准备数据和训练必须一致 |

- 训练入口没有 `--headless` 参数，按上面的命令启动即可。
- 一次策略动作会执行完整抓取和锤击尝试；当前最长覆盖 8980 个物理子步，
  不宜用普通单物理步环境的速度估计训练耗时。
- 若修改锤头额外质量，为其另建缓存和实验目录；评估已训练模型时会读取 run 中的额外质量设置。
  零残差基线则需要显式传入相同的 `--hammer-head-extra-mass`。
- 现有 `runs/*/smoke/` 权重仅验证流程，不代表已收敛策略。
- 256 场景、每场景最多 32 个候选是初始规模；是否需要更多数据或更新次数，依据评估结果决定。

相关实现：[训练入口](train.py)、[数据准备](prepare.py)、[评估入口](evaluate.py)、
[网络和算法配置](rl_cfg.py)、[环境配置](env_cfg.py)。
