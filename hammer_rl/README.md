# 锤击任务强化学习操作指南

训练流程：**离线准备数据 → 教师 PPO → 教师向学生蒸馏 → 学生 PPO 微调 → 固定 actor 校准评分器**。

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

```bash
python -m hammer_rl.prepare \
  --output weight/hammer_rl/reset_bank_256.npz \
  --num-scenes 256 \
  --candidates-per-scene 32 \
  --seed 42 \
  --workers 4
```

## 3. 教师 PPO 训练

教师 actor 和 critic 使用特权观测。平放（`flat`）、锤头朝下斜靠（`lean_head_down`）、
锤头朝上斜靠（`lean_head_up`）和倒立（`inverted`）从首次 reset 起同时开放。
仅教师训练启用动态分配 SGS，仅使用训练样本统计。每类保留最近 512 个已完成尝试，
每条保存 `task_success`、`peak_force`（N）、`impact_impulse`（N·s）。
`task_success` 表示抓取、保持、有效命中且卸载结束前仍持有的完整任务成功，
课程不使用总奖励或单独的抬升成功率；`force` 奖励为 `peak_force / 10`，
权重为 30，即每 1 N 有效峰值力贡献 3 分；冲量用于指标和课程力度统计。
空历史的成功率为 0.5，初始概率四类均为 0.25。

课程参数保存在 `run_config.json` 的 `placement_sgs`，由现有课程项的 `params` 使用：

| 参数 | 值 |
| --- | --- |
| `history_size` | 512 |
| `warmup_samples_per_class` | 256 |
| `update_interval` | 10 |
| `target_success` | 0.9 |
| `force_scale_n` | 100.0 |
| `force_mix` | 0.7 |
| `strength_weight` | 0.5 |
| `priority_power` | 2.0 |
| `min_probability` | 0.075 |
| `max_probability` | 0.775 |
| `probability_smoothing` | 0.3 |
| `min_success_samples` | 32 |

对每类历史计算成功率 `s`。只有完整成功样本至少 32 条时，才启用力度项，
取成功回合峰值力中位数 `F` 和冲量中位数 `J`：

```text
d_force = 100 / (F + 100)
d_impulse = env.cfg.impulse_scale / (J + env.cfg.impulse_scale)
d_strength = 0.7 * d_force + 0.3 * d_impulse
d_success = clamp((0.9 - s) / 0.9, 0, 1)
priority = d_success + 0.5 * (1 - d_success) * d_strength
score = priority ** 2
```

成功不足 32 条时 `d_strength=0`，仅表示暂不启用力度项。失败回合即使保留正峰值力，
也不参与成功力度统计。100 N 是未经当前训练结果调优的峰值力归一化初值，
不是成功阈值或已标定的目标力；冲量尺度直接复用 `env.cfg.impulse_scale`（当前 1 N·s）。

四类历史都达到 256 条前保持均分；此后每 10 个含完成尝试的批次更新一次概率。
随机采样不保证固定批数完成热身。更新公式为：

```text
score.sum() == 0 时：target = [0.25, 0.25, 0.25, 0.25]
否则：
    raw = 0.075 + 0.70 * score / score.sum()
    excess = clamp_min(raw - 0.775, 0)
    receivers = (raw <= 0.775)
    target = raw - excess + excess.sum() * receivers / receivers.sum()
q_next = 0.7 * q + 0.3 * target
```

零成功类别优先级为 1；成功率达到 90% 后仍可因力度不足获得额外份额。
固定保底份额合计 30%（每类 7.5%），剩余 70% 按 score 动态分配。
7.5%～77.5% 是四类目标采样概率边界，实际概率经平滑逐步接近目标，不是每批整数名额；
256 个环境时，每类保底对应的期望数量为 19.2 次，实际数量会波动。
类内场景仍均匀采样，场景内候选也仍均匀采样。
每个训练场景权重为 `q[placement] / 该类训练场景数`，评估场景权重为零。
日志保留 `success_rate_*`、`probability_*`，增加 `strength_deficit_*`、`success_samples_*`。
调整训练资源分配不等于已经证明四类成功率或敲击力度提升。

新 teacher run 使用当前参数、空历史和均分热身；新策略 checkpoint 普通 `--resume`
恢复保存的参数、历史、概率和完成批次计数。旧 SGS checkpoint 切换须显式添加
`--reset-placement-sgs`，保留模型、优化器和迭代，仅采用当前课程配置并重新均分热身，
不自动迁移旧课程状态。具体续训方法见第 7 节。

```bash
python -m hammer_rl.train \
  --stage teacher \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --num-envs 256 \
  --max-iterations 3000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/teacher/formal_02
```

## 4. 教师向学生蒸馏

学生使用深度图和公开抓取候选信息，学习教师输出。此阶段默认开启物理随机化和深度噪声。

```bash
python -m hammer_rl.train \
  --stage distill \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --teacher-checkpoint weight/hammer_rl/runs/teacher/formal_02/model_2999.pt \
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
critic 也使用相同的深度和公开观测，输出一个价值，无动作分布。
actor 与 critic 使用各自的 CNN 和归一化统计；此阶段默认开启物理随机化和深度噪声。

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

### 5.1 固定 actor 校准候选评分器

教师和学生 PPO 仍随机采样一个候选训练。部署前固定最终 actor，逐个训练候选执行
actor 的确定性残差，以当前实际总奖励为标签，仅通过 MSE 更新 critic。
校准沿用对应阶段的随机化，双方观测归一化统计保持固定；重复执行分别保存观测与回报。
“最高分”表示预测的微调后总奖励最高，不等同于抓取成功率最高。

```bash
python -m hammer_rl.calibrate \
  --checkpoint weight/hammer_rl/runs/student/formal_01/model_best.pt \
  --num-envs 16 \
  --repeats 1 \
  --epochs 20 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --device cuda:0
```

也可将 checkpoint 换为已训练的教师，例如 `weight/hammer_rl/runs/teacher/formal_02/model_best.pt`。
默认只采集该缓存的训练划分，不使用留出的评估场景；`--data-file` 可指定缓存，
`--episodes 4 --epochs 1 --num-envs 4` 可限制为短运行。
默认输出到源 checkpoint 同目录的 `calibration_model_best/`，可用 `--output-dir` 指定新目录：

- `samples.pt`：执行前的 critic 输入、本次实际奖励、场景及候选编号。
- `deployment.pt`：配套 actor、校准后的 critic 及各自模型内的归一化统计。
- `run_config.json`：模型配置及本次校准参数。

`deployment.pt` 用于推理和评估，不包含 PPO 优化器状态，不用于 `--resume`。
actor 更新后，应重新校准。蒸馏 checkpoint 包含学生、教师和优化器状态，但没有 critic，仍是中间产物；
完整学生部署模型需要后续新版学生 PPO 和校准。不自动迁移旧的特权学生 critic。

三个阶段还会在运行目录保留 `model_best.pt`：教师和学生 PPO 按本轮
`Train/total_reward` 最高选优（多个采集步取均值），蒸馏按本轮 `Loss/behavior`
最低选优。每轮比较，仅严格改善时覆盖；周期和最终权重仍按原方式保存。
最佳权重包含完整训练状态，可用于评估、下一阶段初始化或 `--resume`。
`infos["best_score"]` 在 PPO 阶段保存奖励，在蒸馏阶段保存损失的负值；同一运行目录已有
`model_best.pt` 时沿用其最佳记录，否则从本次训练开始记录，不回溯旧权重。

## 6. 查看训练曲线

在另一个终端执行：

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
tensorboard --logdir weight/hammer_rl/runs/teacher/formal_02 --port 6006
```

本机浏览器打开 <http://localhost:6006>。

在 Scalars 界面选择 `Train/total_reward`，横轴使用 `Step`，将 Smoothing 设为 `0`，
即可查看每轮总奖励的变化（各并行环境的总奖励均值）。训练期间曲线随日志刷新。
该曲线从使用修改后脚本启动或继续训练时开始记录。

训练目录包含 TensorBoard 日志、`model_*.pt` 权重和 `run_config.json` 配置。
新训练默认教师和学生 PPO 每 100 次更新保存权重，蒸馏每 50 次；
续训以该目录 `run_config.json` 中的 `rl.save_interval` 为准。正常结束时也会保存最终权重。
每次新实验使用新的 `--log-dir`（例如 `formal_02`）；省略该参数时会自动创建时间戳目录。

## 7. 中断后继续训练

同一阶段继续训练使用 `--resume`。例如从已保存完整课程状态的教师
`finish_01/model_best.pt` 追加训练 3000 次更新：

```bash
python -m hammer_rl.train \
  --stage teacher \
  --resume weight/hammer_rl/runs/teacher/finish_01/model_best.pt \
  --num-envs 256 \
  --max-iterations 3000 \
  --device cuda:0
```

- 将 `--resume` 替换为实际存在的 checkpoint 路径；蒸馏和学生阶段也可用相同方式续训。
- `--max-iterations` 始终表示**本次追加的更新次数**，不是累计目标次数。
- 续训会在原权重目录写入结果，并从该目录的 `run_config.json` 恢复阶段、
  数据路径、锤头额外质量、教师 SGS 参数和算法配置。保留权重旁的配置文件；
  同时传入 `--log-dir` 也不会改变续训输出目录。
- 新策略教师周期和最终 checkpoint 均通过 `infos["placement_sgs"]` 保存 `history`、
  `probabilities`、`completed_batches`；普通续训恢复全部课程状态。初始化或恢复后的
  零长度 reset 不追加历史、不增加计数、不平滑概率，但仍把当前概率应用到场景权重。
- `--num-envs`、`--device`、`--seed` 仍取本次命令参数，建议与原训练保持一致。
- `--teacher-checkpoint` 用于首次启动蒸馏，`--init-from` 用于首次启动学生 PPO；
  同阶段恢复时使用 `--resume`。

## 8. 可视化回放

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task

python -m visualization.compare_hammering \
  --checkpoint weight/hammer_rl/runs/teacher/finish_01/model_best.pt \
  --placement flat \
  --split eval \
  --device cuda:0
```

`--checkpoint` 和 `--placement` 必填；四种放置参数如下：

| 参数值 | 放置类型 |
| --- | --- |
| `flat` | 平放 |
| `lean_head_down` | 斜靠，锤头朝下 |
| `lean_head_up` | 斜靠，锤头朝上 |
| `inverted` | 倒立 |

程序每轮从缓存的指定放置类型和 `--split` 划分中随机抽取三个场景，允许重复，复用缓存候选。
默认读取 checkpoint 旁配置中的数据路径，`--data-file` 可指定其他缓存；
当前准备脚本生成的候选已沿接近方向额外深入 30 mm，回放时不再生成候选或增加该偏移。
上排由 critic 选择候选并由 actor 微调，下排使用同场景、同候选集合的 SDK 最高分零残差，逐列配对。
场景标签 `T1/T2/T3` 和 `A1/A2/A3` 分别对应训练组和基线组。
窗口同时显示六个环境的场景编号、敲击峰值力（N）和加权总奖励；执行中数值随阶段更新，
完成后暂停并保留最终画面与数值，按 **回车** 从缓存重新抽样再比较。

## 9. 四类放置的命中率与峰值力对比 CSV

在项目根目录、已激活 `hammer` 环境后，只需指定要对比的权重：

```bash
python -m hammer_rl.compare_evaluations \
  --checkpoint weight/hammer_rl/runs/teacher/finish_01/model_best.pt \
  --data-file weight/hammer_rl/eval_bank_128.npz \
  --anygrasp-extra-depth 0.03
```

分别运行 SDK 最高分＋零残差、critic 选择＋actor 残差两组，每组每场景执行一次。
省略 `--data-file` 时使用 `reset_bank_256.npz` 留出的 52 个场景，每场景 32 个候选；
上例指定的 `eval_bank_128.npz` 包含四类放置各 32 个测试场景，每场景 32 个候选。
候选经过 AnyGrasp NMS，再以最高分位姿为起点，按平移和旋转距离做最远点采样。
可用 `--episodes 4 --num-envs 4` 做短运行，或 `--data-file` 指定其他多候选缓存。

默认输出到 checkpoint 同目录的 `comparison_<checkpoint文件名去扩展名>/`，
可用 `--output-dir` 指定目录。最终仅生成 `metrics_comparison.csv`，按四类放置保存两组的
有效命中率（`valid_strike` 的均值，百分比）、平均有效峰值力（`peak_force` 的均值，N）
以及训练策略减 SDK 基线的差值；命中率差值单位为百分点。峰值力均值包含未命中的零值。
过程中的 `zero.npz` 和 `trained.npz` 在读取后删除，不生成 `selected.npz` 或 `trained_rewards.csv`。
两组按同一场景顺序配对，候选编号允许不同；此对比同时包含候选选择和 actor 微调的效果，
不分别报告二者的收益，也不导出抓取成功率、完整任务成功率或奖励明细。
`--anygrasp-extra-depth` 只调整 SDK 基线的额外深入量，训练策略沿用缓存中的候选深度。

单独评估学习选择：

```bash
python -m hammer_rl.evaluate \
  --data-file weight/hammer_rl/reset_bank_256.npz \
  --policy trained --selection learned \
  --checkpoint weight/hammer_rl/runs/teacher/formal_02/calibration_model_best/deployment.pt \
  --split eval --episodes 0 --num-envs 16 --headless \
  --output weight/hammer_rl/selected_eval.npz
```

`--selection all` 是原有逐候选评估，也是默认值；`sdk` 和 `learned` 每场景执行一次。
`learned` 使用包含 actor 和 critic 的教师/学生 PPO 或部署权重；`all/sdk` 仍支持蒸馏 actor。
评估默认名义物理参数，`--noise` 开启深度噪声，`--seed` 默认 42。

### 9.1 学生在线推理接口

```python
from hammer_rl.inference import load_models, select_grasp

actor, scorer = load_models("weight/hammer_rl/runs/student/formal_01/calibration_model_best/deployment.pt")
result = select_grasp(actor, scorer, depth, scene, grasps)
```

`depth` 为米单位 `(1, 120, 160)` 深度；`scene` 包含未批量的 `finger_pos`、`finger_vel`、
`camera_pos`、`camera_quat`、`intrinsics`、`target_pos`、`target_quat`，字段与训练公开观测一致。
位姿使用世界坐标，四元数为 wxyz，内参顺序为 fx、fy、cx、cy。
手指状态使用与 reset 中按模型关节顺序取得的相同顺序。
`grasps` 沿用 `generate_grasps` 返回的世界坐标候选字典，内部复用
`candidate_to_mount(..., extra_depth=0.03)`，不读取仿真锤子真值。

返回张量包含 `candidate_index`（输入集合内编号）、`score`、所有候选 `scores`、
6 维原始 `action`、最终 `mount_pos/mount_quat`、`pregrasp_pos`、`width` 和 `gripper_open_ctrl`。
除全部候选评分外，单场景结果保留首维 1；CPU 执行器可取第 0 项并转换为 NumPy 使用最终位姿和开合量。
教师评分继续使用特权观测，通过仿真内的 `select_candidates` 入口调用。

## 10. 常用参数与操作提示

| 参数 | 含义 |
| --- | --- |
| `--num-envs 128` | 并行环境数；显存不足时可先降至 32 或 16 |
| `--max-iterations 3000` | 本次追加的更新次数 |
| `--device cuda:0` | 使用第 0 张 GPU |
| `--seed 42` | 随机种子 |
| `--log-dir 路径` | 新训练的日志及权重目录 |
| `--hammer-head-extra-mass 0.0` | 锤头额外质量，单位 kg；准备数据和训练必须一致 |

- 训练入口没有 `--headless` 参数，按上面的命令启动即可。
- 一次策略动作会执行完整抓取和锤击尝试；当前最长覆盖 8980 个物理子步，
  不宜用普通单物理步环境的速度估计训练耗时。
- 若修改锤头额外质量，为其另建缓存和实验目录；评估已训练模型时会读取 run 中的额外质量设置。
  零残差基线则需要显式传入相同的 `--hammer-head-extra-mass`。
- 现有 `runs/*/smoke/` 权重仅验证流程，不代表已收敛策略。
- 256 场景、每场景最多 32 个候选是初始规模；是否需要更多数据或更新次数，依据评估结果决定。

相关实现：[训练入口](train.py)、[数据准备](prepare.py)、[评分校准](calibrate.py)、
[选择与微调](inference.py)、[评估入口](evaluate.py)、
[网络和算法配置](rl_cfg.py)、[环境配置](env_cfg.py)、[四类对比](compare_evaluations.py)、
[六场景展示](../visualization/compare_hammering.py)。
