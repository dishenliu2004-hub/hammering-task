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
  --output weight/hammer_rl/reset_bank_dr_256.npz \
  --num-scenes 256 \
  --candidates-per-scene 32 \
  --seed 42 \
  --workers 4
```

锤子位姿在 L 型支架局部坐标系连续采样：`x∈[0.12,0.42] m`、
`y∈[-0.10,0.10] m`，yaw、roll 覆盖 `[-π,π]`，tilt 覆盖 `[-π/2,π/2]`。
旋转按 `Rz(yaw) @ Ry(-tilt) @ Rx(roll)` 组合，初始高度由现有碰撞几何投影确定，
与底板或挡板留 1 mm 间隙；不限制锤体全部位于底板投影内。
每个候选先沉降 2 s，再观察 0.5 s；仅接受窗口首末位移小于 5 mm、
最短旋转角变化小于 3°，且末态仍与支架底板或挡板接触的结果，否则重新采样。
这些阈值是算法初值，尚未通过本次任务的实验验证。

缓存保存观察窗口结束后的状态、`(1,120,160)` 深度和原有 AnyGrasp 候选及 `offsets`，
不再保存 `placement`、`difficulty`。全部场景由 `--seed` 可复现地置乱，
约 20% 标记为 eval，其余为 train；训练 reset 只抽取离线缓存。

每个场景在编译前分别对锤柄 capsule 半径和圆柱段半长乘以 `U(0.95,1.05)`，
此形态在沉降重采样中保持固定，深度和 AnyGrasp 候选均来自该实际模型。
新 bank 增加场景级 `handle_size`、`handle_convex_planes`；reset 按 `scene_id` 恢复
锤柄尺寸及碰撞包围体，包络评分使用同一场景缓存的凸包。
旧 bank 缺少必需字段，必须重新生成；使用新的输出路径保留旧数据。
新训练、校准及评估须明确指向同一版 bank；另建的评估 bank 也须由新版 prepare 生成。
权重张量格式不变，但旧权重未针对新随机化域训练，建议用新 bank 从头完成三阶段训练。

教师、蒸馏和学生在每次 reset 使用同一在线物理分布：质量/惯量倍率 `U(0.90,1.10)`、
同环境各指面共享摩擦倍率 `U(0.80,1.20)`，两侧主驱动关节共享开口端扩展量
`U(0,0.02) rad` 与闭合端收紧量 `U(0,0.06) rad`，每次从名义限位计算镜像范围。
教师不加深度噪声，蒸馏和学生保留已有 `DepthNoise`；SGS 仍以固定形态的场景为单位。

## 3. 教师 PPO 训练

教师 actor 和 critic 使用特权观测。所有连续稳定位姿的训练场景从首次 reset 起开放。
仅教师训练启用逐场景 SGS，以缓存行号 `scene_id` 为唯一统计键，
每个训练场景保留最近 100 个已完成尝试的二值 `task_success`。
`task_success` 表示抓取、保持、有效命中且卸载结束前仍持有的完整任务成功，
课程仅使用该成功历史；原有力奖励和冲量指标保持不变。
场景无历史时成功率先验为 0.5，首次更新前均匀采样全部训练场景。

课程参数保存在 `run_config.json` 的 `scene_sgs`，由现有课程项的 `params` 使用：

| 参数 | 值 |
| --- | --- |
| `history_size` | 100 |
| `update_interval` | 10 |
| `target_success` | 0.5 |
| `concentration` | 1.0 |
| `temperature` | 2.0 |
| `epsilon` | 1e-4 |

每 10 个包含完成回合的批次更新一次概率，无额外热身。仅在训练场景间计算：

```text
p_i = mean(last_100_task_success_i)  # 无历史为 0.5
w_i = (p_i + eps) ** (kappa * t) * (1 - p_i + eps) ** (kappa * (1 - t))
logit_i = log(max(w_i, eps) + eps) / T
q_train = softmax(logit)
q_eval = 0
```

其中 `t=0.5`、`kappa=1.0`、`T=2.0`、`eps=1e-4`。
成功率接近 0.5 的场景优先级较高，接近 0 或 1 的场景优先级较低。
`q_i` 是场景采样概率，不是成功率；每次 reset 直接使用对应的场景权重，
场景内候选继续均匀采样。评估场景不累计历史，采样权重为零。
同一场景的并行完成回合全部计入历史，初始化和恢复后的零长度 reset 不计入。
日志仅输出 `sgs_mean_success`、`sgs_mean_probability`、`sgs_min_probability`、
`sgs_max_probability`、`sgs_observed_scenes`，不逐场景生成日志键。

新 teacher run 使用当前参数、空历史和均匀初始概率；新 SGS checkpoint 普通 `--resume`
恢复保存的参数、逐场景历史、概率和完成批次计数。旧 SGS 状态不兼容新结构，
切换时须显式添加 `--reset-scene-sgs` 并通过 `--data-file` 指定新的连续场景缓存，
保留模型、优化器和迭代，清空课程状态并采用当前 SGS 参数，不自动迁移旧课程状态。
具体续训方法见第 7 节。

```bash
python -m hammer_rl.train \
  --stage teacher \
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
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
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
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
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
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
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
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
tensorboard --logdir weight/hammer_rl/runs/teacher/finish_01 --port 6006
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
- 新策略教师周期和最终 checkpoint 均通过 `infos["scene_sgs"]` 保存 `history`、
  `probabilities`、`completed_batches`；普通续训恢复全部课程状态。初始化或恢复后的
  零长度 reset 不追加历史、不增加计数、不更新概率，但仍把当前概率应用到场景权重。
- 显式使用 `--reset-scene-sgs --data-file 新缓存路径` 时，采用本次命令的数据路径并写回
  `run_config.json`，重置逐场景课程状态；不加此开关时继续恢复原运行的数据路径。
- `--num-envs`、`--device`、`--seed` 仍取本次命令参数，建议与原训练保持一致。
- `--teacher-checkpoint` 用于首次启动蒸馏，`--init-from` 用于首次启动学生 PPO；
  同阶段恢复时使用 `--resume`。

## 8. 可视化回放

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task

python -m visualization.compare_hammering \
  --checkpoint weight/hammer_rl/runs/teacher/finish_01/model_best.pt \
  --split eval \
  --device cuda:0
```

`--checkpoint` 必填；从指定 `--split` 随机抽取三个连续场景，
以六环境成对展示学习选择＋残差与 AnyGrasp 零残差。

完成后暂停并保留最终画面与数值，按 **回车** 从缓存重新抽样再比较。

## 9. 全部评估场景的命中率与峰值力对比 CSV

在项目根目录、已激活 `hammer` 环境后，只需指定要对比的权重：

```bash
python -m hammer_rl.compare_evaluations \
  --checkpoint weight/hammer_rl/runs/teacher/finish_01/model_best.pt \
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
  --output-dir weight/hammer_rl/runs/teacher/finish_01/comparison_01 \
  --anygrasp-extra-depth 0.03
```

分别运行 SDK 最高分＋零残差、critic 选择＋actor 残差两组，每组每场景执行一次。
该入口默认文件名仍为 `reset_bank_256.npz`，本次应按上例显式指定新 bank，
使用其中留出的约 20% 场景（256 场景时为 51 个），每场景最多 32 个候选。
自定义评估 bank 也须由新版 prepare 生成，实际评估范围由其中 `split` 决定。
候选经过 AnyGrasp NMS，再以最高分位姿为起点，按平移和旋转距离做最远点采样。
可用 `--episodes 4 --num-envs 4` 做短运行，或 `--data-file` 指定其他多候选缓存。

默认输出到 checkpoint 同目录的 `comparison_<checkpoint文件名去扩展名>/`，
可用 `--output-dir` 指定目录。最终仅生成 `metrics_comparison.csv`，以一个 `all` 行汇总两组的
有效命中率（`valid_strike` 的均值，百分比）、平均有效峰值力（`mean_force` 的均值，N）
以及训练策略减 SDK 基线的差值；命中率差值单位为百分点。峰值力均值包含未命中的零值。
过程中的 `zero.npz` 和 `trained.npz` 在读取后删除，不生成 `selected.npz` 或 `trained_rewards.csv`。
两组按同一场景顺序配对，候选编号允许不同；此对比同时包含候选选择和 actor 微调的效果，
不分别报告二者的收益，也不导出抓取成功率、完整任务成功率或奖励明细。
`--anygrasp-extra-depth` 只调整 SDK 基线的额外深入量，训练策略沿用缓存中的候选深度。

单独评估学习选择：

```bash
python -m hammer_rl.evaluate \
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
  --policy trained --selection learned \
  --checkpoint weight/hammer_rl/runs/teacher/formal_02/calibration_model_best/deployment.pt \
  --split eval --episodes 0 --num-envs 16 --headless \
  --output weight/hammer_rl/selected_eval.npz
```

`--selection all` 是原有逐候选评估，也是默认值；`sdk` 和 `learned` 每场景执行一次。
`learned` 使用包含 actor 和 critic 的教师/学生 PPO 或部署权重；`all/sdk` 仍支持蒸馏 actor。
评估默认关闭在线物理/限位随机化，但仍按评估 bank 恢复离线锤柄形态，
因此名义物理参数评估仍包含 bank 中的形态变化；`--noise` 开启深度噪声，`--seed` 默认 42。

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
[网络和算法配置](rl_cfg.py)、[环境配置](env_cfg.py)、[场景总体对比](compare_evaluations.py)、
[六场景展示](../visualization/compare_hammering.py)。
