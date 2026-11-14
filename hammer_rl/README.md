# 锤击任务强化学习操作指南

训练流程：**准备 A 场景 → 教师 PPO → 教师向学生蒸馏 → 学生 PPO 微调 → 固定 actor 校准评分器**。

## 1. 激活运行环境

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer

export MUJOCO_GL=glfw
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
```

## 2. 准备 A 场景和测试集

按顺序生成 256 个训练/评估场景和 128 个独立测试场景。
生成过程中已在相机采集和 AnyGrasp 之前完成每个待保存状态的恢复复核。
以下命令使用单进程生成，降低显存和内存峰值；不要同时运行两次生成命令。
每份 NPZ 仅在该次全部场景生成完成后写出，中断后需从头生成。

```bash
# 生成训练/评估缓存：四类各 45 个，扩展来源 76 个。
python -m hammer_rl.prepare \
  --output weight/hammer_rl/reset_bank_v1_256.npz \
  --num-scenes 256 \
  --legacy-per-placement 45 \
  --random-scenes 76 \
  --candidates-per-scene 32 \
  --seed 42 \
  --workers 1

# 独立生成测试集：128 个场景，使用独立种子 43。
python -m hammer_rl.prepare --test-set \
  --output weight/hammer_rl/test_bank_v1_128.npz \
  --candidates-per-scene 32 \
  --seed 43 \
  --workers 1
```

A 按 `source_type` 保存四类教师 v1（8089d95）姿态各 45 个（0 横躺、1 锤头朝下斜靠、
2 锤头朝上斜靠、3 倒立），以及按 v1 接触几何扩展的姿态 76 个（4）。
四类公式使用实际随机化后的锤柄半径、半长计算接触位置。
锤头朝上斜靠使用 v1 的顶部支撑分支，倾角由锤柄尺寸及底板、挡板高度联动求解，
使锤柄落在底板、锤头由挡板顶面支撑，避免板面侧靠产生慢滑。
来源 4 等概率选择横躺、锤头朝上斜靠、倒立，在原支架局部
`x∈[0.12,0.42]`、`y∈[-0.10,0.10]` 和角度边界内，按底板与锤体投影求可行平移范围；
位置与角度由接触关系联动，不再独立随机转角后抬高下落。
锤头朝下斜靠的 v1 局部 x 小于 0.12，仍由来源 1 保留。
所有来源先沉降 2 s，再观察 30 s，每 20 ms 比较相对观察起点的累计位姿变化：
位移 ≤3 mm、转角 ≤2°，且各采样点均有 GPU 有效支架接触，才将窗口末态确定为待保存状态 S。
随后快照保存全部 `STATE_FIELDS`、锤柄尺寸、模型参考位姿及实际锤子位姿，
释放初筛仿真，按缓存恢复流程重建单场景仿真，从 S 再复核 30 s，采用相同的稳定条件。
未通过时沿用原重采样流程，不采图或调用 AnyGrasp；通过后再次恢复同一个 S，时间归零，
同步 CPU 数据，再执行可见性筛选、采图和候选生成，最终保存 S。
保存真实速度及完整状态，不冻结物体或清零速度，也不保存复核推进后的末态。
支架两块板的 `solref/solimp` 恢复为 v1 默认软度，生成、检查与训练共用同一模型。
同一场景保留原锤柄形态随机化、可见性筛选、深度、AnyGrasp SDK/NMS 和覆盖采样。
分层划分 train/eval：四类各 36/9，随机来源 61/15，总计 205/51。
`--num-scenes` 应等于 `4 * --legacy-per-placement + --random-scenes`；小规模测试可用 4/1/0。

A 缓存保留 `offsets` 和候选字段：原始 `translations/rotations/depths/widths/scores`、
`local_pos/local_quat/open_ctrl` 与转换后的 `mount_pos/mount_quat`。候选沿接近方向
在 SDK 原深度上额外深入 30 mm，reset 不重复叠加。
所有位置以米计，四元数为 wxyz。未指定 `--output` 时，脚本仍默认使用历史文件名
`reset_bank_ab_256.npz` 和 `test_bank_ab_128.npz`，二者实际只包含 A 场景。
旧缓存不会因脚本更新而自动改变，需要重新生成。上面的命令通过 `--output` 写入
`reset_bank_stable_v1_256.npz` 和 `test_bank_stable_v1_128.npz`，保留旧缓存；
后续带 `--data-file` 的正式训练和评估示例也使用新训练缓存路径。

可视化查看生成后的训练/评估缓存：

```bash
python -m visualization.hammer_reset_bank \
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --scene-index 135 --wait-time 30
```

可选：生成后使用独立检查工具复查已有缓存，常规生成无需另行执行。

```bash
python -m hammer_rl.check_reset_bank \
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --duration 30

python -m hammer_rl.check_reset_bank \
  --data-file weight/hammer_rl/test_bank_stable_v1_128.npz \
  --duration 30
```

独立复查各覆盖缓存中的全部场景，使用与生成时相同的 MuJoCo Warp 后端额外仿真 30 s，
不重新采样或改写原 NPZ。每 20 ms 记录位姿、速度及有效支架接触，分别输出至 bank 同目录的
`reset_bank_stable_v1_256_stability_30s/` 和 `test_bank_stable_v1_128_stability_30s/`
（`summary.json`、逐场景 `scenes.csv`、`traces.npz`）。
累计漂移以相对缓存超过 3 mm 或 2° 计；`passes_stability` 同时要求全程有效支架接触，
`fails_stability` 应为 0。另外保留最后 0.5 s 的历史 5 mm / 3° 判定及
1 mm / 1° 窗口指标，便于对比旧报告。浏览器默认也额外等待 30 s，显示累计漂移；
N/P 切换场景、空格暂停/继续、R 从缓存重播，`--wait-time 0` 查看静态缓存。

蒸馏和学生每次 reset 使用物理随机化：质量/惯量 `U(0.90,1.10)`、
指面摩擦 `U(0.80,1.20)`、扭簧刚度 `U(0.80,1.20)` 倍率。
每个环境的窄指和宽指扭簧共用一个刚度倍率，每次 reset 均基于名义刚度
`0.2 N·m/rad` 重新采样，范围为 `0.16～0.24 N·m/rad`。
学生和蒸馏保留深度噪声，SGS 仍以 A 场景为单位。

### 2.1 策略与动作

actor 仅在 A 起点输出一次 6D 抓取残差，保留原平移和旋转尺度。
公开/特权观测分别为 50/87 维，在 reset 时生成并冻结。
抓取抬升后沿固定分段轨迹搬运、调转朝向、下降至挥击起点并保持 1 s，
随后执行原抛物线挥击、卸载和撤离，过程中不重新调用策略。

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
| `target_success` | 0.0 |
| `concentration` | 1.0 |
| `temperature` | 1.0 |
| `epsilon` | 1e-4 |
| `uniform_mix` | 0.2 |

每 10 个包含完成回合的批次更新一次概率，无额外热身。仅在训练场景间计算：

```text
p_i = mean(last_100_task_success_i)  # 无历史为 0.5
w_i = (p_i + eps) ** (kappa * t) * (1 - p_i + eps) ** (kappa * (1 - t))
logit_i = log(max(w_i, eps) + eps) / T
q_adaptive = softmax(logit)
q_train = 0.8 * q_adaptive + 0.2 / N_train
q_eval = 0
```

其中 `t=0.0`、`kappa=1.0`、`T=1.0`、`eps=1e-4`、`uniform_mix=0.2`，
`N_train` 为训练场景数。成功率越接近 0，场景优先级越高；忽略数值处理项时，
80% 的采样概率按失败率分配，其余 20% 均匀分配给全部训练场景。
`q_i` 是场景采样概率，不是成功率；每次 reset 直接使用对应的场景权重，
场景内候选继续均匀采样。评估场景不累计历史，采样权重为零。
同一场景的并行完成回合全部计入历史，初始化和恢复后的零长度 reset 不计入。
日志仅输出 `sgs_mean_success`、`sgs_mean_probability`、`sgs_min_probability`、
`sgs_max_probability`、`sgs_observed_scenes`，不逐场景生成日志键。

新 teacher run 使用当前参数、空历史和均匀初始概率；新 SGS checkpoint 普通 `--resume`
恢复保存的参数、逐场景历史、概率和完成批次计数。旧 SGS 配置或状态切换到当前策略时，
须显式添加 `--reset-scene-sgs` 并通过 `--data-file` 指定场景缓存，可沿用原缓存，
保留模型、优化器和迭代，清空课程状态并采用当前 SGS 参数，不自动迁移旧课程状态。
具体续训方法见第 7 节。

```bash
python -m hammer_rl.train \
  --stage teacher \
  --data-file weight/hammer_rl/reset_bank_v1_256.npz \
  --num-envs 512 \
  --max-iterations 5000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/teacher/a_only_no_guide_01
```

## 4. 教师向学生蒸馏

学生使用深度图和公开抓取候选信息，学习教师输出。此阶段默认开启物理随机化和深度噪声。

```bash
python -m hammer_rl.train \
  --stage distill \
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --teacher-checkpoint weight/hammer_rl/runs/teacher/a_only_01/model_2999.pt \
  --num-envs 128 \
  --max-iterations 1000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/distill/a_only_01
```

从头成功完成 1000 次蒸馏更新后，最终权重为：

```text
weight/hammer_rl/runs/distill/a_only_01/model_999.pt
```

## 5. 学生 PPO 微调

从蒸馏权重初始化学生 actor，再进行 PPO 训练。学生 actor 使用深度图和公开抓取候选信息，
critic 也使用相同的深度和公开观测，输出一个价值，无动作分布。
actor 与 critic 使用各自的 CNN 和归一化统计；此阶段默认开启物理随机化和深度噪声。

```bash
python -m hammer_rl.train \
  --stage student \
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --init-from weight/hammer_rl/runs/distill/a_only_01/model_999.pt \
  --num-envs 128 \
  --max-iterations 3000 \
  --device cuda:0 \
  --seed 42 \
  --log-dir weight/hammer_rl/runs/student/a_only_01
```

从头成功完成 3000 次更新后，最终权重为：

```text
weight/hammer_rl/runs/student/a_only_01/model_2999.pt
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
  --checkpoint weight/hammer_rl/runs/student/a_only_01/model_best.pt \
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --num-envs 16 \
  --repeats 1 \
  --epochs 20 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --device cuda:0
```

也可将 checkpoint 换为已训练的教师，例如 `weight/hammer_rl/runs/teacher/a_only_01/model_best.pt`。
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

## 6. 生成总奖励曲线

在另一个终端执行：

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
python visualization/hammer_rl_training.py \
  --log-dir weight/hammer_rl/runs/teacher/a_only_01
```

读取已保存日志中的 `Train/total_reward`，生成一张 PNG 折线图：横轴为日志中的原始训练轮次，
纵轴为各并行环境的总奖励均值，不进行平滑。输出路径为：

```text
weight/hammer_rl/runs/teacher/a_only_01/total_reward_per_iteration.png
```

图片反映执行命令时已保存的日志，重新运行即可更新并覆盖同名图片。
该曲线从使用修改后脚本启动或继续训练时开始记录。

训练目录包含 TensorBoard 日志、`model_*.pt` 权重和 `run_config.json` 配置。
新训练默认教师和学生 PPO 每 100 次更新保存权重，蒸馏每 50 次；
续训以该目录 `run_config.json` 中的 `rl.save_interval` 为准。正常结束时也会保存最终权重。
每次新实验使用新的 `--log-dir`（例如 `a_only_02`）；省略该参数时会自动创建时间戳目录。

## 7. 中断后继续训练

以下续训适用于 50/87 维公开/特权观测、6D 输出模型。同一阶段继续训练使用 `--resume`。例如从已保存完整课程状态的教师
`a_only_01/model_best.pt` 追加训练 3000 次更新：

```bash
python -m hammer_rl.train \
  --stage teacher \
  --resume weight/hammer_rl/runs/teacher/a_only_no_guide_01/model_best.pt \
  --num-envs 512 \
  --max-iterations 1500 \
  --device cuda:0
```

- 将 `--resume` 替换为实际存在的 checkpoint 路径；蒸馏和学生阶段也可用相同方式续训。
- `--max-iterations` 始终表示**本次追加的更新次数**，不是累计目标次数。
- 续训会在原权重目录写入结果，并从该目录的 `run_config.json` 恢复阶段、
  A 数据路径、锤头额外质量、教师 SGS 参数和算法配置。保留权重旁的配置文件；
  同时传入 `--log-dir` 也不会改变续训输出目录。
- 新策略教师周期和最终 checkpoint 均通过 `infos["scene_sgs"]` 保存 `history`、
  `probabilities`、`completed_batches`；普通续训恢复全部课程状态。初始化或恢复后的
  零长度 reset 不追加历史、不增加计数、不更新概率，但仍把当前概率应用到场景权重。
- 显式使用 `--reset-scene-sgs --data-file 场景缓存路径` 时，采用本次命令的数据路径并写回
  `run_config.json`，重置逐场景课程状态；不加此开关时继续恢复原运行的数据路径。
- `--num-envs`、`--device`、`--seed` 仍取本次命令参数，建议与原训练保持一致。
- `--teacher-checkpoint` 使用同结构教师启动蒸馏；`--init-from` 严格加载同结构 PPO 权重，
  也可从蒸馏 checkpoint 初始化学生 actor。已有 87 维输入、6D 输出的同结构教师权重可直接初始化。
  初始化使用新目录，不恢复旧优化器或课程；`--resume` 用于同阶段、同结构及课程状态匹配的恢复。

## 8. 可视化回放

```bash
cd /home/ubuntu20/CodeMujoco/hammering-task

python -m visualization.compare_hammering \
  --checkpoint weight/hammer_rl/runs/teacher/a_only_01/model_best.pt \
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
  --checkpoint weight/hammer_rl/runs/teacher/a_only_01/model_best.pt \
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --output-dir weight/hammer_rl/runs/teacher/a_only_01/comparison_01 \
  --anygrasp-extra-depth 0.03
```

分别运行 SDK 最高分＋零残差、critic 选择＋actor 残差两组，每组每场景执行一次。
该入口默认读取 checkpoint 配置中的 A bank，
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
  --data-file weight/hammer_rl/reset_bank_stable_v1_256.npz \
  --policy trained --selection learned \
  --checkpoint weight/hammer_rl/runs/teacher/a_only_01/calibration_model_best/deployment.pt \
  --split eval --episodes 0 --num-envs 16 --headless \
  --output weight/hammer_rl/selected_eval.npz
```

`--selection all` 是原有逐候选评估，也是默认值；`sdk` 和 `learned` 每场景执行一次。
`learned` 使用包含 actor 和 critic 的教师/学生 PPO 或部署权重；`all/sdk` 仍支持蒸馏 actor。
评估默认关闭在线物理随机化（含扭簧刚度），但仍按评估 bank 恢复离线锤柄形态，
因此名义物理参数评估仍包含 bank 中的形态变化；`--noise` 开启深度噪声，`--seed` 默认 42。

### 9.1 学生在线推理接口

```python
from hammer_rl.inference import load_models, select_grasp

actor, scorer = load_models("weight/hammer_rl/runs/student/a_only_01/calibration_model_best/deployment.pt")
result = select_grasp(actor, scorer, depth, scene, grasps)
```

`depth` 为米单位 `(1, 120, 160)` 深度；`scene` 包含未批量的 `finger_pos`、`finger_vel`、
`camera_pos`、`camera_quat`、`intrinsics`、`target_pos`、`target_quat`，字段与训练公开观测一致。
位姿使用世界坐标，四元数为 wxyz，内参顺序为 fx、fy、cx、cy。
手指状态使用与 reset 中按模型关节顺序取得的相同顺序。
`grasps` 沿用 `generate_grasps` 返回的世界坐标候选字典，内部复用
`candidate_to_mount(..., extra_depth=0.03)`，不读取仿真锤子真值。

返回张量包含 `candidate_index`（输入集合内编号）、`score`、所有候选 `scores`、
6 维抓取残差 `action`、最终 `mount_pos/mount_quat`、`pregrasp_pos`、`width` 和 `gripper_open_ctrl`。
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

相关实现：[训练入口](train.py)、[A 数据准备](prepare.py)、[评分校准](calibrate.py)、
[选择与微调](inference.py)、[评估入口](evaluate.py)、
[网络和算法配置](rl_cfg.py)、[环境配置](env_cfg.py)、[场景总体对比](compare_evaluations.py)、
[六场景展示](../visualization/compare_hammering.py)。

## 11. 正式生成前的验证

```bash
python -m hammer_rl.prepare \
  --output /tmp/hammer_a4.npz --num-scenes 4 \
  --legacy-per-placement 1 --random-scenes 0 --candidates-per-scene 4 --seed 42
python -m hammer_rl.evaluate \
  --data-file /tmp/hammer_a4.npz \
  --split train --policy zero --selection sdk --episodes 4 --num-envs 4 --headless \
  --output /tmp/hammer_a_eval4.npz
```

四个 A 样本因每类只有一个而全部属于 train。
零残差用于检查真实管线，不代表学习后的任务成功率。正式实验使用上面的完整 bank 与新目录。
