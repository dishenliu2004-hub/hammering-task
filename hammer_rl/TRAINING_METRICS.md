# 锤击强化学习训练日志指标说明

本文说明当前 `python -m hammer_rl.train` 产生的训练指标、网络配置和统计口径，覆盖教师 PPO、教师向学生蒸馏和学生 PPO 微调三个阶段。依据当前工作区代码以及本机安装的 mjlab 1.6.0、rsl-rl-lib 5.4.2 实现核对；续训的网络与算法参数以对应目录的 `run_config.json` 为准。文中的数值边界均指正常有限数值计算；没有硬上限不代表任意大数值都符合正常物理表现。

## 1. 阅读日志前的统计约定

### 1.1 当前网络、观测和训练阶段

| 阶段 | 策略网络及输入 | 价值网络与优化目标 |
| --- | --- | --- |
| `teacher` | `MLPModel`：87 维 `privileged` → 隐藏层 `(512, 256, 128)` → 6 维高斯均值。 | critic：87 维 `privileged` → 隐藏层 `(256, 128, 64)` → 1 维价值；使用 PPO。 |
| `distill` | 学生 `CNNModel` 使用 `depth` 和 50 维 `public_seed`，教师使用上述特权 MLP。 | 无 critic；固定教师，最小化学生与教师的 6 维确定性输出之间的 MSE。采集时执行学生采样动作。 |
| `student` | 与蒸馏学生相同的 `CNNModel`，从蒸馏 checkpoint 初始化 actor。 | critic 与教师 PPO 的结构相同，仍使用特权观测；重新通过 PPO 微调。 |

学生深度输入的单样本形状为 `(1, 120, 160)`，依次经过三层卷积：通道数 `(16, 32, 32)`、卷积核 `(5, 3, 3)`、步长 `(4, 2, 2)`，无 padding、无池化。特征形状依次为 `(16, 29, 39)`、`(32, 14, 19)`、`(32, 6, 9)`；展平为 1728 维，与 50 维公开观测拼接后进入隐藏层 `(256, 128, 64)`，输出 6 维高斯均值。

上述隐藏层和卷积层均使用 ELU。教师 actor、两个 PPO 阶段的 critic 和学生的一维公开观测均开启观测归一化；学生深度图直接进入 CNN，不通过该一维运行统计归一化器。当前没有 RNN 或跨尝试的隐藏状态。

`public_seed` 包含初始候选位姿、候选宽度和开爪控制量、预抓位姿、手指关节位置/速度、相机位姿/内参及目标端面位姿。`privileged` 在此基础上追加锤子世界位姿及相对候选位姿、锤头中心、支撑物位姿/尺寸、质量、惯量和指面摩擦。观测均在 reset 时生成并冻结，不包含本次尝试的结果快照。

三个阶段的策略都配置 `GaussianDistribution(init_std=0.3, std_type="scalar")`：每个动作维度各有一个可学习、与状态无关的标准差，共 6 个参数，并非 6 维共用一个标量。评估入口使用策略的确定性均值输出；训练采集使用采样输出。蒸馏的 MSE 只监督均值，不直接训练标准差。

教师训练默认使用名义物理参数和无噪声缓存深度；蒸馏、学生 PPO 默认开启锤子质量/惯量倍率 `[0.9, 1.1]`、指面摩擦倍率 `[0.8, 1.2]` 随机化以及深度噪声。因此不同阶段的训练均值不能直接当成同分布的策略对比。

### 1.2 日志统计口径

- **一个环境步就是一次完整尝试**：策略输出一次 6 维抓取残差，随后执行接近、闭合、抬升、搬运、挥击、卸载和撤离。一个回合包含一个环境步，因此 `Mean episode length = 1.00` 是当前设计的正常结果。
- 当前场景的物理步长为 `0.002 s`，每个环境步包含 `8980` 个物理子步，回合配置时长 `T = 8980 × 0.002 = 17.96 s`。这是**仿真时间**，与终端的采集耗时不同。
- `Episode_Metrics/*`：除 `collect` 外，先取各回合最终缓存的指标，再对本次结束的环境求平均；因此二值指标打印为成功比例，力峰值打印为各回合峰值的平均，而不是全部回合中的最大峰值。未满足门控条件的零值也参与平均。
- `Episode_Reward/*`：训练入口的 `_RewardLogVecEnvWrapper.step()` 将它们覆盖为 `权重 × 同名 Episode_Metrics`，**不除以回合时长，也不乘时间步长**。它们是十一项原始评分的加权贡献；详见第 4 节。
- `Mean reward`、`Mean episode length`：本机 RSL-RL 的 `rewbuffer` 和 `lenbuffer` 使用最近最多 **128 个已结束回合**的滑动窗口。`Episode_*` 则来自本轮采集的回合统计。默认单进程、128 环境、每轮采集 1 步时，两者覆盖同一批 128 个回合；改变环境数或采集步数后，样本集合可能不同。
- 下文的 `[a, b]` 包含端点，`[a, b)` 不包含右端点，`∞` 表示没有代码硬上限。表中给出的是单回合取值的边界；相应均值仍落在这些边界内。

## 2. 训练进度、速度与 PPO 优化指标

| 指标 | 含义与单位 | 数值范围及解读 |
| --- | --- | --- |
| `Total steps` | 当前 runner 日志累计记录的环境样本步数，包含所有并行环境。不是物理子步数，也不是更新次数。 | 非负整数，无硬上限。每轮增加 `num_envs × num_steps_per_env × GPU进程数`；单进程、128 环境、每环境采集 1 步时，3584 对应本次 runner 记录了 28 轮采集。续训重新创建 logger 时，此计数从 0 开始，不能直接当作跨进程生命周期的累计步数。 |
| `Steps per second` | 每秒处理的环境样本数，包含采集和学习耗时。 | 非负整数，无硬上限；实际打印 `int(本轮样本数 / (Collection time + Learning time))`。例如 `int(128 / (16.151 + 0.061)) = 7`。不是物理仿真频率。 |
| `Collection time` | 本轮采集所用墙钟时间，包含策略采样、环境仿真、重置及采集阶段的记录等。 | `[0, ∞)` 秒。与并行环境数、物理子步数和设备有关，没有固定合格阈值。 |
| `Learning time` | 本轮学习阶段所用墙钟时间；PPO 包括回报计算与网络更新，蒸馏为行为克隆更新。 | `[0, ∞)` 秒。用于评估训练开销，不直接衡量策略效果。 |
| `Mean value loss` | critic 预测价值与目标回报之间的平方误差，默认使用裁剪价值损失；对本轮各训练 epoch、mini-batch 求平均。 | `[0, ∞)`；没有通用合格阈值。量级受奖励尺度和数据分布影响，通常希望在可比较的训练条件下下降并稳定。 |
| `Mean surrogate loss` | PPO 裁剪策略目标的负值，用于优化 actor；对本轮各 mini-batch 求平均。 | 可正、可负、可为零，无固定上下界。负值是正常情况，不是失败奖励；不能只凭它的正负判断训练成功。 |
| `Mean entropy loss` | 虽然名称带 `loss`，实际打印的是动作分布的平均**微分熵 H**，单位 nat；不是取负后的熵正则项。 | 一般高斯微分熵可为任意实数。当前 6 维高斯、各维标准差默认限制为 `[10⁻⁶, 10⁶]`，所以代码边界约为 `[-74.3794, 91.4067]` nat。可为负；越大表示原始动作分布越分散，没有“越大越好”的固定标准。 |
| `Mean reward` | 最近最多 128 个已结束回合的实际总奖励均值，当前每回合只有一次奖励。 | 非终局正奖励上界为 44，完整成功另加 50，掉落项权重为 −20；详见第 5 节；不是百分比，也没有归一化到 `[0,1]`。在相同配置和评估分布下通常越大越好。 |
| `Mean episode length` | 最近最多 128 个已结束回合的平均环境步数。 | 当前有效回合恒为 `1`。单位是环境步，不是秒，也不是 8980 个物理子步。 |
| `Mean action std` | 策略高斯动作分布在各动作维度上的标准差均值，描述探索幅度。 | 当前默认 `[10⁻⁶, 10⁶]`，初值为 `0.3`。这是 `tanh` 前无量纲动作的标准差，不是 `0.30 m` 或 `0.30 rad`；策略输出随后经过 `tanh` 并缩放为抓取残差。 |

当前动作先经过 `tanh`，再在候选抓取的局部坐标系中施加残差：平移各轴尺度为 `0.02 m`，旋转向量各分量尺度为 `π/16 rad`（11.25°）。这是每个分量的尺度，标准差本身不等于执行后的米制或角度误差；熵和 PPO 动作概率也均对应 `tanh` 前的高斯分布。

`Mean value loss`、`Mean surrogate loss`、`Mean entropy loss` 只属于两个 PPO 阶段。当前 PPO 初始学习率 `3e-4`，Adam 优化器，每轮 `5` 个 epoch、每个 epoch `4` 个 mini-batch；默认 128 环境时每批 32 条尝试，共 20 次梯度更新。采用自适应学习率，目标 KL 为 `0.01`，学习率调整边界为 `[1e-5, 1e-2]`；最大梯度范数为 `1.0`，优势默认在整轮采集数据上归一化。折扣 `gamma=0.99`、GAE `lam=0.95`，但当前每步为真实 terminal，不会跨尝试 bootstrap，单步目标回报就是本次奖励。

PPO 的主要公式为：

```text
r = π_new(a|s) / π_old(a|s)
L_surrogate = mean(max(-A*r, -A*clip(r, 1-ε, 1+ε)))
V_clip = V_old + clip(V_new - V_old, -ε, ε)
L_value = mean(max((V_new - G)², (V_clip - G)²))
H = Σ_i log(σ_i * sqrt(2πe))
L_total = L_surrogate + c_value * L_value - c_entropy * H
```

其中 `A` 为优势，`G` 为目标回报，当前默认 `ε = 0.2`、`c_value = 1`、`c_entropy = 0.005`。日志中的熵在优化中以负号加入总损失。

### 蒸馏指标

蒸馏使用 `DistillationRunner`，记录 `Mean behavior loss`，对应 TensorBoard 的 `Loss/behavior`：

```text
L_behavior = mean((μ_student(depth, public_seed) - μ_teacher(privileged))²)
```

均值覆盖采集样本和 6 个动作分量，再对本轮各次学习求平均；范围为 `[0, ∞)`。该误差位于 `tanh` 前的动作空间，没有米、弧度或力的单位，越小表示学生均值越接近教师均值。当前使用 MSE、学习率 `3e-4`、每轮 `5` 个学习 epoch、`gradient_length=1`、最大梯度范数 `1.0`。此阶段没有 PPO 的价值、策略代理或熵损失；任务奖励仍记录，但不用于蒸馏梯度优化。

### 终端名称与 TensorBoard 标签

| 终端名称 | TensorBoard 标签 | 适用阶段 |
| --- | --- | --- |
| `Mean value loss` / `Mean surrogate loss` / `Mean entropy loss` | `Loss/value` / `Loss/surrogate` / `Loss/entropy` | 教师、学生 PPO |
| `Mean behavior loss` | `Loss/behavior` | 蒸馏 |
| 学习率（终端没有单独一行） | `Loss/learning_rate` | 全部阶段 |
| `Mean action std` | `Policy/mean_std` | 全部阶段；蒸馏对应学生分布 |
| `Mean reward` / `Mean episode length` | `Train/mean_reward` / `Train/mean_episode_length` | 全部阶段 |
| `Steps per second` / `Collection time` / `Learning time` | `Perf/total_fps` / `Perf/collection_time` / `Perf/learning_time` | 全部阶段 |
| `Episode_Metrics/*` / `Episode_Reward/*` / `Episode_Termination/*` | 与终端名称相同 | 全部阶段 |
| `Curriculum/placement/success_rate_flat` | 与终端名称相同 | 仅教师训练，平放近期完整成功率 |
| `Curriculum/placement/success_rate_lean_head_down` | 与终端名称相同 | 仅教师训练，锤头朝下斜靠近期完整成功率 |
| `Curriculum/placement/success_rate_lean_head_up` | 与终端名称相同 | 仅教师训练，锤头朝上斜靠近期完整成功率 |
| `Curriculum/placement/success_rate_inverted` | 与终端名称相同 | 仅教师训练，倒立近期完整成功率 |
| `Curriculum/placement/probability_flat` | 与终端名称相同 | 仅教师训练，下次 reset 的平放目标概率 |
| `Curriculum/placement/probability_lean_head_down` | 与终端名称相同 | 仅教师训练，下次 reset 的锤头朝下斜靠目标概率 |
| `Curriculum/placement/probability_lean_head_up` | 与终端名称相同 | 仅教师训练，下次 reset 的锤头朝上斜靠目标概率 |
| `Curriculum/placement/probability_inverted` | 与终端名称相同 | 仅教师训练，下次 reset 的倒立目标概率 |

上述曲线的默认横轴是 runner 更新编号 `it`，从 0 开始，续训会恢复更新编号；`Train/mean_reward/time` 和 `Train/mean_episode_length/time` 则使用当前 logger 累计的采集与学习墙钟秒数。`Total steps` 仅在终端打印，不是同名 TensorBoard 标量。

## 3. 任务原始指标：Episode_Metrics

关键时点：闭合结束约 `2.4 s`；抬升结束为快照 **A**，约 `3.2 s`；挥击前为快照 **B**，约 `16.2 s`。这些均为每次尝试内的仿真时间。

| 指标 | 含义、计算时点与单位 | 单回合范围 |
| --- | --- | --- |
| `capture` | 闭合结束时 `(I_left + I_right) / 2`，接触形成分，无量纲。 | `{0, 0.5, 1}` |
| `lift_progress` | A 时刻双侧接触下，`clip(lift_height / lift_threshold, 0, 1)`；不足 15 cm 也可得部分分数。 | `[0,1]` |
| `lift_success` | A 时刻锤体真实高度增量至少 0.15 m，且双侧接触；记为 L。 | `{0,1}` |
| `grasp_before_swing` | B 时刻 L 成立、未锁存掉落且仍双侧接触；记为 G，之后保持历史阶段事件含义。 | `{0,1}` |
| `wrap` | 闭合有符号包络加三阶段顶靠奖励后除以 1.5，公式见第 4 节。 | `[-38/165,1]`，约 `[-0.230303,1]` |
| `orientation` | A 时刻横向夹柄，平放时增加双向侧向接近偏好；乘 L，无量纲。 | `[0,1]` |
| `distance` | A 时刻抓取中心到锤头中心沿柄轴的投影，除以 0.315 m 后裁剪并乘 L。 | `[0,1]` |
| `stability` | A 之后至卸载结束、撤离之前的最大相对漂移评分；B 起按当前实时持锤条件 H 门控。 | `[0,1]` |
| `force` | 归一化法向冲量评分 `I_H × J / (J + impulse_scale)`，无量纲。 | `[0,1)` |
| `impact_impulse` | 首次有效命中后 40 ms 内、仍满足 H 时的压缩法向力积分 J，单位 N·s；掉落后保留历史值。 | `[0,∞)` |
| `valid_strike` | B 之后、撤离之前，是否曾有当时有效持锤的锤头—端盖正压缩法向接触；之后掉落也保留。 | `{0,1}` |
| `task_success` | `I_H × I[valid_strike > 0]`，在卸载结束、开始撤离前冻结；表示抓取—保持—命中—卸载结束仍持有。 | `{0,1}` |
| `drop` | L 成立后，双侧接触连续缺失 20 ms 锁存的掉落事件。 | `{0,1}` |
| `lift_height` | A 时刻锤体原点相对场景初态的真实 z 增量，单位 m。 | 有符号实数 |
| `grip_distance` | A 时刻两指尖中点到锤头中心的三维欧氏距离，单位 m；与轴向奖励 distance 区分。 | `[0,∞)` |
| `translation_drift` | A 之后至卸载结束、撤离之前，相对 A 基准的最大夹爪系锤体位置漂移，单位 m；失败回合也记录。 | `[0,∞)` |
| `rotation_drift` | 同一评价时段相对 A 的最大姿态最短旋转角误差，单位 rad。 | `[0,π]` |
| `peak_force` | B 之后至卸载结束、撤离之前，实时有效持锤命中的压缩法向力峰值，单位 N；后来掉落不追溯清零。 | `[0,∞)` |
| `touch_peak` | 原端面 touch 传感器峰值，单位 N；保留原 `~finished & grasp_valid` 诊断门控。 | `[0,∞)` |
| `return_touch_peak` | `returning=True` 时原 touch 峰值，单位 N；仅作撤离诊断。 | `[0,∞)` |
| `seat_closure` / `seat_lift` / `seat_before_swing` | 闭合/A/B 时刻末尾五个连续 250 Hz 采样帧是否有效顶靠；各自无抬升门控，加入 wrap 时分别由 L/L/G 门控。 | 各为 `{0,1}` |
| `collect` | 子步收集函数返回累计 peak_force 的子步平均，再对回合平均，单位 N；不是最终峰值或瞬时力均值。 | `[0,∞)`，同批次不超过平均 peak_force |

### 接触、掉落、顶靠与评分时窗

复用六个指面各两个锤柄接触槽：前六槽属于 narrow（left），后六槽属于 wide（right）。每侧至少一个槽满足 `found > 0` 且法向力 `> 0`，才算该侧接触；`bilateral = left & right`。

从 A 后第一个子步到既有卸载结束，只对尚未 `returning` 的环境更新失联计数。L 成立后，bilateral 连续失败 `round(0.02 / physics_dt)` 步（当前 10 步）锁存 `dropped`；恢复接触仅清零计数，不恢复掉落资格。B 时刻按原定义锁存 `grasp_valid = G`，之后不因挥击中掉落清零此控制快照。纯评分条件为 `H(t) = G & ~dropped(t) & bilateral(t)`。

短于 20 ms 的暂时失联也会使当前 stability、force、task_success 为零；未锁存掉落时恢复双侧接触可重新按累计漂移和冲量评分。最后一帧 bilateral 必须成立才能保留这些正分。掉落不回溯清零阶段事件、A 时刻几何分、历史 valid_strike、peak_force 或 impact_impulse。

**有效顶靠**沿用锤柄与基座前端支撑面的原判据：法向力绝对值至少 `0.1 N`，高度偏差不超过 `2 mm`，法向夹角不超过 `30°`。各关键时点末尾五个连续 250 Hz 采样帧须均成立，首尾约 16 ms。

**闭合包络 E**继续使用 `compute_enclosure_score()`。掌面接触以 250 Hz 采样，末尾至少五帧满足原强度判据 `F >= 0.1` 或 `F / 250 >= 0.0001`（当前等效至少 `0.025 N`）。原几何判定仍要求有效前向外切线交点、两侧距离在 `(0,0.15] m`，以及膨胀 1 mm 凸包内穿透长度不超过 3 mm。

两侧内弯角度从 5° 归一化到 `[0,1]`，满分角约为 narrow `66.1299°`、wide `65.6153°`；外弯侧固定为 0.1。设两侧角度分数为 l、r，`u=max(l,r)`，`h=2lr/(l+r)`（两侧为零时 h=0），内弯分为 `0.8h^0.7+0.2u^0.7`，任一外弯时为 `−(0.8h+0.2u)`。通过原真包络判定才得到 E，否则 E=0。E 的代数包络为 `[-19/55,1]`，归一化后 wrap 的包络为 `[-38/165,1]`，不保证几何上每个端点都可达。

**冻结时点**沿用既有卸载规则：原控制 `impact_started`、touch 连续零计数和锤头接触消失仍决定提前卸载结束。提前结束的本帧仍完成评分，下一帧进入 returning 后冻结；未提前结束则在 `k == unload_end` 留下最终评分。撤离仅继续记录 return_touch_peak。全部本帧评分在现有评估回调前写入。

本轮没有增加冲击后保持动作或统一保持时长；task_success 不表示钉入、做功或冲量阈值达标。

## 4. 加权奖励日志：Episode_Reward

### 4.1 单次尝试的奖励公式

`I[条件]` 表示条件成立取 1，否则取 0；`clip` 为截断。闭合、A、B 时点以及 L、G、H 定义见第 3 节。事件每次尝试只计一次，子步仅更新缓存，reward manager 在环境步末加权求和。

| 键名 | 原始评分 | 权重 |
| --- | --- | ---: |
| capture | `(I_left + I_right) / 2`，闭合结束写入 | 2 |
| lift_progress | `I_bilateral(A) × clip(Δz / lift_threshold, 0, 1)` | 3 |
| lift_success | `L = I[Δz ≥ lift_threshold] × I_bilateral(A)` | 10 |
| grasp_before_swing | `G = L × I[未掉落(B)] × I_bilateral(B)` | 5 |
| wrap | `[L(E + 0.10 seat_closure + 0.15 seat_lift) + 0.25 G seat_before_swing] / 1.5` | 3 |
| orientation | `L × (flat ? 0.4 Q_perp + 0.6 Q_side : Q_perp)` | 2 |
| distance | `L × clip((p_grip − p_head)·h_A / distance_ref, 0, 1)` | 1 |
| stability | `I_H × 252 / (252 + 48x − 43x² + 23x³)` | 8 |
| force | `I_H × J / (J + impulse_scale)` | 20 |
| task_success | `I_H × I[valid_strike > 0]`，撤离前终值 | 100 |
| drop | `I_dropped`，锁存 | −20 |

**朝向**：`h_A` 是 A 时刻锤体局部 +X 的世界系柄轴，`c_A` 是 narrow 指尖到 wide 指尖的单位向量。`Q_perp = clip(1 − (c_A·h_A)², 0, 1)`。抓取前缓存姿态对应的柄轴为 `h_0`，桌面法向为世界 +Z，`d_side = normalize(n_table × h_0)`；最终抓姿 final_quat 对应的 mount 局部 +Z 接近方向为 a，`Q_side = clip((a·d_side)², 0, 1)`。仅 `placement == 0` 的平放样本使用侧向分，不要求夹指朝某个固定正方向，也不改变接近轨迹。

**距离**：`p_grip` 为 A 时刻两指尖中点，`p_head = p_hammer − 0.24 h_A`。奖励使用有符号柄轴投影，横向偏移不增加 distance；grip_distance 仍为三维欧氏距离诊断。

**稳定性**：相对 A 基准逐子步累计 `ep_max = max ||p_rel(t) − p_rel(A)||`、`er_max = max angle(q_rel(A), q_rel(t))`。归一化误差为：

```text
x = sqrt(0.5 × ((ep_max / stability_position_scale)²
             + (er_max / stability_rotation_scale)²))
```

实际读取配置 `stability_position_scale = 0.005 m` 和 `stability_rotation_scale = π/36 rad`（5°）。B 前只累计漂移，B 起根据实时 H 更新评分；平移和旋转同时达到尺度的 1/2/3 倍时，在 H 成立下分别得 0.9/0.7/0.4。

**有效命中与冲量**：B 之后、卸载结束且尚未撤离时，`F_n = max(−F_head→cap · n_target, 0)`，沿用锤头 geom—端盖 geom 匹配。`V = H & touching & (F_n > 0)`；valid_strike 锁存是否曾出现 V，peak_force 仅累计 V 时的峰值，保证 `valid_strike == 0 ⇒ peak_force == 0`。

首次 V 的物理子步记为 first_hit_step。设 `N_window = round(impact_window_s / physics_dt)`，在 `first_hit_step > 0` 且 `0 ≤ k − first_hit_step < N_window`、H 成立、尚未撤离时，`J ← J + F_n × physics_dt`。首次命中当前子步计入，窗口内接触反复可以继续积分，窗口外不再增加；不对整段卸载或撤离积分。

`impact_window_s = 0.04 s`、`impulse_scale = 1.0 N·s` 是待实验标定的首轮超参数，不代表实际脉冲长度或任务达标阈值。J=1 N·s 且 H 成立时 force=0.5；无有效命中时 J 和 force 均为零。掉落后原始 J 保留，最终 force、stability、task_success 为零。force 无量纲，peak_force 单位 N，impact_impulse 单位 N·s。

### 4.2 加权奖励的日志统计

当前训练入口 `_RewardLogVecEnvWrapper` 动态遍历 cfg.rewards，覆盖所有十一项奖励日志：

```text
Episode_Reward/name = weight_name × Episode_Metrics/name
                    = mean(weight_name × s_name)
```

每个奖励键都有同名 metric。impact_impulse 仅作为原始 metric，不重复加入奖励。例如 `mean(force)=0.2` 时 `Episode_Reward/force=2`；`mean(drop)=0.25` 时 `Episode_Reward/drop=−5`。

普通 `ManagerBasedRlEnv` / `RslRlVecEnvWrapper` 的底层奖励日志仍可能按配置时长 T 归一化；训练入口覆盖后的日志不除以 T。evaluate.py 使用普通 wrapper，但终端和 NPZ 保存的是实际 reward 和各项原始 values。比较历史实验时须确认统计口径；新旧策略应由同一修改后的评估器重新生成结果，不为旧 NPZ 补字段。

## 5. 总奖励、课程与终止统计

两个 PPO 阶段使用、蒸馏阶段仅记录的单回合任务奖励为：

```text
R = 2 capture + 3 lift_progress + 10 lift_success + 5 grasp_before_swing
  + 3 wrap + 2 orientation + distance + 8 stability + 10 force
  + 50 task_success − 20 drop
```

非终局正奖励上界为 44，完整成功另加 50；有限冲量下 force < 1。总奖励的宽松代数包络为 `[-20 − 38/55, 94)`，各门控相关，不保证每个值可达。抬升失败时仍可获得 capture 和双侧接触下的部分 lift_progress。wrap 保留小幅负值，其余正向质量分在 `[0,1]`；本轮权重体现任务优先级，是尚未验证最优性的工程起点。

训练日志直接满足 `sum(Episode_Reward/*) =` 同批次平均总奖励，无需乘回合时长 17.96 s。默认单进程、128 环境、每轮采集一步时 Mean reward 与该和应在打印精度内一致；环境数改变后，128 回合滑动窗口与本轮指标样本集合可能不同。

| 指标 | 含义 | 数值范围及解读 |
| --- | --- | --- |
| `Curriculum/placement/success_rate_*`（上表四类） | 每类最近最多 100 个已完成尝试的原始二值 `task_success` 均值；仅教师训练。 | `[0,1]`；完整成功表示抓取、保持、有效命中且卸载结束前仍持有。空历史的 `0.5` 是启动先验，不是实测成功率；初始化和恢复后的零长度回合不计入。 |
| `Curriculum/placement/probability_*`（上表四类） | 由各类近期完整成功率计算的下次 reset 目标概率；仅教师训练。 | 各类大于零且总和为 1，首次各为 `0.25`。M 个重置环境的预期类别数量为 `M*q_k`，实际数量可波动，不是已分配的整数环境数。 |
| `Episode_Termination/attempt_complete` | 本次重置的环境中，满足 `episode_length_buf >= 1` 的数量，表示一次尝试正常结束。 | 单次统计为整数 `[0, num_envs]`；日志还会对本轮各次统计求均值，因此格式可带小数。当前 128 个并行环境每步都结束一个回合，通常为 128；它是**结束数量**，不是成功数量，也不是百分比。 |

教师从第一次 reset 起同时开放四类训练摆放。课程在 reset 事件和动作评分清零前读取旧场景及最终结果，
按传入环境顺序追加同类结果后截取最近 100 条，不按迭代均值统计。设成功率为 `p_k`，
`w_k=(p_k+epsilon)^(kappa*t)*(1-p_k+epsilon)^(kappa*(1-t))`，
`q=softmax(log(w+epsilon)/T)`；默认 `t=0.5`、`kappa=1.0`、`T=2.0`、`epsilon=1e-4`，
实际参数从 run 配置的 `placement_sgs` 恢复。接近 50% 成功率的类别概率更高，四类成功率
相同（包括全部失败）时均匀采样。每个训练场景权重为 `q_k/该类训练场景数`，评估场景权重为零。
四类历史随周期和最终 checkpoint 保存，续训恢复历史后重算概率；课程不再按环境步里程碑切换。

`attempt_complete` 配置为正常 terminal，`time_out=False`，环境还设置 `is_finite_horizon=True`；抬升失败、掉落、未命中同样执行到完整环境步结束，不产生独立的失败终止日志。PPO 不将下一次 reset 后的初始状态作为本次尝试的 bootstrap 价值。

## 6. 如何解读训练和评估结果

- 对同一批次、相同样本，应满足 `task_success ≤ valid_strike ≤ grasp_before_swing ≤ lift_success`。抬升成功率高而挥击前有效抓持低，说明搬运过程中发生掉落锁存或 B 时刻失去双侧接触；不能仅凭这两个均值确定具体原因。
- `peak_force` 是包含失败和未命中零值的平均有效峰值，同时受命中率和命中力度影响。若同批次 `mean(valid_strike) > 0`，可用 `mean(peak_force) / mean(valid_strike)` 查看命中回合的平均峰值；更准确的结果应从逐回合 NPZ 按 `valid_strike == 1` 筛选计算。
- `translation_drift` 和 `rotation_drift` 包含失败回合。平移单位是 m、旋转单位是 rad，较大的漂移均值需要结合逐回合分布和回放判断，不能直接解释为有效抓持样本的最大漂移。
- `wrap`、`stability`、`force` 是连续评分；例如 `stability=0.2` 不表示稳定成功率为 20%。只有单回合二值指标的均值才可直接读成比例。`seat_*` 的均值也只是各时点的顶靠比例，不等于抓持成功率。
- 教师课程切换会改变采样分布，阶段边界附近奖励或成功率回落不一定表示策略退化。蒸馏时应同时看 `Loss/behavior` 和实际任务指标；模仿误差下降不保证命中率同步上升。
- 正式比较策略时，使用 `evaluate.py` 在相同缓存、划分、候选、随机种子和锤头额外质量下评估。该入口默认关闭物理随机化和深度噪声，`--noise` 可启用深度噪声。评估执行确定性均值，训练执行高斯采样，二者的均值无需完全相等。
- `evaluate.py` 的 `--episodes 0` 遍历选定划分和摆放类型下的全部候选；`compare_evaluations.py` 再筛选每个场景 SDK 分数最高的一个候选，分别按四类放置汇总。全候选均值、每场景最高分候选均值和带课程的训练均值属于不同统计集合。

范围和换算关系假设原始指标为有限值。底层奖励管理器会将非有限的加权奖励项置零，而训练 wrapper 直接读取指标日志；若原始指标出现 NaN/Inf，则加权日志与实际 reward 可能不再满足上述等式，应结合原始状态排查。

## 7. 实现依据

- [环境配置与奖励权重](env_cfg.py)：奖励、指标、单回合时长、传感器和课程注册。
- [动作与逐子步指标计算](mdp/actions.py)：关键时点、阶段门控、漂移、包络、稳定性、法向冲量与有效力峰值。
- [包络和顶靠几何评分](../generation/enclosure.py)：真包络与顶靠判据、角度评分。
- [奖励读取](mdp/rewards.py)、[终止条件](mdp/terminations.py)、[课程逻辑](mdp/curriculum.py)。
- [观测读取](mdp/observations.py)、[重置与观测构造](mdp/resets.py)：87/50 维观测、冻结深度和随机化。
- [网络与算法配置](rl_cfg.py)、[训练入口](train.py)：教师/学生网络、PPO/蒸馏参数、每次采集步数、课程命令行默认值，以及 `_RewardLogVecEnvWrapper` 的加权奖励日志覆盖逻辑。
- [评估入口](evaluate.py)、[四类放置对比](compare_evaluations.py)：确定性评估、逐回合原始指标与 SDK 最高分候选统计。
- 本机框架源码位于 `/home/ubuntu20/anaconda3/envs/hammer/lib/python3.11/site-packages/` 下：`mjlab/managers/{metrics,reward,termination}_manager.py` 定义底层日志聚合；`mjlab/rl/config.py` 定义 PPO 默认值；`rsl_rl/utils/logger.py` 定义吞吐、128 回合滑动窗口与打印；`rsl_rl/algorithms/{ppo,distillation}.py`、`rsl_rl/models/{mlp_model,cnn_model}.py`、`rsl_rl/modules/{distribution,cnn}.py` 定义损失、网络、熵与标准差边界。
