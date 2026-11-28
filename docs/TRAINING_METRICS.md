# 锤击强化学习训练日志指标说明

本文说明 `python -m hammer_rl.train` 的训练指标、网络配置和统计口径，覆盖教师 PPO、教师向学生蒸馏和学生 PPO 微调。第 4 节给出默认的十三项人工奖励与 V2.0 奖励配方，第 6 节说明 VLM 闭环评估的逐回合反馈。网络与算法参数以对应目录的 `run_config.json` 为准。本文按项目代码及本机 mjlab 1.6.0、rsl-rl-lib 5.4.2 核对，数值边界基于有限值计算。

## 1. 阅读日志前的统计约定

### 1.1 当前网络、观测和训练阶段

| 阶段 | 策略网络及输入 | 价值网络与优化目标 |
| --- | --- | --- |
| `teacher` | `MLPModel` 使用 87 维 `privileged`，经隐藏层 `(512, 256, 128)` 输出 6 维高斯均值。 | critic 使用 87 维 `privileged`，经隐藏层 `(256, 128, 64)` 输出 1 维价值。使用 PPO。 |
| `distill` | 学生 `CNNModel` 使用 `depth` 和 50 维 `public_seed`，教师使用上述特权 MLP。 | 固定教师，最小化学生与教师的 6 维确定性输出之间的 MSE。采集时执行学生采样动作。 |
| `student` | 与蒸馏学生相同的 `CNNModel`，从蒸馏 checkpoint 初始化 actor。 | 独立 `CNNModel` 使用相同深度和公开观测，输出 1 维价值、无动作分布；重新通过 PPO 微调。 |

学生深度输入的单样本形状为 `(1, 120, 160)`，依次经过三层卷积：通道数 `(16, 32, 32)`、卷积核 `(5, 3, 3)`、步长 `(4, 2, 2)`，无 padding、无池化。特征形状依次为 `(16, 29, 39)`、`(32, 14, 19)`、`(32, 6, 9)`。展平为 1728 维，与 50 维公开观测拼接后进入隐藏层 `(256, 128, 64)`，输出 6 维高斯均值。

上述隐藏层和卷积层均使用 ELU。教师 actor、两个 PPO 阶段的 critic 和学生的一维公开观测均开启观测归一化；学生深度图直接进入 CNN，不通过该一维运行统计归一化器。当前没有 RNN 或跨尝试的隐藏状态。

训练时，reset 先抽取场景，再从该场景的 AnyGrasp 缓存中均匀抽取一个候选。Actor 对绑定的候选输出局部 6D 残差。VLM 闭环评估使用 `--selection learned`，由当前 critic 为同场景的候选评分，再由 actor 修正得分最高的候选。`calibrate.py` 可在固定 actor 后，另行用训练集候选实际执行的总奖励校准 critic。

`public_seed` 包含候选夹爪基座位姿、开度和开爪控制量、预抓位姿、手指关节位置与速度、相机位姿与内参，以及目标端面位姿。`privileged` 在此基础上追加锤子世界位姿及其相对候选位姿、锤头中心、支撑物位姿和尺寸、质量、惯量与指面摩擦。观测在决策前构造并冻结。Actor 在尝试开始时输出一次 6 维残差，与候选合成执行抓姿。A 是抬升结束后的快照，后续三次敲击及两次恢复均由固定轨迹执行。

三个阶段的策略都配置 `GaussianDistribution(init_std=0.3, std_type="scalar")`：每个动作维度各有一个可学习、与状态无关的标准差，共 6 个参数，并非 6 维共用一个标量。评估入口使用策略的确定性均值输出；训练采集使用采样输出。蒸馏的 MSE 只监督均值，不直接训练标准差。

当前 `make_env_cfg()` 对教师、蒸馏和学生三个阶段均开启锤子质量/惯量倍率 `[0.9, 1.1]`、指面摩擦倍率 `[0.8, 1.2]` 和扭簧刚度倍率 `[0.8, 1.2]` 的均匀随机化；蒸馏及学生另启用深度噪声。
每个环境在每次 reset 时为窄指和宽指扭簧采样一个共享倍率，基于名义刚度 `0.2 N·m/rad` 得到 `0.16～0.24 N·m/rad` 的刚度，改变扭簧恢复力矩。

`peak_force` 记录三次有效命中评分时窗内的最大瞬时压缩法向力，单位 N；`mean_force` 是三次独立 40 ms 冲量窗口的平均力均值，单位 N；`force` 是由三次窗口平均力计算的无量纲对数评分。三者不能互换。

### 1.2 日志统计口径

- **一个环境步就是一次抓取与敲击尝试**。策略输出一次 6 维抓姿残差，正常轨迹依次执行接近、闭合、抬升、搬运、挥击 1、恢复 1、挥击 2、恢复 2、挥击 3、最终卸载和撤离。一个回合包含一个环境步，`Mean episode length = 1.00`。
- 当前场景的物理步长为 `0.002 s`，每次挥击为 100 个子步（`0.20 s`），两次恢复各 `0.50 s`，最终卸载时窗 `0.50 s`，预留撤离时间按代码取整为 `0.44 s`。完整轨迹共 `3570` 个物理子步，最长仿真时长 `T = 7.14 s`。训练在锁存掉锤的子步结算并重置该世界，每个世界的实际子步数可不同。这是**仿真时间**，与终端的采集耗时不同。
- `Episode_Metrics/*`：除 `collect` 外，先取各回合最终缓存的指标，再对本次结束的环境求平均；因此二值指标打印为成功比例，力峰值打印为各回合峰值的平均，而不是全部回合中的最大峰值。未满足门控条件的零值也参与平均。
- `Episode_Reward/*` 按已结束的尝试计算 `权重 × 同名 Episode_Metrics`。默认奖励记录十三项加权贡献，配方模式记录 `Episode_Reward/recipe`。两种模式均直接记录回合奖励，冲量积分中的 `physics_dt` 只用于计算物理量。详见第 4 节。
- `Mean reward`、`Mean episode length`：本机 RSL-RL 的 `rewbuffer` 和 `lenbuffer` 使用最近最多 **128 个已结束回合**的滑动窗口。`Episode_*` 则来自本轮采集的回合统计。每轮采集 `num_envs × ROLLOUT_STEPS` 次尝试。
- 下文的 `[a, b]` 包含端点，`[a, b)` 不包含右端点，`∞` 表示没有代码硬上限。表中给出的是单回合取值的边界；相应均值仍落在这些边界内。

## 2. 训练进度、速度与 PPO 优化指标

| 指标 | 含义与单位 | 数值范围及解读 |
| --- | --- | --- |
| `Total steps` | 当前 runner 日志累计记录的环境样本步数，包含所有并行环境。不是物理子步数，也不是更新次数。 | 非负整数，无硬上限。每轮增加 `num_envs × num_steps_per_env × GPU进程数`。单进程、128 个世界、`ROLLOUT_STEPS=1` 时，每轮增加 128。续训重新创建 logger 时，此计数从 0 开始，不能直接当作跨进程生命周期的累计步数。 |
| `Steps per second` | 每秒处理的环境样本数，包含采集和学习耗时。 | 非负整数，无硬上限；实际打印 `int(本轮样本数 / (Collection time + Learning time))`。例如 `int(128 / (16.151 + 0.061)) = 7`。不是物理仿真频率。 |
| `Collection time` | 本轮采集所用墙钟时间，包含策略采样、环境仿真、重置及采集阶段的记录等。 | `[0, ∞)` 秒。与并行环境数、物理子步数和设备有关，没有固定合格阈值。 |
| `Learning time` | 本轮学习阶段所用墙钟时间；PPO 包括回报计算与网络更新，蒸馏为行为克隆更新。 | `[0, ∞)` 秒。用于评估训练开销，不直接衡量策略效果。 |
| `Mean value loss` | critic 预测价值与目标回报之间的平方误差，默认使用裁剪价值损失；对本轮各训练 epoch、mini-batch 求平均。 | `[0, ∞)`；没有通用合格阈值。量级受奖励尺度和数据分布影响，通常希望在可比较的训练条件下下降并稳定。 |
| `Mean surrogate loss` | PPO 裁剪策略目标的负值，用于优化 actor；对本轮各 mini-batch 求平均。 | 可正、可负、可为零，无固定上下界。负值是正常情况，不是失败奖励；不能只凭它的正负判断训练成功。 |
| `Mean entropy loss` | 虽然名称带 `loss`，实际打印的是动作分布的平均**微分熵 H**，单位 nat；不是取负后的熵正则项。 | 一般高斯微分熵可为任意实数。当前 6 维高斯、各维标准差默认限制为 `[10⁻⁶, 10⁶]`，所以代码边界约为 `[-74.3794, 91.4067]` nat。可为负；越大表示原始动作分布越分散，没有“越大越好”的固定标准。 |
| `Mean reward` | 最近最多 128 个已结束回合的实际总奖励均值，当前每回合只有一次奖励。 | 量级取决于本阶段的奖励配置。第 5 节给出默认人工奖励的范围。 |
| `Train/total_reward` | 本轮采集的实际总奖励均值。 | 默认等于同批次十三项加权奖励之和。配方模式等于 `Episode_Reward/recipe`。 |
| `Mean episode length` | 最近最多 128 个已结束回合的平均环境步数。 | 当前有效回合恒为 `1`。一次尝试最多执行 3570 个物理子步。 |
| `Mean action std` | 策略高斯动作分布在各动作维度上的标准差均值，描述探索幅度。 | 当前默认 `[10⁻⁶, 10⁶]`，初值为 `0.3`，对应 `tanh` 前的无量纲动作。 |

当前动作先逐维经过 `tanh`。前三维乘 `0.10 m`，在候选夹爪的局部坐标系内平移基座。后三维乘 `π/5 rad`，由 `quat_box_plus` 转成局部旋转并右乘到候选四元数。预抓位置沿修正后的接近轴后退 `0.10 m`。熵和 PPO 动作概率均以 `tanh` 前的高斯分布计算。

`Mean value loss`、`Mean surrogate loss`、`Mean entropy loss` 只属于两个 PPO 阶段。当前 PPO 初始学习率 `3e-4`，使用 Adam 优化器。每轮执行 `5` 个 epoch，每个 epoch 划分 `4` 个 mini-batch，共 20 次梯度更新。`hammer_rl.train` 默认 128 个环境，`ROLLOUT_STEPS=1` 时每个 mini-batch 有 32 条尝试。采用自适应学习率，目标 KL 为 `0.01`，学习率调整边界为 `[1e-5, 1e-2]`。最大梯度范数为 `1.0`，优势默认在整轮采集数据上归一化。折扣 `gamma=0.99`、GAE `lam=0.95`，每步为真实终局，单步目标回报等于本次奖励。

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

当前终端仅保留迭代编号、`Episode_Reward/*`、`Train/total_reward`、损失、`Total steps` 和 `Iteration time`；下表其余指标仍写入 TensorBoard。

| 终端名称 | TensorBoard 标签 | 适用阶段 |
| --- | --- | --- |
| `Mean value loss` / `Mean surrogate loss` / `Mean entropy loss` | `Loss/value` / `Loss/surrogate` / `Loss/entropy` | 教师、学生 PPO |
| `Mean behavior loss` | `Loss/behavior` | 蒸馏 |
| 学习率（终端没有单独一行） | `Loss/learning_rate` | 全部阶段 |
| `Mean action std` | `Policy/mean_std` | 全部阶段；蒸馏对应学生分布 |
| `Mean reward` / `Mean episode length` | `Train/mean_reward` / `Train/mean_episode_length` | 全部阶段 |
| `Train/total_reward` | 同名 | 全部阶段 |
| `Steps per second` / `Collection time` / `Learning time` | `Perf/total_fps` / `Perf/collection_time` / `Perf/learning_time` | 全部阶段 |
| `Episode_Metrics/*` / `Episode_Reward/*` / `Episode_Termination/*` | 与终端名称相同 | 全部阶段 |
| `Curriculum/scene_sgs/sgs_mean_success` | 同名 | 仅教师训练，已有历史的训练场景近期完整成功率的均值 |
| `Curriculum/scene_sgs/sgs_mean_probability` | 同名 | 仅教师训练，训练场景采样概率的均值 |
| `Curriculum/scene_sgs/sgs_min_probability` / `Curriculum/scene_sgs/sgs_max_probability` | 同名 | 仅教师训练，训练场景采样概率的最小值与最大值 |
| `Curriculum/scene_sgs/sgs_observed_scenes` | 同名 | 仅教师训练，已有完成回合历史的训练场景数 |

上述曲线的默认横轴是 runner 更新编号 `it`，从 0 开始，续训会恢复更新编号；`Train/mean_reward/time` 和 `Train/mean_episode_length/time` 则使用当前 logger 累计的采集与学习墙钟秒数。`Total steps` 仅在终端打印，不是同名 TensorBoard 标量。

## 3. 任务原始指标：Episode_Metrics

闭合结束于 `0.70 s`，抬升结束于 `1.10 s` 并记录快照 **A**，挥击前快照记录于 `4.60 s`。这些均为每次尝试内的仿真时间。初次抬升指令为 `0.15 m`，随后夹爪基座在 `3.00 s` 内沿直线搬运至挥击起点并同步调姿，再稳定等待 `0.50 s`；有效抬升门槛仍为锤体真实高度增量至少 `0.15 m` 且双侧有效接触。

默认三次挥击的子步区间分别对应 `(4.60, 4.80] s`、`(5.30, 5.50] s`、`(6.00, 6.20] s`，两次恢复结束时点为 `5.30 s` 和 `6.00 s`。第三次挥击后才进入最终卸载，固定卸载窗口截止于 `6.70 s`，最迟在其后进入撤离。各时点由物理子步数确定。

下文漂移指标的参考基准：人工奖励模式使用最终修正抓姿与静止锤子的初始相对位姿；VLM 配方模式沿用闭合结束的实际相对位姿。

| 指标 | 含义、计算时点与单位 | 单回合范围 | 注释 |
| --- | --- | --- | --- |
| `capture` | 闭合结束时 `(I_left + I_right) / 2`，接触形成分，无量纲。 | `{0, 0.5, 1}` | 闭合结束时，夹爪对应一侧是否存在至少一个有效锤柄接触（0，0.5，1） |
| `lift_progress` | A 时刻双侧接触下，`clip(lift_height / lift_threshold, 0, 1)`；不足 15 cm 也可得部分分数。 | `[0,1]` | 抬升进度奖励，补充了不同高度提供的奖励，信息更密集 |
| `lift_success` | A 时刻锤体真实高度增量至少 0.15 m，且双侧接触；记为 L。 | `{0,1}` | 完整抬升成功奖励 |
| `grasp_before_swing` | 挥击前时刻 L 成立、未锁存掉落且仍双侧接触；记为 G，之后保持历史阶段事件含义。 | `{0,1}` | 验证挥击之前是否还能保持抓取 |
| `hold_fraction` | 从 A 之后到第一次挥击前快照，双侧接触且未锁存掉落的子步数除以这段窗口的总子步数（1750，即 3.50 s）。 | `[0,1]` | 提前终止的训练回合仍按原定窗口计分母 |
| `wrap` | 闭合有符号包络放大 2 倍，加三阶段顶靠奖励后除以 1.5，公式见第 4 节。 | `[-76/165,5/3]`，约 `[-0.460606,1.666667]` | 判断是否形成有效包络以及计算两侧指节的包覆质量 |
| `orientation` | A 时刻横向夹柄，平放时增加双向侧向接近偏好；乘 L，无量纲。 | `[0,1]` | 横向抓取与侧向抓取奖励 |
| `distance` | A 时刻抓取中心到锤头中心沿柄轴的投影，除以 0.315 m 后裁剪并乘 L。 | `[0,1]` | 抓取位姿能形成的有效力臂 |
| `stability` | 第一次挥击前计算并锁存 `G × S(x)`，x 由参考基准与挥击前实际相对位姿的差计算。 | `[0,1]` | 后续三次挥击、两次恢复、卸载及掉落均不再改变该分数；不是三次最低分或最终持锤分 |
| `force` | `0.5 mean_i(ln(1 + F_i / 10)) + 0.5 min_i(ln(1 + F_i / 10))`，`F_i` 为各次窗口平均力，单位 N。 | `[0,∞)` | 无量纲对数评分；最弱一次占一半评分，未命中一次对应 `F_i = 0`；不直接按峰值评分 |
| `hit_progress` | 三次有效命中数除以 3，无量纲。 | `{0, 1/3, 2/3, 1}` | 每次挥击最多记一次，历史命中不因后续掉落撤销 |
| `recovery_hold` | 两次恢复结束时的 ready 判定均值，无量纲；判据见第 4 节。 | `{0, 0.5, 1}` | 检查有效持锤且夹爪基座已回到挥击准备位姿；不要求此前已命中 |
| `task_success` | `I[三次均有效命中] × I_H`，在最终卸载结束、开始撤离前冻结。 | `{0,1}` | 只有三次均命中且最终仍有效持锤才算完整成功 |
| `drop` | 人工模式：A 时刻 L 不成立，或 L 成立后双侧接触连续缺失 20 ms，记为 1 并锁存。 | `{0,1}` | 配方模式仅统计成功抬升后的掉落事件 |
| `lift_height` | A 时刻锤体原点相对场景初态的真实 z 增量，单位 m。 | 有符号实数 | 与初态比较，不是夹爪的指令抬升量 |
| `translation_drift` | 参考基准与挥击前实际夹爪系锤体位置差的欧氏距离，挥击前时刻写入后保持不变，单位 m；失败回合也记录。 | `[0,∞)` | 用于第一次挥击前的 stability |
| `rotation_drift` | 参考基准与挥击前实际夹爪系锤体姿态的最短旋转角误差，挥击前时刻写入后保持不变，单位 rad。 | `[0,π]` | 用于第一次挥击前的 stability |
| `peak_force` | 三次有效命中评分时窗内的最大瞬时压缩法向力，单位 N；后来掉落不追溯清零。 | `[0,∞)` | 仅在当帧有效命中时更新；不能直接换算 force 奖励 |
| `valid_strike` | 三次中是否至少一次有效命中，锁存后不清零。 | `{0,1}` | 不是三次完整成功标志 |
| `hit_count` | 三次中已有效命中的次数。 | `{0,1,2,3}` | `hit_count = 3 × hit_progress` |
| `impact_impulse` | 三次独立窗口的法向冲量之和 `J_1 + J_2 + J_3`，单位 N·s。 | `[0,∞)` | 默认奖励使用综合 `force`，V2.0 配方也可选此指标 |
| `mean_force` | `(F_1 + F_2 + F_3) / 3`，单位 N。 | `[0,∞)` | 三次固定窗口平均力的算术均值，未命中零值参与；不是峰值或实际接触时长内的均值 |
| `strike_force_1` / `strike_force_2` / `strike_force_3` | 各次首次有效命中后的冲量除以固定窗口时长：`F_i = J_i / (N_window × physics_dt)`，单位 N。 | 各为 `[0,∞)` | 默认分母均为 40 ms；未命中为零，提前撤离也不缩短分母 |
| `seat_closure` / `seat_lift` / `seat_before_swing` | 闭合/抬升/挥击前时刻末尾五个连续 250 Hz 采样帧是否有效顶靠；各自无抬升门控，加入 wrap 时分别由 L/L/G 门控。 | 各为 `{0,1}` | 作为 wrap 的组成项，不单独加权 |
| `collect` | 子步收集函数返回当前累计 `mean_force`，框架先对整次尝试的物理子步求平均，再对回合平均，单位 N。 | `[0,∞)` | 包含挥击前的零值；不等于最终 mean_force、peak_force 或实际接触期间的平均力 |

加载奖励配方时，A 时刻还会记录以下抓取特征。它们同时写入训练 metrics 与评估 NPZ。

| 指标 | 含义 | 单回合范围 |
| --- | --- | --- |
| `grasp_u_rl` | 实际两指尖 Site 中点在锤子局部坐标系中的 X 坐标除以锤柄半长，在 A 时刻记录。 | 有符号实数，代码未裁剪 |
| `grasp_axis_cos2` | A 时刻实际两指尖对向轴与配方 `closing_axis_O` 点积的平方，裁剪到 `[0,1]`。 | `[0,1]` |
| `prior_region` | 以配方 `u_mean_RL` 为中心、`u_sigma_RL` 为宽度计算 `grasp_u_rl` 的 Gaussian 匹配分，乘 L。 | `[0,1]` |
| `prior_axis` | 实际对向轴与配方 `closing_axis_O` 的点积平方，乘 L。 | 适配轴为单位向量时为 `[0,1]` |

同一回合中，`grasp_u_rl` 与 `grasp_axis_cos2` 的原始特征在 A 时刻直接记录，`prior_region` 与 `prior_axis` 则带有抬升门控。解读抓取分布时可用 `lift_success` 筛选成功抬升的尝试。

### 接触、掉落、顶靠与评分时窗

复用六个指面各两个锤柄接触槽：前六槽属于 narrow（left），后六槽属于 wide（right）。每侧至少一个槽满足 `found > 0` 且法向力 `> 0`，才算该侧接触；`bilateral = left & right`。

从 A 后第一个子步到第三次挥击后的最终卸载结束，只对尚未 `returning` 的环境更新失联计数，覆盖三次挥击和两次恢复。L 成立后，bilateral 连续失败 `round(0.02 / physics_dt)` 步（当前 10 步）锁存 `dropped`；恢复接触仅清零计数，不撤销掉落锁存。第一次挥击前时刻锁存 `grasp_valid = G`，之后不因挥击中掉落清零此控制快照。实时评分条件为 `H(t) = G & ~dropped(t) & bilateral(t)`。

人工模式在 A 时刻 L 不成立时立即令 `drop = 1`，不等待失联计数；后续采样保留该惩罚。`dropped` 仍仅表示成功抬升后的实际掉落状态，配方模式的 `drop` 仍只记录该状态。

第一次挥击前当帧若没有双侧接触，G 和 stability 为零；该时点之后 stability 不再随 H 更新。后续短于 20 ms 的暂时失联会使当帧 H 为假，从而暂停有效命中和冲量积分；未锁存掉落时，恢复双侧接触后可以继续计分，但各次 40 ms 窗口仍按物理时间推进，不因失联暂停。掉落不回溯清零阶段事件、A 时刻几何分、已锁存的 stability、命中记录、恢复记录、峰值和已累计冲量；后续 H 为假，不能继续增加有效命中和冲量，最终 task_success 为零。

**有效顶靠**沿用锤柄与基座前端支撑面的原判据：法向力绝对值至少 `0.1 N`，高度偏差不超过 `2 mm`，法向夹角不超过 `30°`。各关键时点末尾五个连续 250 Hz 采样帧须均成立，首尾约 16 ms。

**闭合包络 E**继续使用 `compute_enclosure_score()`。掌面接触以 250 Hz 采样，末尾至少五帧满足原强度判据 `F >= 0.1` 或 `F / 250 >= 0.0001`（当前等效至少 `0.025 N`）。原几何判定仍要求有效前向外切线交点、两侧距离在 `(0,0.15] m`，以及膨胀 1 mm 凸包内穿透长度不超过 3 mm。

两侧内弯角度从 5° 归一化到 `[0,1]`，满分角约为 narrow `66.1299°`、wide `65.6153°`；外弯侧固定为 0.1。设两侧角度分数为 l、r，`u=max(l,r)`，`h=2lr/(l+r)`（两侧为零时 h=0），内弯分为 `0.8h^0.7+0.2u^0.7`，任一外弯时为 `−(0.8h+0.2u)`。通过原真包络判定才得到 E，否则 E=0。E 的代数包络为 `[-19/55,1]`，放大并加入顶靠分后 wrap 的包络为 `[-76/165,5/3]`，不保证几何上每个端点都可达。

**冻结时点**：stability 只在第一次挥击前冻结；命中记录及两次恢复记录分别锁存。只有第三次挥击及其最终卸载阶段的接触能设置原控制 `impact_started`，它与 touch 连续零计数和锤头接触消失共同决定提前卸载结束，前两次命中不会提前触发最终撤离。提前结束的本帧仍完成评分，下一帧进入 returning 后冻结 force 和 task_success，且停止新增冲量；未提前结束则在 `k == unload_end` 留下最终评分。全部本帧评分在现有评估回调前写入。

三次挥击之间各增加 `0.50 s` 的固定恢复动作，沿原挥击路径逆向平滑返回准备位姿，不重新抓取。第三次后沿用最终卸载机制，没有额外统一的冲击后保持时长。task_success 不表示钉入、做功或冲量阈值达标。

## 4. 加权奖励日志：Episode_Reward

### 4.1 默认人工奖励

从头训练时省略 `--reward-recipe` 即使用本节的十三项人工奖励。权重及示例均按默认配置计算。

`I[条件]` 表示条件成立取 1，否则取 0；`clip` 为截断。闭合、抬升、挥击前时点以及 L、G、H 定义见第 3 节。每个阶段事件、每次挥击命中和每次恢复事件分别只记一次，子步仅更新缓存，reward manager 在环境步末加权求和。

| 键名 | 原始评分 | 权重 | 奖励（惩罚）的情况与引导效果 |
| --- | --- | ---: | --- |
| capture | `(I_left + I_right) / 2`，闭合结束写入 | 2 | 闭合结束时，单侧形成有效锤柄接触得一半分，双侧接触得满分；引导抓取位姿先建立接触，再形成双侧夹持。 |
| lift_progress | `I_bilateral(A) × clip(Δz / lift_threshold, 0, 1)` | 3 | A 时刻保持双侧接触且锤体被抬高时，按高度给予连续奖励，达到 15 cm 得满分；为尚未完成抬升的尝试提供渐进反馈，引导抓姿从接触向有效抬升改进。 |
| lift_success | `L = I[Δz ≥ lift_threshold] × I_bilateral(A)` | 10 | A 时刻真实抬升高度至少 15 cm 且仍有双侧接触时奖励；引导形成能将锤子可靠抬起的有效抓取。 |
| grasp_before_swing | `G = L × I[未掉落(pre_swing)] × I_bilateral(pre_swing)` | 5 | 已成功抬升、到挥击前时刻未锁存掉落且仍有双侧接触时奖励；引导选择能经受搬运和挥击前姿态调整的抓姿。 |
| wrap | `[L(2E + 0.10 seat_closure + 0.15 seat_lift) + 0.25 G seat_before_swing] / 1.5` | 3 | 成功抬升后，对闭合时真包络下的指节内弯包覆给予正分，外弯包络给予负分，并奖励闭合、抬升时的有效顶靠；挥击前时刻满足 G 再奖励有效顶靠。引导形成内弯包覆、锤柄顶靠支撑面的抓姿，提高抓持约束。 |
| orientation | `L × ((1 − 0.6 f) Q_perp + 0.6 f Q_side)` | 2 | 成功抬升后，两指连线越垂直于柄轴得分越高；按初态柄轴接近水平的程度连续加权侧向抓取偏好，f 定义见下文。 |
| distance | `L × clip((p_grip − p_head)·h_A / distance_ref, 0, 1)` | 5 | 成功抬升后，抓取中心沿柄轴正向距锤头越远，得分越高，达到参考距离后饱和；引导在锤柄上选择远离锤头的位置，增大有效力臂。 |
| stability | `G × S(x)`，第一次挥击前计算并锁存 | 30 | 相对初始抓取关系，第一次挥击前锤子相对夹爪的平移、旋转差越小，得分越高；后续掉落不改变该项，引导抓姿减少闭合、抬升和搬运后的滑移和转动。 |
| force | `0.5 mean_i(f_i) + 0.5 min_i(f_i)`，`f_i = ln(1 + F_i / force_ref_n)` | 20 | 根据三次独立窗口平均力综合评分；对数项压缩大力值，最弱项鼓励三次都有力度。已获得的力度分不会因后续掉落清零。 |
| hit_progress | `(hit_1 + hit_2 + hit_3) / 3` | 30 | 每次有效命中增加原始评分 `1/3`，即加权奖励 10；同一次挥击内重复接触不重复增加，引导逐步完成三次命中。 |
| recovery_hold | `(ready_1 + ready_2) / 2` | 20 | 每次恢复结束仍满足 H，且基座位置误差 `< 0.03 m`、姿态误差 `< 10°` 时记一次 ready，即加权奖励 10；两次独立锁存，不要求此前命中。 |
| task_success | `I[hit_1 & hit_2 & hit_3] × I_H`，最终撤离前终值 | 100 | 三次分别有效命中，且最终卸载结束、开始撤离前仍满足 H 时奖励；引导完成连续三次敲击的完整任务。 |
| drop | `I[非 L 或 dropped]`，从 A 时刻起锁存 | −30 | A 时刻抬升失败立即扣 30；成功抬升后，双侧接触条件连续不成立至少 20 ms 也扣 30，恢复接触不撤销；单回合该项最多扣 30。 |

**朝向**：`h_A` 是 A 时刻锤体局部 +X 的世界系柄轴，`c_A` 是 narrow 指尖到 wide 指尖的单位向量。`Q_perp = clip(1 − (c_A·h_A)², 0, 1)`。抓取前缓存姿态对应的柄轴为 `h_0`，桌面法向为世界 +Z，`d_side = normalize(n_table × h_0)`；最终抓姿 final_quat 对应的 mount 局部 +Z 接近方向为 a，`Q_side = clip((a·d_side)², 0, 1)`。连续平躺权重 `f = clip(1 − abs(h_0,z) / sin(π/6), 0, 1)`：初态柄轴水平时保留 `0.4 Q_perp + 0.6 Q_side`，倾角达到 30° 时变为 `Q_perp`；不要求夹指朝某个固定正方向，也不改变接近轨迹。

**距离**：`p_grip` 为 A 时刻两指尖中点，`p_head = p_hammer − 0.24 h_A`。奖励使用有符号柄轴投影，横向偏移不增加 distance。

**稳定性**：人工奖励模式在动作处理时，由最终修正抓姿和缓存的静止锤子位姿计算夹爪系下的初始相对位姿 `p_rel(initial)`、`q_rel(initial)`，在挥击前（4.60 s）计算 `ep = ||p_rel(pre_swing) − p_rel(initial)||`、`er = angle(q_rel(initial), q_rel(pre_swing))`。不读取中间过程或挥击开始之后的位姿漂移。搬运及挥击规划也使用该初始抓取关系，A 快照保留抬升和几何评分功能。归一化误差为：

```text
x = sqrt(0.5 × ((ep / stability_position_scale)²
             + (er / stability_rotation_scale)²))
```

实际读取配置 `stability_position_scale = 0.015 m` 和 `stability_rotation_scale = π/36 rad`（5°）。设 `b = √26 / 3`、`c = √10`，评分为：

```text
S(x) = 1 − 0.1x                              (0 ≤ x ≤ 1)
       0.9 − 0.15(x − 1) / (b − 1)           (1 < x ≤ b)
       0.75 − 0.25(x − b) / (c − b)          (b < x ≤ c)
       0.5 exp(−(x − c) / (2(c − b)))        (x > c)
```

第一次挥击前 G 成立时，零误差、15 mm/5°、20 mm/10°、30 mm/20° 的平移/旋转误差分别得 1、0.90、0.75、0.50，对应加权奖励 30、27、22.5、15。节点之间线性插值，末段与上一段的分数和斜率连续，随误差增大单调趋近零。两项误差不匹配时仍按上述均方根合成计算。到第一次挥击前时点之前，稳定性和两项误差的缓存均为零；在该时点写入并锁存，之后不因三次挥击、恢复、卸载或掉落更新。stability 不取三次最小值、过程最小值、平均值或终局值。

**有效命中与评分时窗**：`F_n = max(−F_head→cap · n_target, 0)`，只匹配锤头 geom 与目标端盖 geom 的世界系合力，取目标外法向的压缩分量；切向力和负投影不计。每次的有效命中条件为 `V_i = scoring_i & ~returning & H & touching & (F_n > 0)`。第一、第二次的评分区间为该次正向挥击开始之后至挥击结束后 40 ms（含恢复开始后的前 40 ms）；第三次从正向挥击开始之后延续至最终卸载窗口结束，提前 returning 后停止计分。

`hit_i` 锁存该次是否曾出现 V_i；`valid_strike = I[any(hit_i)]`，`hit_count = sum(hit_i)`，`hit_progress = hit_count / 3`。`peak_force` 仅在 V_i 成立的子步更新全回合最大值，保证 `valid_strike == 0 ⇒ peak_force == 0`。同一次挥击中多次接触只算一次命中。

**各次独立冲量窗口**：每次首次 V_i 的物理子步记为 `first_i`。设 `N_window = round(impact_window_s / physics_dt)`，默认 `impact_window_s = 0.04 s`、`physics_dt = 0.002 s`，故 `N_window = 20`。只在 `first_i > 0`、`0 ≤ k − first_i < N_window`、尚未 returning、H 成立且锤头仍接触端盖时累加：

```text
J_i ← J_i + F_n × physics_dt
F_i = J_i / (N_window × physics_dt)
impact_impulse = J_1 + J_2 + J_3
strike_force_i = F_i
mean_force = (F_1 + F_2 + F_3) / 3
```

首次有效命中当前子步计入，窗口内接触中断后恢复可继续积分，窗口不会重新计时。窗口允许跨入恢复或最终卸载段，但恢复后期不能创建新的该次命中事件；不对整段卸载或撤离积分。提前撤离或掉落可能截断积分，分母仍为固定 40 ms，不缩短为实际接触时间。保留的 `impulse_scale = 1.0 N·s` 不参与当前力度奖励或 SGS，也不是成功阈值。

**综合力度评分**：默认 `force_ref_n = 10 N`，对每次窗口平均力分别取自然对数：

```text
f_i = ln(1 + F_i / force_ref_n)
force = 0.5 × (f_1 + f_2 + f_3) / 3 + 0.5 × min(f_1, f_2, f_3)
加权力度奖励 = 20 × force
```

均值项衡量三次总体力度，最小值项鼓励三次均有实际力量；未有效命中的那次 `F_i = 0`，其对数项也为零。已有 J_i 不因后来掉落清零，因而已获得的 force 保留；掉落会阻止后续有效积分并影响完整成功。`force` 无物理单位，也没有硬上限，权重 20 不是满分。

| 单回合三次窗口平均力（N） | 原始 force（约） | 加权力度奖励（约） |
| --- | ---: | ---: |
| 5、5、5 | 0.4055 | 8.11 |
| 10、10、10 | 0.6931 | 13.86 |
| 10、10、0 | 0.2310 | 4.62 |
| 10、0、0 | 0.1155 | 2.31 |

例如，假设一次接触力为 50 N、持续 4 ms，其余 36 ms 为零，则该次 `F_i = 5 N`。瞬时峰值较大不等于窗口平均力较大；仅凭 peak_force 无法换算 force。上述表按单回合计算，不能将批次平均的三项 F_i 直接代入公式来代替真实平均奖励，因为对数和最小值运算均为非线性。

**恢复与完整成功**：在两次恢复结束的固定子步，分别比较实际基座位姿与第一次挥击准备位姿 `strike_pos / raised_quat`；H 成立且平移误差严格小于 3 cm、转角严格小于 10° 才锁存 ready_i。该项不检查锤子相对夹爪的漂移，也不检查此前是否命中。`task_success` 仅在第三次挥击后的最终卸载阶段更新，最终值为 `I[hit_1 & hit_2 & hit_3] × I_H`；不要求 recovery_hold 为 1，也没有额外力、冲量阈值。有效命中、恢复成功和最终完整成功是不同指标。

### 4.2 V2.0 奖励配方

VLM 生成的 `reward_vN.json` 的 `version` 为 `"2.0"`。下标 `N` 表示训练阶段。每个 `terms` 项包含 `id`、`weight`、`formula`、`params` 和 `gate`，例如 `{"id": "task_success", "weight": 100.0, "formula": "identity", "params": [], "gate": "none"}`。`mapping_prior` 提供机器人适配后的 `u_mean_RL`、`u_sigma_RL` 和工具系单位向量 `closing_axis_O`。

单回合总奖励为所有项的 `weight × 公式结果 × 门控值` 之和。门控 `none` 取 1，`lift_success` 取 L，`grasp_before_swing` 取 G。公式及参数顺序由执行器规定。

| `formula` | `params` | 计算 |
| --- | --- | --- |
| `identity` | `[]` | 读取 `id` 对应的原始指标。 |
| `power` | `[exponent]` | 将指标裁剪到 `[0,1]` 后取幂。用于 `wrap` 时，负值裁为 0，超过 1 的值裁为 1。 |
| `prior_gaussian` | `[sigma_scale]` | `exp(-0.5 × ((grasp_u_rl - u_mean_RL) / (u_sigma_RL × sigma_scale))²)`。 |
| `prior_axis_power` | `[exponent]` | `grasp_axis_cos2` 取幂。 |
| `force_mean_min` | `[alpha, force_ref_n]` | 先对三次平均力计算 `s_i = log1p(F_i / force_ref_n)`，再取 `alpha × mean(s_i) + (1 - alpha) × min(s_i)`。参考力单位为 N。 |
| `hold_drift` | `[position_scale, rotation_scale]` | 挥击前快照已记录时，计算 `hold_fraction × exp(-translation_drift / position_scale - rotation_drift / rotation_scale)`，否则取 0。两个尺度的单位分别为 m 和 rad。 |

VLM 提示词要求两种先验公式以 L 门控，敲击力奖励以 G 门控，以 `task_success` 为主要终局目标，`drop` 采用负权重。可用指标与公式以结构化输出契约为准。 [配方 Schema](../vlm_reward/reward_schema.py) · [奖励执行器](../generation/rewards.py)

### 4.3 加权奖励的日志统计

训练入口按 `cfg.rewards` 汇总每项奖励。

```text
Episode_Reward/name = weight_name × Episode_Metrics/name
                    = mean(weight_name × s_name)
```

默认人工奖励为每个奖励键使用同名 metric。例如 `mean(force)=0.2` 时 `Episode_Reward/force=4`，`mean(drop)=0.25` 时 `Episode_Reward/drop=−7.5`，`mean(hit_count)=2` 时 `Episode_Reward/hit_progress=20`。配方模式将单回合计算结果记录为 `Episode_Metrics/recipe`，以权重 1 写入 `Episode_Reward/recipe`。物理原始指标仍按第 3 节各自记录。带非线性或门控的配方应先逐回合计算，再对结果求均值。

训练入口按实际回合奖励汇总日志。评估入口将实际 `reward` 与原始指标分别写入 NPZ。奖励配方变化会改变总奖励的尺度与含义，闭环的物理反馈以逐回合原始指标为准。

## 5. 总奖励、课程与终止统计

仅使用默认人工奖励时，单回合总奖励为：

```text
R = 2 capture + 3 lift_progress + 10 lift_success + 5 grasp_before_swing
  + 3 wrap + 2 orientation + 5 distance + 30 stability + 20 force
  + 30 hit_progress + 20 recovery_hold + 100 task_success − 30 drop
```

除 force、task_success 外的正奖励宽松上界为 112；完整成功另加 100，force 的加权贡献为 `20 × force`，无硬上限。总奖励的宽松代数包络为 `[-30 − 76/55, ∞)`，各门控相关，不保证每个值可达。抬升失败时 drop 扣 30，仍可获得 capture 和双侧接触下的部分 lift_progress。wrap 可为负且上界为 `5/3`；其余正向原始评分除 force 外均在 `[0,1]`。原始 hit_count 的范围为 `[0,3]`，但只通过除以 3 后的 hit_progress 加入奖励。

训练日志直接满足 `sum(Episode_Reward/*) = Train/total_reward`，无需乘回合时长 7.14 s。该和覆盖本轮全部尝试，`Mean reward` 取日志最近收到的最多 128 条。

| 指标 | 含义 | 数值范围及解读 |
| --- | --- | --- |
| `Curriculum/scene_sgs/sgs_mean_success` | 每个已有历史的训练场景先计算最近最多 100 个已完成尝试的二值 `task_success` 均值，再对这些场景等权求均值。 | `[0,1]`；尚无任何历史时日志为 0。未观察场景用于采样的 `p_i=0.5` 是先验，不计入该实测均值；初始化和恢复后的零长度回合不计入历史。 |
| `Curriculum/scene_sgs/sgs_mean_probability` | 全部训练场景的归一化采样概率均值。 | 有 N 个训练场景时为 `1/N`，不是成功率或每批整数名额。 |
| `Curriculum/scene_sgs/sgs_min_probability` / `Curriculum/scene_sgs/sgs_max_probability` | 全部训练场景采样概率的最小值与最大值。 | 位于 `[0,1]`，首次更新前均为 `1/N`；默认均匀混合后每个训练场景概率至少为 `0.2/N`。评估场景不参与这些统计。M 个重置环境对场景 i 的预期采样数为 `M*q_i`，实际数量可波动。 |
| `Curriculum/scene_sgs/sgs_observed_scenes` | 有完成回合历史的训练场景数。 | 整数 `[0,N]`；同场景的并行完成回合全部计入最近 100 条历史。 |
| `Episode_Termination/attempt_complete` | 提交给 runner 的每批真实终局样本数，含掉锤与走完轨迹的尝试。 | 每批 N 条，一轮提交 `ROLLOUT_STEPS` 批，当前默认 1 批。 |

教师从第一次 reset 起开放全部训练场景，reset 仍只从离线连续稳定位姿缓存抽样。课程在 reset 事件和动作评分清零前读取旧 `scene_id` 及最终 `task_success`，每个训练场景单独保存最近 100 条二值成功历史；评估场景不累计历史。场景成功率 `p_i` 为历史均值，无历史时以 0.5 计算优先级。首次更新前在训练场景间均匀采样，此后按每 N 个已完成尝试计一批，每 10 批更新一次，无额外热身。

默认 `t=0.0`、`kappa=1.0`、`T=1.0`、`eps=1e-4`、`uniform_mix=0.2`，仅在训练场景间计算 `w_i = (p_i+eps)^(kappa*t) × (1-p_i+eps)^(kappa*(1-t))`、`logit_i = log(max(w_i,eps)+eps)/T`，再用 `q_adaptive = softmax(logit)` 得到自适应概率，最终 `q_i = 0.8 × q_adaptive_i + 0.2/N_train`。成功率越接近 0，场景优先级越高。忽略数值处理项时，80% 的采样概率按失败率分配，其余 20% 均匀分配给全部训练场景。VLM 闭环用 `--split train` 训练，反馈评估场景的采样权重为零。SGS 使用 `task_success` 历史，参数以 `run_config.json` 的 `scene_sgs` 为准。checkpoint 的 `infos["scene_sgs"]` 保存逐场景历史、按训练场景行号递增顺序排列的概率及完成批次计数。切换配方时，闭环通过 `--resume` 恢复该状态并追加 PPO 更新。

掉锤和走完轨迹都以真实终局提交，PPO 的单步目标回报就是该次奖励。未成功抬升或单次未命中仍按原轨迹推进。首次挥击前掉锤时，`hold_fraction` 用已累计的有效持有步数除以原定窗口步数，尚未采样的稳定性和配方 `hold_drift` 为零。

## 6. 如何解读训练和评估结果

- 对同一批次、相同样本，应满足 `task_success ≤ hit_progress ≤ valid_strike ≤ grasp_before_swing ≤ lift_success`，且 `hit_count = 3 × hit_progress`。valid_strike 是至少一次命中的比例，task_success 是三次均命中且最终仍有效持锤的比例，hit_progress 是三次敲击机会中的有效命中比例。抬升成功率高而挥击前有效抓持低，说明搬运过程中发生掉落锁存或挥击前时刻失去双侧接触；不能仅凭这两个均值确定具体原因。
- `peak_force` 是包含失败和未命中零值的平均有效峰值，同时受命中率和命中力度影响。若同批次 `mean(valid_strike) > 0`，可用 `mean(peak_force) / mean(valid_strike)` 查看命中回合的平均峰值；更准确的结果应从逐回合 NPZ 按 `valid_strike == 1` 筛选计算。
- `mean_force = mean(strike_force_1, strike_force_2, strike_force_3)`，且当前固定窗口下 `impact_impulse = 0.04 × (strike_force_1 + strike_force_2 + strike_force_3)`；这些线性关系对同批次日志均值也成立。单回合至少一次有效命中时 mean_force 为正，但它不是成功命中时的条件均值。若同批次 `mean(hit_count) > 0`，可用 `3 × mean(mean_force) / mean(hit_count)` 计算所有有效命中次数上的窗口平均力均值，勿将其解释为峰值或接触期间的平均力。
- `force` 同时受未命中零值、固定窗口平均、对数压缩和单回合最弱一次影响。即使 peak_force 较大，force 仍可能较小。应同时观察 strike_force_1/2/3、hit_progress、drop 和 task_success，不能仅用 force 判断物理冲击大小。
- `translation_drift` 和 `rotation_drift` 是参考基准与挥击前实际相对位姿的差值，包含失败回合；人工奖励基准是初始抓取关系，VLM 配方基准仍为闭合结束状态。平移单位是 m、旋转单位是 rad，较大的漂移均值需要结合逐回合分布和回放判断；这两项不表示过程最大漂移，与旧人工奖励评估结果的含义不同。
- `wrap`、`stability`、`force` 是连续评分；例如 `stability=0.2` 不表示稳定成功率为 20%。stability 在第一次挥击前冻结，因此可以与后续 drop 同时为正；这不是日志矛盾。hit_progress 和 recovery_hold 分别是三个、两个二值事件的均值，分别描述命中机会和恢复机会的成功比例；不代表完整回合成功率。`seat_*` 的均值也只是各时点的顶靠比例。
- 教师课程概率更新会改变采样分布，更新附近奖励或成功率回落不一定表示策略退化。蒸馏时应同时看 `Loss/behavior` 和实际任务指标；模仿误差下降不保证命中率同步上升。
- `evaluate.py` 默认使用名义物理参数和无噪深度，可用 `--noise` 开启深度噪声。`--policy zero` 执行已绑定的候选位姿，`--policy trained` 使用 actor 的确定性均值残差。`--selection all` 逐候选评估，`sdk` 绑定每场景 SDK 最高分候选，`learned` 由 critic 在每场景内选出最高分候选。

范围和换算关系假设原始指标为有限值。底层奖励管理器会将非有限的加权奖励项置零，而训练 wrapper 直接读取指标日志；若原始指标出现 NaN/Inf，则加权日志与实际 reward 可能不再满足上述等式，应结合原始状态排查。

### 6.1 VLM 闭环的评估输出

每阶段训练结束后，调度器用该阶段最终 `model_<iteration>.pt` 执行 `--split eval --policy trained --selection learned --episodes 0 --headless`。`--episodes 0` 在该选择模式下评估所有 `eval` 场景，每场景执行一次。评估从 checkpoint 同目录的 `run_config.json` 读取当轮配方和锤头额外质量。该划分参与奖励修正，独立测试可使用 `hammer_rl.prepare --test-set` 生成的另一份缓存。 [闭环调度器](../vlm_reward/reward_evolution.py) · [评估入口](../hammer_rl/evaluate.py)

| 阶段输出 | 内容 |
| --- | --- |
| `feedback_vN/metrics.npz` | 按回合保存 `scene_id`、`candidate_id`、实际 `reward`、终局原始指标，以及 `learned` 模式下的 `predicted_score`。 |
| `feedback_vN/frames/*.png` | 回放本轮阶段进展最低和最高的代表性尝试，保存 `closure_end`、`lift_end`、`swing_start`、`swing_end` 四个阶段。 |
| `feedback_vN/frames/index.json` | 每张图的 `episode`、`scene_id`、`candidate_id`、`stage`、`file`，以及回放终局的抓取与任务指标。`episode` 对应 NPZ 中从 0 开始的行号。 |

反思请求从 NPZ 读取仿生抓取指标与动态任务指标，按整个反馈集计算 `mean`、`median` 和 `p90`，并发送逐回合数组、图像索引及图像。接触和承托诊断可结合 `capture`、`wrap`、`seat_*` 与四阶段图像，动态表现以 `task_success`、`hit_progress`、各次 `strike_force_*`、漂移、掉落及恢复等原始字段为依据。 [反思提示词](../vlm_reward/prompts/reward_reflection.md)

## 7. 实现依据

- 默认奖励、指标与课程注册。 [环境配置](../hammer_rl/env_cfg.py)
- 候选局部残差合成、固定轨迹、阶段门控、接触包络与敲击力。 [动作与子步指标](../generation/actions.py) · [包络和顶靠判据](../generation/enclosure.py)
- 配方执行、终止条件与课程状态。 [奖励执行器](../generation/rewards.py) · [终止条件](../generation/terminations.py) · [课程逻辑](../generation/curriculum.py)
- 87/50 维观测、冻结深度与随机化。 [观测读取](../generation/observations.py) · [重置与观测构造](../generation/resets.py)
- 网络、PPO 与蒸馏配置，按终局样本汇总训练日志和续训。 [网络与算法配置](../hammer_rl/rl_cfg.py) · [训练入口](../hammer_rl/train.py)
- 候选评分、确定性残差、逐回合评估和闭环产物组织。 [候选推理](../hammer_rl/inference.py) · [评估入口](../hammer_rl/evaluate.py) · [项目框架说明](human_guided_dynamic_grasp_framework.md)
- 本机框架源码位于 `/home/ubuntu20/anaconda3/envs/hammer/lib/python3.11/site-packages/` 下：`mjlab/managers/{metrics,reward,termination}_manager.py` 定义底层日志聚合；`mjlab/rl/config.py` 定义 PPO 默认值；`rsl_rl/utils/logger.py` 定义吞吐、128 回合滑动窗口与打印；`rsl_rl/algorithms/{ppo,distillation}.py`、`rsl_rl/models/{mlp_model,cnn_model}.py`、`rsl_rl/modules/{distribution,cnn}.py` 定义损失、网络、熵与标准差边界。
