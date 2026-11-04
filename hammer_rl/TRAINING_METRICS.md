# 锤击强化学习训练日志指标说明

本文说明当前 `python -m hammer_rl.train` 产生的训练指标、网络配置和统计口径，覆盖教师 PPO、教师向学生蒸馏和学生 PPO 微调三个阶段。依据当前工作区代码以及本机安装的 mjlab 1.6.0、rsl-rl-lib 5.4.2 实现核对；续训的网络与算法参数以对应目录的 `run_config.json` 为准。文中的数值边界均指正常有限数值计算；没有硬上限不代表任意大数值都符合正常物理表现。

## 1. 阅读日志前的统计约定

### 1.1 当前网络、观测和训练阶段

| 阶段 | 策略网络及输入 | 价值网络与优化目标 |
| --- | --- | --- |
| `teacher` | `MLPModel`：87 维 `privileged` → 隐藏层 `(512, 256, 128)` → 6 维高斯均值。 | critic：87 维 `privileged` → 隐藏层 `(256, 128, 64)` → 1 维价值；使用 PPO。 |
| `distill` | 学生 `CNNModel` 使用 `depth` 和 50 维 `public_seed`，教师使用上述特权 MLP。 | 无 critic；固定教师，最小化学生与教师的 6 维确定性输出之间的 MSE。采集时执行学生采样动作。 |
| `student` | 与蒸馏学生相同的 `CNNModel`，从蒸馏 checkpoint 初始化 actor。 | 独立 `CNNModel` 使用相同深度和公开观测，输出 1 维价值、无动作分布；重新通过 PPO 微调。 |

学生深度输入的单样本形状为 `(1, 120, 160)`，依次经过三层卷积：通道数 `(16, 32, 32)`、卷积核 `(5, 3, 3)`、步长 `(4, 2, 2)`，无 padding、无池化。特征形状依次为 `(16, 29, 39)`、`(32, 14, 19)`、`(32, 6, 9)`；展平为 1728 维，与 50 维公开观测拼接后进入隐藏层 `(256, 128, 64)`，输出 6 维高斯均值。

上述隐藏层和卷积层均使用 ELU。教师 actor、两个 PPO 阶段的 critic 和学生的一维公开观测均开启观测归一化；学生深度图直接进入 CNN，不通过该一维运行统计归一化器。当前没有 RNN 或跨尝试的隐藏状态。

部署前通过 `python -m hammer_rl.calibrate` 冻结最终 actor，以训练候选执行确定性残差后的实际总奖励为标签，仅用 MSE 校准 critic。actor 参数和双方归一化统计保持不变。候选评分预测微调后的总奖励，选择时对场景内候选取 argmax；该阶段不改变 PPO 或蒸馏损失。部署权重包含配套 actor 和 critic，蒸馏权重仍只用于动作初始化或单候选评估。

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
| `Mean reward` | 最近最多 128 个已结束回合的实际总奖励均值，当前每回合只有一次奖励。 | force 无硬上限，完整成功另加 100，掉落项权重为 −20；详见第 5 节；不是百分比，也没有归一化到 `[0,1]`。在相同配置和评估分布下通常越大越好。 |
| `Train/total_reward` | 本轮采集的实际总奖励均值，不使用 128 回合滑动窗口。 | 与单回合总奖励具有相同边界，等于同批次十一项加权奖励之和。 |
| `Mean episode length` | 最近最多 128 个已结束回合的平均环境步数。 | 当前有效回合恒为 `1`。单位是环境步，不是秒，也不是 8980 个物理子步。 |
| `Mean action std` | 策略高斯动作分布在各动作维度上的标准差均值，描述探索幅度。 | 当前默认 `[10⁻⁶, 10⁶]`，初值为 `0.3`。这是 `tanh` 前无量纲动作的标准差，不是 `0.30 m` 或 `0.30 rad`；策略输出随后经过 `tanh` 并缩放为抓取残差。 |

当前动作先经过 `tanh`，再在候选抓取的局部坐标系中施加残差：平移各轴尺度为 `0.10 m`，旋转向量各分量尺度为 `π/5 rad`（36°）。这是每个分量的尺度，标准差本身不等于执行后的米制或角度误差；熵和 PPO 动作概率也均对应 `tanh` 前的高斯分布。

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
| `Curriculum/placement/success_rate_flat` | 与终端名称相同 | 仅教师训练，平放近期完整成功率 |
| `Curriculum/placement/success_rate_lean_head_down` | 与终端名称相同 | 仅教师训练，锤头朝下斜靠近期完整成功率 |
| `Curriculum/placement/success_rate_lean_head_up` | 与终端名称相同 | 仅教师训练，锤头朝上斜靠近期完整成功率 |
| `Curriculum/placement/success_rate_inverted` | 与终端名称相同 | 仅教师训练，倒立近期完整成功率 |
| `Curriculum/placement/probability_flat` | 与终端名称相同 | 仅教师训练，下次 reset 的平放目标概率 |
| `Curriculum/placement/probability_lean_head_down` | 与终端名称相同 | 仅教师训练，下次 reset 的锤头朝下斜靠目标概率 |
| `Curriculum/placement/probability_lean_head_up` | 与终端名称相同 | 仅教师训练，下次 reset 的锤头朝上斜靠目标概率 |
| `Curriculum/placement/probability_inverted` | 与终端名称相同 | 仅教师训练，下次 reset 的倒立目标概率 |
| `Curriculum/placement/strength_deficit_*` / `Curriculum/placement/success_samples_*` | 同名 | 仅教师训练，各类成功回合的力度缺口与样本数 |

上述曲线的默认横轴是 runner 更新编号 `it`，从 0 开始，续训会恢复更新编号；`Train/mean_reward/time` 和 `Train/mean_episode_length/time` 则使用当前 logger 累计的采集与学习墙钟秒数。`Total steps` 仅在终端打印，不是同名 TensorBoard 标量。

## 3. 任务原始指标：Episode_Metrics

关键时点：闭合结束约 `2.4 s`；抬升结束为快照 **A**，约 `3.2 s`；挥击前为快照 **B**，约 `16.2 s`。这些均为每次尝试内的仿真时间。

| 指标 | 含义、计算时点与单位 | 单回合范围 | 注释 |
| --- | --- | --- | --- |
| `capture` | 闭合结束时 `(I_left + I_right) / 2`，接触形成分，无量纲。 | `{0, 0.5, 1}` | 闭合结束时，夹爪对应一侧是否存在至少一个有效锤柄接触（0，0.5，1） |
| `lift_progress` | A 时刻双侧接触下，`clip(lift_height / lift_threshold, 0, 1)`；不足 15 cm 也可得部分分数。 | `[0,1]` | 抬升进度奖励，补充了不同高度提供的奖励，信息更密集 |
| `lift_success` | A 时刻锤体真实高度增量至少 0.15 m，且双侧接触；记为 L。 | `{0,1}` | 完整抬升成功奖励 | 
| `grasp_before_swing` | B 时刻 L 成立、未锁存掉落且仍双侧接触；记为 G，之后保持历史阶段事件含义。 | `{0,1}` | 验证挥击之前是否还能保持抓取 |
| `wrap` | 闭合有符号包络放大 2 倍，加三阶段顶靠奖励后除以 1.5，公式见第 4 节。 | `[-76/165,5/3]`，约 `[-0.460606,1.666667]` | 判断是否形成有效包络以及计算两侧指节的包覆质量 |
| `orientation` | A 时刻横向夹柄，平放时增加双向侧向接近偏好；乘 L，无量纲。 | `[0,1]` | 横向抓取与侧向抓取奖励 |
| `distance` | A 时刻抓取中心到锤头中心沿柄轴的投影，除以 0.315 m 后裁剪并乘 L。 | `[0,1]` | 抓取位姿能形成的有效力臂 |
| `stability` | 闭合结束与 B 两点的相对位姿差评分；B 起按当前实时持锤条件 H 门控。 | `[0,1]` | 仅比较闭合结束与挥击前锤子相较于夹爪的转动和滑动，不累计过程最大值 |
| `force` | 历史有效压缩法向力峰值 `peak_force / 10`，即每 10 N 得 1 分。 | `[0,∞)` | 命中后掉落不清零；不按冲量评分 |
| `task_success` | `I_H × I[valid_strike > 0]`，在卸载结束、开始撤离前冻结；表示抓取—保持—命中—卸载结束仍持有。 | `{0,1}` |
| `drop` | L 成立后，双侧接触连续缺失 20 ms 锁存的掉落事件。 | `{0,1}` | 抓取掉落后会施加惩罚 |
| `lift_height` | A 时刻锤体原点相对场景初态的真实 z 增量，单位 m。 | 有符号实数 |
| `translation_drift` | 闭合结束与 B 两点的夹爪系锤体位置差的欧氏距离，B 时刻写入后保持不变，单位 m；失败回合也记录。 | `[0,∞)` |
| `rotation_drift` | 闭合结束与 B 两点的夹爪系锤体姿态最短旋转角误差，B 时刻写入后保持不变，单位 rad。 | `[0,π]` |
| `peak_force` | B 之后至卸载结束、撤离之前，实时有效持锤命中的压缩法向力峰值，单位 N；后来掉落不追溯清零。 | `[0,∞)` |
| `valid_strike` | 是否曾在评分时窗内实时有效持锤命中，锁存后不清零。 | `{0,1}` |
| `impact_impulse` | 首次有效命中起 40 ms 窗口内、H 成立时累计的法向冲量，单位 N·s；仅作指标和课程统计。 | `[0,∞)` |
| `seat_closure` / `seat_lift` / `seat_before_swing` | 闭合/A/B 时刻末尾五个连续 250 Hz 采样帧是否有效顶靠；各自无抬升门控，加入 wrap 时分别由 L/L/G 门控。 | 各为 `{0,1}` |
| `collect` | 子步收集函数返回累计 peak_force 的子步平均，再对回合平均，单位 N；不是最终峰值或瞬时力均值。 | `[0,∞)`，同批次不超过平均 peak_force |

### 接触、掉落、顶靠与评分时窗

复用六个指面各两个锤柄接触槽：前六槽属于 narrow（left），后六槽属于 wide（right）。每侧至少一个槽满足 `found > 0` 且法向力 `> 0`，才算该侧接触；`bilateral = left & right`。

从 A 后第一个子步到既有卸载结束，只对尚未 `returning` 的环境更新失联计数。L 成立后，bilateral 连续失败 `round(0.02 / physics_dt)` 步（当前 10 步）锁存 `dropped`；恢复接触仅清零计数，不恢复掉落资格。B 时刻按原定义锁存 `grasp_valid = G`，之后不因挥击中掉落清零此控制快照。纯评分条件为 `H(t) = G & ~dropped(t) & bilateral(t)`。

短于 20 ms 的暂时失联也会使当前 stability 为零；未锁存掉落时恢复双侧接触，stability 可重新按缓存的两点位姿差评分。最后一帧 bilateral 必须成立才能保留 stability 正分。掉落不回溯清零阶段事件、A 时刻几何分、历史 valid_strike、peak_force 或 impact_impulse。

**有效顶靠**沿用锤柄与基座前端支撑面的原判据：法向力绝对值至少 `0.1 N`，高度偏差不超过 `2 mm`，法向夹角不超过 `30°`。各关键时点末尾五个连续 250 Hz 采样帧须均成立，首尾约 16 ms。

**闭合包络 E**继续使用 `compute_enclosure_score()`。掌面接触以 250 Hz 采样，末尾至少五帧满足原强度判据 `F >= 0.1` 或 `F / 250 >= 0.0001`（当前等效至少 `0.025 N`）。原几何判定仍要求有效前向外切线交点、两侧距离在 `(0,0.15] m`，以及膨胀 1 mm 凸包内穿透长度不超过 3 mm。

两侧内弯角度从 5° 归一化到 `[0,1]`，满分角约为 narrow `66.1299°`、wide `65.6153°`；外弯侧固定为 0.1。设两侧角度分数为 l、r，`u=max(l,r)`，`h=2lr/(l+r)`（两侧为零时 h=0），内弯分为 `0.8h^0.7+0.2u^0.7`，任一外弯时为 `−(0.8h+0.2u)`。通过原真包络判定才得到 E，否则 E=0。E 的代数包络为 `[-19/55,1]`，放大并加入顶靠分后 wrap 的包络为 `[-76/165,5/3]`，不保证几何上每个端点都可达。

**冻结时点**沿用既有卸载规则：原控制 `impact_started`、touch 连续零计数和锤头接触消失仍决定提前卸载结束。提前结束的本帧仍完成评分，下一帧进入 returning 后冻结；未提前结束则在 `k == unload_end` 留下最终评分。全部本帧评分在现有评估回调前写入。

本轮没有增加冲击后保持动作或统一保持时长；task_success 不表示钉入、做功或冲量阈值达标。

## 4. 加权奖励日志：Episode_Reward

### 4.1 单次尝试的奖励公式

`I[条件]` 表示条件成立取 1，否则取 0；`clip` 为截断。闭合、A、B 时点以及 L、G、H 定义见第 3 节。事件每次尝试只计一次，子步仅更新缓存，reward manager 在环境步末加权求和。

| 键名 | 原始评分 | 权重 | 奖励（惩罚）的情况与引导效果 |
| --- | --- | ---: | --- |
| capture | `(I_left + I_right) / 2`，闭合结束写入 | 2 | 闭合结束时，单侧形成有效锤柄接触得一半分，双侧接触得满分；引导抓取位姿先建立接触，再形成双侧夹持。 |
| lift_progress | `I_bilateral(A) × clip(Δz / lift_threshold, 0, 1)` | 3 | A 时刻保持双侧接触且锤体被抬高时，按高度给予连续奖励，达到 15 cm 得满分；为尚未完成抬升的尝试提供渐进反馈，引导抓姿从接触向有效抬升改进。 |
| lift_success | `L = I[Δz ≥ lift_threshold] × I_bilateral(A)` | 10 | A 时刻真实抬升高度至少 15 cm 且仍有双侧接触时奖励；引导形成能将锤子可靠抬起的有效抓取。 |
| grasp_before_swing | `G = L × I[未掉落(B)] × I_bilateral(B)` | 5 | 已成功抬升、到 B 时刻未锁存掉落且仍有双侧接触时奖励；引导选择能经受搬运和挥击前姿态调整的抓姿。 |
| wrap | `[L(2E + 0.10 seat_closure + 0.15 seat_lift) + 0.25 G seat_before_swing] / 1.5` | 3 | 成功抬升后，对闭合时真包络下的指节内弯包覆给予正分，外弯包络给予负分，并奖励闭合、抬升时的有效顶靠；B 时刻满足 G 再奖励有效顶靠。引导形成内弯包覆、锤柄顶靠支撑面的抓姿，提高抓持约束。 |
| orientation | `L × (flat ? 0.4 Q_perp + 0.6 Q_side : Q_perp)` | 2 | 成功抬升后，两指连线越垂直于柄轴得分越高；平放样本还奖励抓姿接近方向与桌面内柄轴侧向对齐。引导横向夹柄，并在平放时偏好从两侧接近的抓姿。 |
| distance | `L × clip((p_grip − p_head)·h_A / distance_ref, 0, 1)` | 1 | 成功抬升后，抓取中心沿柄轴正向距锤头越远，得分越高，达到参考距离后饱和；引导在锤柄上选择远离锤头的位置，增大有效力臂。 |
| stability | `I_H × S(x)`，分段线性插值与指数衰减，见下文 | 20 | 实时持锤条件 H 成立时，闭合结束与 B 两点之间锤子相对夹爪的平移、旋转差越小，得分越高；引导抓姿减少搬运后的滑移和转动，并在评分结束时保持有效持锤。 |
| force | `peak_force / 10`，有效命中峰值保留 | 30 | 评分时窗内满足 H 且锤头有效命中目标端盖时，压缩法向力峰值越大，奖励越高；引导选择能将挥击转化为较大目标法向冲击力的抓姿，历史得分在随后掉落时保留。 |
| task_success | `I_H × I[valid_strike > 0]`，撤离前终值 | 100 | 曾有效命中，且卸载结束、开始撤离前仍满足 H 时奖励；引导完成抓取、抬升、保持、命中并持锤至卸载结束的完整任务。 |
| drop | `I_dropped`，锁存 | −20 | 成功抬升后，在掉落检测时窗内双侧接触条件连续不成立至少 20 ms 时锁存惩罚，恢复接触也不撤销；引导避免搬运、挥击和卸载期间失去有效夹持。 |

**朝向**：`h_A` 是 A 时刻锤体局部 +X 的世界系柄轴，`c_A` 是 narrow 指尖到 wide 指尖的单位向量。`Q_perp = clip(1 − (c_A·h_A)², 0, 1)`。抓取前缓存姿态对应的柄轴为 `h_0`，桌面法向为世界 +Z，`d_side = normalize(n_table × h_0)`；最终抓姿 final_quat 对应的 mount 局部 +Z 接近方向为 a，`Q_side = clip((a·d_side)², 0, 1)`。仅 `placement == 0` 的平放样本使用侧向分，不要求夹指朝某个固定正方向，也不改变接近轨迹。

**距离**：`p_grip` 为 A 时刻两指尖中点，`p_head = p_hammer − 0.24 h_A`。奖励使用有符号柄轴投影，横向偏移不增加 distance。

**稳定性**：闭合结束（约 2.4 s）保存夹爪系锤体位姿，在 B（挥击前，约 16.2 s）计算 `ep = ||p_rel(B) − p_rel(closure)||`、`er = angle(q_rel(closure), q_rel(B))`。不读取中间过程或 B 之后的位姿漂移。A 快照仍用于原有挥击轨迹规划。归一化误差为：

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

H 成立时，零误差、15 mm/5°、20 mm/10°、30 mm/20° 的平移/旋转误差分别得 1、0.90、0.75、0.50，对应加权奖励 20、18、15、10。节点之间线性插值，末段与上一段的分数和斜率连续，随误差增大单调趋近零。两项误差不匹配时仍按上述均方根合成计算。B 前稳定性和两项误差均为零；B 时刻写入两点误差，之后仅根据实时 H 更新评分，卸载结束后冻结。

**有效命中与冲量**：B 之后、卸载结束且尚未撤离时，`F_n = max(−F_head→cap · n_target, 0)`，沿用锤头 geom—端盖 geom 匹配。`V = H & touching & (F_n > 0)`；valid_strike 锁存是否曾出现 V，peak_force 仅累计 V 时的峰值，保证 `valid_strike == 0 ⇒ peak_force == 0`。

首次 V 的物理子步记为 first_hit_step。设 `N_window = round(impact_window_s / physics_dt)`，在 `first_hit_step > 0` 且 `0 ≤ k − first_hit_step < N_window`、H 成立、尚未撤离时，`J ← J + F_n × physics_dt`。首次命中当前子步计入，窗口内接触反复可以继续积分，窗口外不再增加；不对整段卸载或撤离积分。

`impact_window_s = 0.04 s` 控制冲量统计窗口，`impulse_scale = 1.0 N·s` 仅用于课程力度缺口归一化，均不作为成功阈值。peak_force=100 N 时 force=10，加权奖励为 300；无有效命中时 J 和 force 均为零。掉落后 J、peak_force 和 force 保留，最终 stability、task_success 为零。force 是按 10 N 缩放的无量纲分数，peak_force 单位 N，impact_impulse 单位 N·s。

### 4.2 加权奖励的日志统计

当前训练入口 `_RewardLogVecEnvWrapper` 动态遍历 cfg.rewards，覆盖所有十一项奖励日志：

```text
Episode_Reward/name = weight_name × Episode_Metrics/name
                    = mean(weight_name × s_name)
```

每个奖励键都有同名 metric。impact_impulse 仅作为原始 metric，不重复加入奖励。例如 `mean(force)=0.2` 时 `Episode_Reward/force=6`；`mean(drop)=0.25` 时 `Episode_Reward/drop=−5`。

普通 `ManagerBasedRlEnv` / `RslRlVecEnvWrapper` 的底层奖励日志仍可能按配置时长 T 归一化；训练入口覆盖后的日志不除以 T。evaluate.py 使用普通 wrapper，但终端和 NPZ 保存的是实际 reward 和各项原始 values。比较历史实验时须确认统计口径；新旧策略应由同一修改后的评估器重新生成结果，不为旧 NPZ 补字段。

## 5. 总奖励、课程与终止统计

两个 PPO 阶段使用、蒸馏阶段仅记录的单回合任务奖励为：

```text
R = 2 capture + 3 lift_progress + 10 lift_success + 5 grasp_before_swing
  + 3 wrap + 2 orientation + distance + 20 stability + 30 force
  + 100 task_success − 20 drop
```

除 force、task_success 外的正奖励上界为 48，完整成功另加 100；force 的加权贡献为 `3 × peak_force`（峰值以 N 计），无硬上限。总奖励的宽松代数包络为 `[-20 − 76/55, ∞)`，各门控相关，不保证每个值可达。抬升失败时仍可获得 capture 和双侧接触下的部分 lift_progress。wrap 可为负且上界为 `5/3`；其余正向评分除 force 外均在 `[0,1]`。

训练日志直接满足 `sum(Episode_Reward/*) = Train/total_reward`，无需乘回合时长 17.96 s。默认单进程、128 环境、每轮采集一步时 Mean reward 与该和应在打印精度内一致；环境数改变后，128 回合滑动窗口与本轮指标样本集合可能不同。

| 指标 | 含义 | 数值范围及解读 |
| --- | --- | --- |
| `Curriculum/placement/success_rate_*`（上表四类） | 每类最近最多 512 个已完成尝试的原始二值 `task_success` 均值；仅教师训练。 | `[0,1]`；完整成功表示抓取、保持、有效命中且卸载结束前仍持有。空历史的 `0.5` 是启动先验，不是实测成功率；初始化和恢复后的零长度回合不计入。 |
| `Curriculum/placement/probability_*`（上表四类） | 由各类完整成功缺口及成功回合力度缺口计算的下次 reset 目标概率；仅教师训练。 | 默认各类在 `[0.075,0.775]` 内且总和为 1，首次各为 `0.25`；固定保底合计 30%，剩余 70% 动态分配，实际概率经平滑逐步接近目标。M 个重置环境的预期类别数量为 `M*q_k`，实际数量可波动，不是已分配的整数环境数。 |
| `Curriculum/placement/strength_deficit_*` / `Curriculum/placement/success_samples_*` | 每类成功回合的力度缺口 / 历史窗口内完整成功样本数。 | 缺口在 `[0,1]`；成功样本不足 32 条时置零。样本数为 `[0,512]` 的整数。 |
| `Episode_Termination/attempt_complete` | 本次重置的环境中，满足 `episode_length_buf >= 1` 的数量，表示一次尝试正常结束。 | 单次统计为整数 `[0, num_envs]`；日志还会对本轮各次统计求均值，因此格式可带小数。当前 128 个并行环境每步都结束一个回合，通常为 128；它是**结束数量**，不是成功数量，也不是百分比。 |

教师从第一次 reset 起同时开放四类训练摆放。课程在 reset 事件和动作评分清零前读取旧场景及最终结果，每类保留最近 512 条 `task_success`、`peak_force`、`impact_impulse`，不按迭代均值统计。默认四类各满 256 条前均分，此后每 10 个完成批次更新概率。完整成功样本至少 32 条时，以成功回合峰值力、冲量中位数 F、J 计算 `d_strength = 0.7 × 100/(F+100) + 0.3 × impulse_scale/(J+impulse_scale)`，否则为零；`d_success = clip((0.9-p_k)/0.9, 0, 1)`，优先级为 `d_success + 0.5(1-d_success)d_strength`。令 score 为优先级平方，按 `0.075 + 0.70 × score/sum(score)` 分配（全零时均分），即每类保底 7.5%、合计保底 30%，剩余 70% 动态分配；保留将超过 0.775 的份额均分给其余类的逻辑，再以 0.3 的系数平滑更新概率。每个训练场景权重为 `q_k/该类训练场景数`，评估场景权重为零。参数以 `run_config.json` 的 `placement_sgs` 为准；checkpoint 保存并恢复历史、概率及完成批次计数。

`attempt_complete` 配置为正常 terminal，`time_out=False`，环境还设置 `is_finite_horizon=True`；抬升失败、掉落、未命中同样执行到完整环境步结束，不产生独立的失败终止日志。PPO 不将下一次 reset 后的初始状态作为本次尝试的 bootstrap 价值。

## 6. 如何解读训练和评估结果

- 对同一批次、相同样本，应满足 `task_success ≤ valid_strike ≤ grasp_before_swing ≤ lift_success`。抬升成功率高而挥击前有效抓持低，说明搬运过程中发生掉落锁存或 B 时刻失去双侧接触；不能仅凭这两个均值确定具体原因。
- `peak_force` 是包含失败和未命中零值的平均有效峰值，同时受命中率和命中力度影响。若同批次 `mean(valid_strike) > 0`，可用 `mean(peak_force) / mean(valid_strike)` 查看命中回合的平均峰值；更准确的结果应从逐回合 NPZ 按 `valid_strike == 1` 筛选计算。
- `translation_drift` 和 `rotation_drift` 是闭合结束与挥击前的两点差值，包含失败回合。平移单位是 m、旋转单位是 rad，较大的漂移均值需要结合逐回合分布和回放判断；这两项不再表示过程最大漂移，与旧评估结果的含义不同。
- `wrap`、`stability`、`force` 是连续评分；例如 `stability=0.2` 不表示稳定成功率为 20%。只有单回合二值指标的均值才可直接读成比例。`seat_*` 的均值也只是各时点的顶靠比例，不等于抓持成功率。
- 教师课程概率更新会改变采样分布，更新附近奖励或成功率回落不一定表示策略退化。蒸馏时应同时看 `Loss/behavior` 和实际任务指标；模仿误差下降不保证命中率同步上升。
- 正式比较策略时，使用 `evaluate.py` 在相同缓存、划分、候选、随机种子和锤头额外质量下评估。该入口默认关闭物理随机化和深度噪声，`--noise` 可启用深度噪声。评估执行确定性均值，训练执行高斯采样，二者的均值无需完全相等。
- `compare_evaluations.py --checkpoint 权重路径` 自动读取运行配置并调用 `evaluate.py`，分别以 `--selection sdk` 评估 SDK 最高分＋零残差、以 `--selection learned` 评估 critic 选择＋actor 残差。默认 `--episodes 0` 遍历全部评估场景，每组每场景执行一次，最终仅导出 `metrics_comparison.csv`，按四类放置汇总两组的有效命中率（百分比）、平均有效峰值力（N）及训练策略减 SDK 基线的差值；临时 NPZ 在读取后删除。该脚本不导出奖励或抓取成功率，也不分别衡量候选选择和 actor 微调的收益。操作见 [训练指南第 9 节](README.md#9-四类放置的命中率与峰值力对比-csv)。全候选均值、SDK 最高分候选均值、critic 所选候选均值和带课程的训练均值属于不同统计集合。

范围和换算关系假设原始指标为有限值。底层奖励管理器会将非有限的加权奖励项置零，而训练 wrapper 直接读取指标日志；若原始指标出现 NaN/Inf，则加权日志与实际 reward 可能不再满足上述等式，应结合原始状态排查。

## 7. 实现依据

- [环境配置与奖励权重](env_cfg.py)：奖励、指标、单回合时长、传感器和课程注册。
- [动作与逐子步指标计算](mdp/actions.py)：关键时点、阶段门控、漂移、包络、稳定性、法向冲量与有效力峰值。
- [包络和顶靠几何评分](../generation/enclosure.py)：真包络与顶靠判据、角度评分。
- [奖励读取](mdp/rewards.py)、[终止条件](mdp/terminations.py)、[课程逻辑](mdp/curriculum.py)。
- [观测读取](mdp/observations.py)、[重置与观测构造](mdp/resets.py)：87/50 维观测、冻结深度和随机化。
- [网络与算法配置](rl_cfg.py)、[训练入口](train.py)：教师/学生网络、PPO/蒸馏参数、每次采集步数、课程命令行默认值，以及 `_RewardLogVecEnvWrapper` 的加权奖励日志覆盖逻辑。
- [评估入口](evaluate.py)、[四类放置对比](compare_evaluations.py)：确定性评估、逐回合原始指标，以及 SDK 最高分与 critic 所选候选的对比统计。
- 本机框架源码位于 `/home/ubuntu20/anaconda3/envs/hammer/lib/python3.11/site-packages/` 下：`mjlab/managers/{metrics,reward,termination}_manager.py` 定义底层日志聚合；`mjlab/rl/config.py` 定义 PPO 默认值；`rsl_rl/utils/logger.py` 定义吞吐、128 回合滑动窗口与打印；`rsl_rl/algorithms/{ppo,distillation}.py`、`rsl_rl/models/{mlp_model,cnn_model}.py`、`rsl_rl/modules/{distribution,cnn}.py` 定义损失、网络、熵与标准差边界。
