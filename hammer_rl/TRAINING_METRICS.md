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
- `Episode_Reward/*`：训练入口的 `_RewardLogVecEnvWrapper.step()` 将它们覆盖为 `权重 × 同名 Episode_Metrics`，**不除以回合时长，也不乘时间步长**。它们是五项原始评分的加权贡献；详见第 4 节。
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
| `Mean reward` | 最近最多 128 个已结束回合的实际总奖励均值，当前每回合只有一次奖励。 | 当前公式的理论包络约为 `[-0.345455, 10)`，推导见第 5 节；不是百分比，也没有归一化到 `[0,1]`。在相同配置和评估分布下通常越大越好。 |
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
| `Curriculum/placement/phase` | 与终端名称相同 | 仅教师训练 |

上述曲线的默认横轴是 runner 更新编号 `it`，从 0 开始，续训会恢复更新编号；`Train/mean_reward/time` 和 `Train/mean_episode_length/time` 则使用当前 logger 累计的采集与学习墙钟秒数。`Total steps` 仅在终端打印，不是同名 TensorBoard 标量。

## 3. 任务原始指标：Episode_Metrics

关键时点：闭合结束约 `2.4 s`；抬升结束为快照 **A**，约 `3.2 s`；挥击前为快照 **B**，约 `16.2 s`。这些均为每次尝试内的仿真时间。

| 指标 | 含义、计算时点与单位 | 数值范围及解读 |
| --- | --- | --- |
| `Episode_Metrics/lift_height` | A 时刻锤子刚体原点的世界系 z 坐标减去该场景初始 z 坐标，单位 m。 | 有符号实数，代码没有裁剪或固定上下界。成功阈值是每回合 `≥ 0.15 m`，还须满足双侧接触。平均高度不能直接代替成功率。 |
| `Episode_Metrics/orientation` | A 时刻，从窄指尖指向宽指尖的单位向量与锤体局部 +Z 在世界系中的方向做点积，负值截为 0，再乘抬升成功标志。 | `[0,1]`，无量纲。有效抬升且两方向同向平行为 1，垂直或反向为 0，抬升失败也为 0。 |
| `Episode_Metrics/peak_force` | 有效计分窗口内，锤头对目标端盖的接触合力沿端面压缩法向投影的峰值，单位 N；要求 B 时刻有效抓持。 | `[0, ∞)`，无硬上限。日志是包含零值回合的平均有效峰值，不是单次最大峰值，也不是只对命中回合求平均。 |
| `Episode_Metrics/lift_success` | A 时刻高度增量 `≥ 0.15 m`，且夹爪两侧均有对锤柄的正法向接触。 | 单回合 `{0,1}`，日志均值 `[0,1]`；例如 0.75 表示抬升成功率 75%。 |
| `Episode_Metrics/stability` | A 到 B 期间，锤子在夹爪坐标系中的相对位置、姿态变化所产生的稳定评分；要求 B 时刻有效抓持。 | `[0,1]`，无量纲，越大越稳定；有效抓持且零漂移得 1，无有效抓持得 0。具体非线性公式见下文。 |
| `Episode_Metrics/translation_drift` | A、B 两时刻，锤子在夹爪坐标系中的相对位置之差的欧氏范数，单位 m。失败回合也记录。 | `[0, ∞)`，无硬上限，通常越小越好。异常大值可能由脱落、远距离运动或数值异常等造成，单靠该平均值无法确定原因。 |
| `Episode_Metrics/grasp_before_swing` | B 时刻有效抓持：A 已抬升成功、A 后至 B 未锁存掉落，且 B 时刻双侧仍有锤柄正法向接触。该标志随后锁存。 | 单回合 `{0,1}`，均值 `[0,1]`。途中双侧接触判据连续失败 20 ms 即永久失效，恢复接触也不能恢复资格；挥击开始后不再检测此判据。 |
| `Episode_Metrics/return_touch_peak` | 撤离阶段 `returning=True` 时，原场景端面 touch 传感器的峰值，单位 N。 | `[0, ∞)`，无硬上限。诊断撤离时的再接触或残余接触，不参与奖励；不要求抬升或抓持成功。 |
| `Episode_Metrics/seat_before_swing` | B 时刻末尾 5 个连续采样帧，锤柄是否持续满足基座前端面的有效顶靠判据。 | 单回合 `{0,1}`，均值 `[0,1]`。该指标本身不要求抬升成功或途中未掉落，加入 `wrap` 时只乘抬升成功标志。 |
| `Episode_Metrics/wrap` | 抬升成功后的闭合末帧真包络角度评分，加上闭合、抬升、挥击前三阶段的顶靠奖励。 | 当前实现的包络边界约 `[-0.345455, 1.5]`；不是 `[0,1]`。外弯包络可产生负分，顶靠加分合计最多 0.5；抬升失败为 0。详见下文。 |
| `Episode_Metrics/distance` | A 时刻抓取中心到锤头中心的距离 `d`，计算 `clip(d / 0.315, 0, 1) × lift_success`。 | `[0,1]`，无量纲。它鼓励有效抓持时抓取位置离锤头更远，达到 0.315 m 后饱和；不是“离目标越近越好”的距离误差。 |
| `Episode_Metrics/valid_strike` | 有效计分窗口内，在 B 时刻有效抓持的前提下，是否出现过锤头与目标端盖接触且压缩法向力 `> 0`。 | 单回合 `{0,1}`，均值 `[0,1]`。没有额外的 60 N 命中阈值，也不直接单独加奖励。途中已锁存掉落的回合，即便锤子之后碰到目标也不计有效命中。 |
| `Episode_Metrics/force` | 用最终有效峰值 `F_peak` 计算 `F_peak / (F_peak + 60)`，其中力的单位为 N。 | `[0,1)`，无量纲，随峰值力单调增加；0 N 得 0，60 N 得 0.5，120 N 得约 0.6667。均值不能直接反解为平均峰值力。 |
| `Episode_Metrics/seat_closure` | 闭合结束约 2.4 s 时，末尾 5 个连续采样帧是否均满足锤柄顶靠判据。 | 单回合 `{0,1}`，均值 `[0,1]`。满足条件且抬升成功时，给 `wrap` 加 0.10。 |
| `Episode_Metrics/collect` | 每个物理子步执行指标收集函数，并对其返回的“截至当前子步的累计 `peak_force`”求子步平均，再对回合求平均，单位 N。 | `[0, ∞)`，无硬上限。是累计峰值序列的均值；不是最终峰值，也不是瞬时力的时间平均，更不是采集耗时。正常有限数值下，同一批回合的 `collect ≤ peak_force`。 |
| `Episode_Metrics/seat_lift` | A 时刻末尾 5 个连续采样帧是否均满足锤柄顶靠判据。 | 单回合 `{0,1}`，均值 `[0,1]`。满足条件且抬升成功时，给 `wrap` 加 0.15。 |
| `Episode_Metrics/touch_peak` | 有效计分窗口内，原场景端面 touch 传感器的峰值，单位 N；与 `peak_force` 使用同样的有效抓持和结束门控。 | `[0, ∞)`，无硬上限。touch 传感区域汇总的接触与锤头—端盖接触合力投影口径不同，因此不要求等于 `peak_force`；本项不直接用于力奖励。 |
| `Episode_Metrics/grip_distance` | A 时刻两指尖中点到锤头中心的三维欧氏距离，单位 m；锤头中心按锤体原点沿局部 -X 方向偏移 0.24 m 计算。 | `[0, ∞)`，无硬上限。未通过抬升判定时仍记录此值，而 `distance` 会置 0，因此两个均值不能简单相除互算。 |
| `Episode_Metrics/rotation_drift` | A、B 两时刻锤子相对于夹爪的姿态之间的最短旋转角误差，单位 rad。失败回合也记录。 | `[0, π]` rad，即 `[0°,180°]`，通常越小越好；不是四元数分量差。 |

### 顶靠、包络与稳定性的具体判据

**有效抓持与掉落锁存**：六个指面各最多保留两个锤柄接触槽，前 6 槽属于 narrow 侧，后 6 槽属于 wide 侧。每侧至少一个槽同时满足 `found > 0` 和法向力 `> 0`，才算双侧有效接触。A 时刻以该接触条件和抬升高度判定 `lift_success`；从 A 后第一个物理子步至 B，若双侧接触判据连续失败 `round(0.02 / physics_dt)` 步（当前 10 步，即 20 ms），则锁存 `dropped=True`。任一侧失联即可使双侧判据失败，不要求两侧同时失联；锁存后，即使 B 时刻重新接触，`grasp_before_swing`、`stability`、`peak_force` 和 `force` 仍为 0。短于 20 ms 的连续失联允许恢复，B 时刻仍必须有双侧有效接触。

`dropped`、`lost_contact_steps` 和 `grasp_valid` 是内部状态，当前没有独立日志字段；不能从 `grasp_before_swing=0` 单独区分抬升失败、途中掉落或 B 时刻接触失败。掉落锁存只影响稳定与敲击资格，不回溯清零 A 时刻的 `orientation`、`distance` 或 `wrap`，也不禁止给 `wrap` 追加抬升成功门控下的挥击前顶靠奖励。

**有效顶靠**要求锤柄接触基座前端支撑面，法向力绝对值至少 `0.1 N`，接触点到支撑面的高度偏差不超过 `2 mm`，接触法向与支撑面法向夹角不超过 `30°`。每个关键时点检查末尾 5 个连续采样帧，采样频率为 250 Hz，首尾覆盖约 16 ms；这是持续接触判据，不是单帧触碰。

**包络评分**计算为：

```text
wrap = lift_success × (E_closure
                       + 0.10 × seat_closure
                       + 0.15 × seat_lift
                       + 0.25 × seat_before_swing)
```

`E_closure` 是闭合末帧的有符号真包络角度评分。掌面接触按 250 Hz 采样，末尾至少 5 帧均须存在足够强的掌面—锤子接触；当前等效法向力绝对值至少为 `0.025 N`（代码判据为 `F >= 0.1` 或 `F / 250 >= 0.0001`）。几何上要求两侧外切线在 TCP YZ 投影中具有有效前向交点，两侧距离均在 `(0, 0.15] m`，且各线段在膨胀 1 mm 的锤子碰撞凸包内的穿透长度不超过 3 mm。未通过真包络判定时为 0。

内弯侧角度分数从 5° 起归一化并裁剪至 `[0,1]`，narrow、wide 两侧满分角分别约为 `66.1299°`、`65.6153°`；外弯侧固定为 `0.1`，任一侧外弯时采用负向评分。真包络的掌面接触条件与上面的锤柄顶靠条件分别计算，二者不能相互替代。

设两侧角度分数为 `l、r`，`u=max(l,r)`，`h=2lr/(l+r)`，两侧均为 0 时令 `h=0`。内弯评分为 `0.8h^0.7+0.2u^0.7`，有外弯时为 `-(0.8h+0.2u)`。由外弯侧固定为 0.1 可得 `E_closure` 的代数包络为 `[-19/55, 1] ≈ [-0.345455,1]`，加上最多 0.5 的顶靠奖励，得到 `wrap` 的上述范围。这是公式边界，不表示场景几何一定能达到所有端点。

**稳定评分**计算为：

```text
ep = translation_drift，单位 m
er = rotation_drift，单位 rad
x = sqrt(0.5 × ((ep / 0.005)² + (er / (π/36))²))
stability = grasp_before_swing × 252 / (252 + 48x - 43x² + 23x³)
```

平移和旋转同时为 `5 mm / 5°`、`10 mm / 10°`、`15 mm / 15°` 时，在有效抓持前提下分别得 `0.9`、`0.7`、`0.4`。实际公式硬编码使用 **5 mm、5°** 的尺度；`env_cfg.py` 中的 `stability_position_scale=0.01` 和 `stability_rotation_scale=π/36` 当前均未被该公式读取，修改这两个配置字段不会改变稳定评分。

**力计分窗口**从 B 时刻之后开始，当前挥击为 `16 × 10 = 160` 个物理子步，约 `0.32 s`，随后最多卸载 `1 s`，即固定窗口最晚到约 `17.52 s`。压缩法向力为 `max(-F_head→cap · n_outward, 0)`，只取锤头—端盖世界系接触合力的压缩法向分量。挥击结束后，若已有效命中、touch 连续至少 3 个物理子步为 0，且锤头接触已消失，则提前结束该环境的计分；之后撤离时的 touch 另记为 `return_touch_peak`。提前结束计分不提前结束环境 step；每个环境仍执行配置的 8980 个子步并在同一环境步结束回合。

## 4. 加权奖励日志：Episode_Reward

### 4.1 五项奖励函数的计算公式

以下公式均针对**单次尝试**，`s_name` 对应 `Episode_Metrics/name` 的原始评分，`r_name = weight_name × s_name` 是加入总奖励的加权贡献。`clip(x, a, b) = min(max(x, a), b)`，`I[条件]` 表示条件成立取 1，否则取 0；位置单位为 m，力单位为 N，旋转误差单位为 rad。

共用的两个门控量为：

```text
L = lift_success = I[z_hammer,A - z_hammer,initial ≥ 0.15] × I[A 时刻双侧有效接触]
G = grasp_before_swing = L × I[A 后至 B 未锁存掉落] × I[B 时刻双侧有效接触]
```

A 为抬升结束快照，B 为挥击前快照；双侧接触、掉落锁存及各顶靠标志的具体判据见第 3 节。

**1. 包络奖励 `wrap`（权重 1）**

```text
s_wrap = L × (E_closure + 0.10 × seat_closure
                        + 0.15 × seat_lift + 0.25 × seat_before_swing)
r_wrap = s_wrap
```

其中 `E_closure` 为闭合末帧的真包络角度评分，可展开为：

```text
l = 0.1（narrow 侧外弯），否则 clip((θ_narrow - 5°) / (66.129901697435° - 5°), 0, 1)
r = 0.1（wide 侧外弯），否则 clip((θ_wide - 5°) / (65.615340743932° - 5°), 0, 1)
u = max(l, r)
h = 0（l + r ≤ 10⁻⁸），否则 2lr / (l + r)
S_enc = -(0.8h + 0.2u)（任一侧外弯），否则 0.8h^0.7 + 0.2u^0.7
E_closure = I[闭合末帧通过真包络判定] × S_enc
```

`θ_narrow`、`θ_wide` 是两侧近端指节方向与末端指节到指尖方向在关节轴法平面内的投影锐角，单位为度；当 `u ≤ 10⁻⁸` 时，代码直接令 `S_enc = 0`。真包络判定包含持续掌面接触、有效外切线交点及凸包穿透限制；`seat_*` 为相应时点末尾 5 个采样帧持续有效顶靠的二值标志。此处的 `r` 仅表示 wide 侧角度分数，不是总奖励。

**2. 朝向奖励 `orientation`（权重 1）**

```text
a = (p_wide,A - p_narrow,A) / ||p_wide,A - p_narrow,A||₂
n = Q_hammer,A × (0, 0, 1)ᵀ
s_orientation = L × max(a · n, 0)
r_orientation = s_orientation
```

`p_wide,A`、`p_narrow,A` 为 A 时刻两侧指尖的世界坐标，`Q_hammer,A` 为锤体局部系到世界系的旋转矩阵。`a` 从 narrow 指尖指向 wide 指尖，`n` 为锤体局部 +Z 在世界系中的单位方向。

**3. 距离奖励 `distance`（权重 0.5）**

```text
p_grip = (p_wide,A + p_narrow,A) / 2
p_head = p_hammer,A - 0.24 × Q_hammer,A × (1, 0, 0)ᵀ
d = ||p_grip - p_head||₂
s_distance = L × clip(d / 0.315, 0, 1)
r_distance = 0.5 × s_distance
```

`p_hammer,A` 为 A 时刻锤体刚体原点的世界坐标，`p_head` 为锤头中心。有效抬升后，抓取中心离锤头越远，评分越高，在 `d ≥ 0.315 m` 时饱和。

**4. 稳定奖励 `stability`（权重 2）**

```text
ep = ||p_rel,B - p_rel,A||₂
er = angle(q_rel,A, q_rel,B)
x = sqrt(0.5 × ((ep / 0.005)² + (er / (π/36))²))
s_stability = G × 252 / (252 + 48x - 43x² + 23x³)
r_stability = 2 × s_stability
```

`p_rel`、`q_rel` 分别为锤子在夹爪坐标系中的相对位置和单位姿态四元数，`angle` 为两个姿态间的最短旋转角，取值 `[0, π]`。代码实际使用 `5 mm` 和 `5°` 归一化漂移；该项在 B 时刻计算。

**5. 敲击力奖励 `force`（权重 5）**

```text
F_compression(t) = max(-F_head→cap(t) · n_outward, 0)
F_peak = max({0} ∪ {G × F_compression(t) : t 属于有效计分窗口})
s_force = F_peak / (F_peak + 60)
r_force = 5 × s_force
```

`F_head→cap(t)` 为锤头到目标端盖的世界系接触合力，`n_outward` 为目标端面外法向。有效计分窗口从 B 之后开始，到固定卸载窗口结束或满足提前结束条件为止，详见第 3 节。无有效抓持或无压缩接触时 `F_peak = 0`；`60 N` 是评分尺度，力达到该值时 `s_force = 0.5`。

### 4.2 加权奖励的日志统计

当前训练入口使用 `_RewardLogVecEnvWrapper`。底层 `RewardManager` 原本会按配置回合时长归一化日志，但该 wrapper 在每次 step 返回时覆盖全部五项 `Episode_Reward/*`。设同名原始评分为 `s`，训练日志的实际公式是：

```text
Episode_Reward/name = weight_name × Episode_Metrics/name
                    = mean(weight_name × s_name)
```

| 指标 | 计算与含义 | 当前数值范围 |
| --- | --- | --- |
| `Episode_Reward/force` | `5 × mean(force)`，力评分的加权贡献。 | `[0, 5)`。 |
| `Episode_Reward/stability` | `2 × mean(stability)`，稳定评分的加权贡献。 | `[0, 2]`。 |
| `Episode_Reward/wrap` | `mean(wrap)`，包络与顶靠评分的贡献。 | `[-19/55, 1.5]`，约 `[-0.345455, 1.5]`。 |
| `Episode_Reward/orientation` | `mean(orientation)`，朝向评分的贡献。 | `[0, 1]`。 |
| `Episode_Reward/distance` | `0.5 × mean(distance)`，距离评分的加权贡献。 | `[0, 0.5]`。 |

例如，若 `Episode_Metrics/force = 0.1618`，当前训练日志应有 `Episode_Reward/force ≈ 5 × 0.1618 = 0.8090`。显示值已四舍五入，复算时可能有末位差异。

若直接使用 `ManagerBasedRlEnv` 或普通 `RslRlVecEnvWrapper`，未经过训练入口的覆盖逻辑，则底层 `extras["log"]` 中的 `Episode_Reward/*` 仍是 `mean(weight × s) / T`。`evaluate.py` 使用普通 wrapper，但终端和 NPZ 保存的是 `reward` 与各项原始指标，不输出该归一化奖励日志。比较历史实验时，应先确认其训练入口是否包含上述 wrapper；仅凭相同的日志标签不能认定数值口径相同。

## 5. 总奖励、课程与终止统计

当前两个 PPO 阶段真正使用的单回合奖励，以及蒸馏阶段仅用于记录的任务奖励，均为：

```text
R = wrap + orientation + 0.5 × distance + 2 × stability + 5 × force
```

各项边界相加给出总奖励的理论包络 `[-19/55, 10) ≈ [-0.345455,10)`；各指标存在门控与几何关联，这个区间不是所有数值均可达到的保证。尤其 `force < 1`，因此有限力下总奖励不能达到 10。抬升失败时，在正常计算下五项奖励均为 0。

例如，假设同批次五项原始评分均值如下，则该批平均总奖励为：

```text
0.0995 + 0.2468 + 0.5×0.5143 + 2×0.1865 + 5×0.1618 = 1.78545
```

当前应直接有 `sum(Episode_Reward/*) ≈ 1.78545`，无需再乘 `17.96`。在默认单进程、128 环境、每轮采集 1 步时，`Mean reward` 与该和应在打印和浮点精度内一致；环境数少于 128 时，`Mean reward` 可能混合之前的回合，环境数大于 128 时只保留最近 128 个回合，因而可能与本轮全批均值不同。

只有 `stability`、`force` 依赖挥击前有效抓持；`wrap`、`orientation`、`distance` 依赖抬升成功。因此抬升后掉落的回合仍可能获得部分奖励。当前没有单独的掉落惩罚、命中奖励或每步存活奖励。

| 指标 | 含义 | 数值范围及解读 |
| --- | --- | --- |
| `Curriculum/placement/phase` | 教师训练的摆放课程阶段编号，按 `common_step_counter` 切换。 | 原始值 `{0,1,2}`。0：只采样倒立类；1：倒立类加两种倾斜类；2：再加入平躺类。不是成功率，也不是奖励。当前训练入口默认在 500、1200 次共同环境步处切换；直接调用环境构造函数的默认值是 600、1800，应以实际参数和 `run_config.json` 为准。学生和蒸馏默认不启用此课程。 |
| `Episode_Termination/attempt_complete` | 本次重置的环境中，满足 `episode_length_buf >= 1` 的数量，表示一次尝试正常结束。 | 单次统计为整数 `[0, num_envs]`；日志还会对本轮各次统计求均值，因此格式可带小数。当前 128 个并行环境每步都结束一个回合，通常为 128；它是**结束数量**，不是成功数量，也不是百分比。 |

课程使用的 `common_step_counter` 每次向量化环境步只增加 1；而 `Total steps` 会按并行环境总数增加。当前单进程且每轮采集一步时，课程计数通常随更新次数增长，不随 `Total steps` 按 128 倍增长。

`attempt_complete` 配置为正常 terminal，`time_out=False`，环境还设置 `is_finite_horizon=True`；抬升失败、掉落、未命中同样执行到完整环境步结束，不产生独立的失败终止日志。PPO 不将下一次 reset 后的初始状态作为本次尝试的 bootstrap 价值。

## 6. 如何解读训练和评估结果

- 对同一批次、相同样本，应满足 `valid_strike ≤ grasp_before_swing ≤ lift_success`。抬升成功率高而挥击前有效抓持低，说明搬运过程中发生掉落锁存或 B 时刻失去双侧接触；不能仅凭这两个均值确定具体原因。
- `peak_force` 是包含失败和未命中零值的平均有效峰值，同时受命中率和命中力度影响。若同批次 `mean(valid_strike) > 0`，可用 `mean(peak_force) / mean(valid_strike)` 查看命中回合的平均峰值；更准确的结果应从逐回合 NPZ 按 `valid_strike == 1` 筛选计算。
- `translation_drift` 和 `rotation_drift` 包含失败回合。平移单位是 m、旋转单位是 rad，较大的漂移均值需要结合逐回合分布和回放判断，不能直接解释为有效抓持样本的漂移。
- `wrap`、`stability`、`force` 是连续评分；例如 `stability=0.2` 不表示稳定成功率为 20%。只有单回合二值指标的均值才可直接读成比例。`seat_*` 的均值也只是各时点的顶靠比例，不等于抓持成功率。
- 教师课程切换会改变采样分布，阶段边界附近奖励或成功率回落不一定表示策略退化。蒸馏时应同时看 `Loss/behavior` 和实际任务指标；模仿误差下降不保证命中率同步上升。
- 正式比较策略时，使用 `evaluate.py` 在相同缓存、划分、候选、随机种子和锤头额外质量下评估。该入口默认关闭物理随机化和深度噪声，`--noise` 可启用深度噪声。评估执行确定性均值，训练执行高斯采样，二者的均值无需完全相等。
- `evaluate.py` 的 `--episodes 0` 遍历选定划分和摆放类型下的全部候选；`compare_evaluations.py` 再筛选每个场景 SDK 分数最高的一个候选，分别按四类放置汇总。全候选均值、每场景最高分候选均值和带课程的训练均值属于不同统计集合。

范围和换算关系假设原始指标为有限值。底层奖励管理器会将非有限的加权奖励项置零，而训练 wrapper 直接读取指标日志；若原始指标出现 NaN/Inf，则加权日志与实际 reward 可能不再满足上述等式，应结合原始状态排查。

## 7. 实现依据

- [环境配置与奖励权重](env_cfg.py)：奖励、指标、单回合时长、传感器和课程注册。
- [动作与逐子步指标计算](mdp/actions.py)：关键时点、阶段门控、漂移、包络、稳定性与力峰值。
- [包络和顶靠几何评分](../generation/enclosure.py)：真包络与顶靠判据、角度评分。
- [奖励读取](mdp/rewards.py)、[终止条件](mdp/terminations.py)、[课程逻辑](mdp/curriculum.py)。
- [观测读取](mdp/observations.py)、[重置与观测构造](mdp/resets.py)：87/50 维观测、冻结深度和随机化。
- [网络与算法配置](rl_cfg.py)、[训练入口](train.py)：教师/学生网络、PPO/蒸馏参数、每次采集步数、课程命令行默认值，以及 `_RewardLogVecEnvWrapper` 的加权奖励日志覆盖逻辑。
- [评估入口](evaluate.py)、[四类放置对比](compare_evaluations.py)：确定性评估、逐回合原始指标与 SDK 最高分候选统计。
- 本机框架源码位于 `/home/ubuntu20/anaconda3/envs/hammer/lib/python3.11/site-packages/` 下：`mjlab/managers/{metrics,reward,termination}_manager.py` 定义底层日志聚合；`mjlab/rl/config.py` 定义 PPO 默认值；`rsl_rl/utils/logger.py` 定义吞吐、128 回合滑动窗口与打印；`rsl_rl/algorithms/{ppo,distillation}.py`、`rsl_rl/models/{mlp_model,cnn_model}.py`、`rsl_rl/modules/{distribution,cnn}.py` 定义损失、网络、熵与标准差边界。

`mdp/actions.py` 中有一处旧注释写成 `distance + 4*force`，与实际注册权重不一致。本文使用 `env_cfg.py` 中实际生效的 `0.5*distance + 5*force`；后续修改配置或框架版本时，应按实际运行实现重新核对范围。
