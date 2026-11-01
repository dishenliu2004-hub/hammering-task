# hammering-task：奖励函数最小修改操作说明

> 给本地 Codex：直接在当前本地仓库完成下述修改，随后执行最小验证并报告结果。不要只给方案。
>
> 本轮只修改“一次初始抓姿决策”的奖励计算、必要的评分缓存及现有日志说明。不得把本说明扩展成完整系统重构、动作空间改造或新训练框架。

## 1. 任务目标与依据

仓库：<https://github.com/dishenliu2004-hub/hammering-task.git>

分析基准提交：`f91fd72395c7438f4a490e75709aaaf3d9cbbccb`，2026-10-31。该提交仅用于说明已审计实现，不要求回退、切换或覆盖当前本地版本。以本地当前源码和用户已有修改为基础，定位对应函数后实施最小补丁。

当前任务始终是：

1. 策略在尝试开始时输出一次抓取位姿残差。
2. 现有固定程序完成接近、闭合、抬升、搬运、调姿、挥击、卸载及撤离。
3. 整次物理执行的结果给最开始这一次抓姿决策评分。
4. 后续运动、夹持控制及执行时序不由强化学习训练。

本轮解决以下具体问题：

- 旧朝向项将窄指到宽指的方向锁定为与锤子局部 `+Z` 同向，不能正确鼓励平放锤子的侧抓。
- 抬升未达到阈值时，现有五项奖励全为零，缺少接触形成和部分抬升信号。
- 搬运后的抓持资格在挥击开始时锁存，挥击中脱手仍可能拿到敲击分。
- 稳定性只比较抬升结束与挥击前两个端点，且没有读取配置中的尺度。
- 缺少独立的阶段成功、完整任务成功及掉落奖励。
- 峰值力直接作为输出评分，对单个瞬时峰值过于敏感。

本说明是在前一轮完整分析基础上收敛出的最小实施版本。新增几何距离场、碰撞筛查、可行通道检测等建议不在本轮执行范围内。

## 2. 必须遵守的修改约束

### 2.1 最小修改、复用现有实现

- 优先在现有类、现有方法和配置中直接修改，不新建奖励管理器、状态机、包装器、兼容层或工具模块。
- 复用现有 `handle_contact`、`head_contact`、掌部顶靠指标、`_relative_pose()`、四元数工具和 `values` 缓存。
- 复用现有 `RewardTermCfg`、`MetricsTermCfg`、`rewards.score()`、训练日志和评估入口。
- 不复制已有算法，不自行实现四元数运算、接触查询、轨迹插值、日志保存或评估器。
- 不增加新依赖、新命令行开关、新运行模式或并存的旧版/新版奖励实现。
- 不增加防御性代码、异常吞掉、自动重试、自动回退、缺字段补零、NaN 替换、通用输入校验、运行时断言或安全检查。
- 不新增安全过滤器、碰撞检测器、限速/限力保护、轨迹中止条件或权限流程。
- 不顺手删除或重构项目原本已有的检查；本轮要求是不增加此类代码。
- 不增加与本次奖励无关的测试、报告模块、绘图面板或通用抽象。
- 不做整文件格式化、无关改名、目录移动或其他顺手清理。
- 只删除因本补丁确实不再使用的字段或局部代码，不保留闲置参数和兼容别名。

公式中明确规定的归一化、截断和抓持/命中判据属于本次奖励定义；不要将它们扩展成新的安全检查系统。

### 2.2 固定项目边界

保持以下内容不变：

- 一次策略动作、一个完整尝试、一次终局回报的环境结构。
- 六维动作维数、`process_actions()` 的位姿映射、动作幅度和旋转中心。
- AnyGrasp 调用、候选库、候选排序、开度标定及额外深入偏移。
- `_plan_strike()`、`_swing_pose()`、`_swing_progress()`、`apply_actions()` 的运动与控制行为。
- 轨迹、速度、加速度、阶段时长、闭合控制、卸载及撤离规则。
- PPO、网络结构、观测、教师—学生流程、随机化及课程采样。
- XML、锤子模型、接触参数、摩擦、质量、惯量及控制约束。
- 当前 `scale_rewards_by_dt=False` 和真实 terminal 语义。

不要因为发现侧抓候选不足而扩大动作范围、生成新候选或引入 SGS。本轮仅改变奖励，动作覆盖不足应在最终说明中如实列为剩余限制。

不要提交或推送代码，不启动长期训练，不覆盖已有模型和训练结果。

## 3. 文件范围与复用关系

先阅读当前仓库说明及适用的 `AGENTS.md`，确认当前实现位置，不需要重新搜索论文。

| 文件 | 本轮处理 |
| --- | --- |
| `hammer_rl/mdp/actions.py` | 修改奖励计算、少量评分缓存、评分缓存重置及评价时窗 |
| `hammer_rl/env_cfg.py` | 修改奖励权重、尺度参数，补齐新增同名 metrics |
| `hammer_rl/TRAINING_METRICS.md` | 同步奖励、指标、单位、门控及评价时点 |
| `hammer_rl/compare_evaluations.py` | 仅同步确实失真的说明/标题，不扩展指标面板和统计架构 |

以下文件在基准版本中已经能够复用，默认不改：

- `hammer_rl/mdp/rewards.py`：`score(env, name)` 直接读取 `values[name]`，足以支持新增项。
- `hammer_rl/train.py`：已动态遍历 `cfg.rewards`。
- `hammer_rl/evaluate.py`：已动态遍历 `term.values` 捕获结果、打印并写入 NPZ。
- `visualization/compare_hammering.py`：已按配置动态计算总奖励。
- `visualization/hammer_rl_training.py`：继续使用“有效命中次数”和“命中条件下的平均峰值力”。

不要为了保持固定文件数量而让实际接口失配。若本地版本与上述位置不同，只在对应现有实现处做同等必要修改，并在最终报告中说明。

## 4. 最重要的时序约束：评分变量不能改变固定执行流程

基准实现中的以下变量参与控制流程，不都是普通奖励缓存：

- `grasp_valid`：挥击前 B 时刻的抓持资格快照。
- `impact_started`：也参与提前卸载结束判断。
- `finished`、`unloaded`、`returning`、`return_start`：决定卸载和撤离过程。

必须保持这些变量原有的控制含义与赋值条件。特别是：

- 不把 `grasp_valid` 改成随挥击期间掉落而清零的变量。
- 不把 `impact_started` 改成新的“实时有效命中”评分标志。
- 不因新的 `task_success`、`drop` 或冲量结果提前结束、延后结束或重置环境。
- 不通过修改 `finished` 的条件来凑出新的保持窗口。

新的实时保持条件只用于奖励、指标和冲量积分。`impact_started` 原来服务控制的计算必须保留；纯评分的“曾发生过有效命中”使用 `values["valid_strike"]`。

### 4.1 评分结束时点

复用现有时序，不新增 `finalize_reward()` 或其他终局处理框架：

- 从 A 时刻之后到 `unload_end`，只对 `returning == False` 的环境更新保持、漂移及相关终局评分。
- 提前卸载结束的本帧仍完成评分；下一帧进入 `returning` 后冻结这些评分。
- 达到最长卸载时间的环境，在 `k == unload_end` 留下最终评分。
- 撤离期间继续保留原有 `return_touch_peak` 诊断，但不再改变本轮任务评分。
- 所有本帧评分写入必须发生在现有 `on_physics_step` 回调之前，避免评估入口捕获旧值。

“任务结束仍持锤”准确指“既有卸载结束、开始撤离之前仍持锤”。不额外增加 50 ms、100 ms 或其他保持动作，也不宣称本轮建立了统一的冲击后保持时长。

## 5. 本轮采用的奖励与参数

### 5.1 总奖励

严格使用下面这组首轮权重，本轮不要自动搜索或调整权重：

| `values` 键名 | 权重 | 定义 |
| --- | ---: | --- |
| `capture` | 2 | 闭合结束时的接触形成分，取 0、0.5 或 1 |
| `lift_progress` | 3 | 仍有双侧接触条件下的部分抬升进展 |
| `lift_success` | 10 | 现有抬升成功事件 |
| `grasp_before_swing` | 5 | 现有挥击前有效抓持事件 |
| `wrap` | 3 | 现有有符号包络/顶靠分除以 1.5 |
| `orientation` | 2 | 横向夹柄；平放时增加侧向接近偏好 |
| `distance` | 1 | 沿柄轴的有效抓取距离分 |
| `stability` | 8 | 全评价时段最大相对漂移对应的保持评分 |
| `force` | 10 | 首次有效命中短时窗的归一化法向冲量分 |
| `task_success` | 50 | 有效命中过，且既有卸载结束时仍持锤 |
| `drop` | −20 | 抬升成功后锁存的掉落事件 |

所有事件每次尝试只计一次。通过现有 reward manager 在环境步末求和，不在物理子步里反复累加这些奖励。

除 `wrap` 保留小幅有符号负值外，正向原始质量分均在 `[0,1]`。非终局正奖励上界为 44，完整成功奖励为 50。这是明确任务优先级的工程起点，不是已经验证的最优参数。

### 5.2 参数

| 参数 | 本轮取值与处理 |
| --- | --- |
| `lift_threshold` | 保留 `0.15 m` |
| `distance_ref` | 保留 `0.315 m` |
| `stability_position_scale` | 默认设为 `0.005 m`，实际计算必须读取该字段 |
| `stability_rotation_scale` | 保留 `π/36 rad`，实际计算必须读取该字段 |
| `impact_window_s` | 新增 `0.04 s` |
| `impulse_scale` | 新增 `1.0 N·s` |
| `force_scale` | 删除已不再使用的旧 60 N 归一化字段及其失真说明，不保留别名 |

40 ms 和 1 N·s 是让首轮修改可直接执行的显式起始超参数，未经过实物或本机数据标定。它们不代表真实冲击脉冲长度或任务达标阈值，也不是从旧 60 N 参数换算而来。本轮不新增自动标定、动态归一化或兜底逻辑；在完成后报告原始冲量分布，供后续实验决定是否调整。

## 6. 逐项实现要求

### 6.1 共用接触定义与最少缓存

复用现有 `handle_contact`：

- narrow 侧至少一个槽有接触且法向力大于零，记作 `left`。
- wide 侧同理，记作 `right`。
- `bilateral = left & right`。

继续使用本项目现有接触槽排列和原始判据，不增加新的接触强度阈值，不重建接触查询。

新增必要的 `values` 项：

- `capture`
- `lift_progress`
- `impact_impulse`
- `task_success`
- `drop`

另新增一个逐环境整数缓存 `first_hit_step`，初始值为 0，用来记录首次有效命中的物理子步编号。复用现有 `values` 分配/重置方式；将该整数缓存加入现有 `reset(env_ids)` 的清零范围。

现有 `translation_drift`、`rotation_drift` 直接用于保存累计最大漂移，不新增历史轨迹数组。不要同时建立同含义的另一套状态或缓存。

### 6.2 闭合接触形成奖励 `capture`

在现有 `closure_end` 时刻写入：

$$
C=\frac{I_{\rm left}+I_{\rm right}}{2}.
$$

- 抓空：0。
- 一侧形成接触：0.5。
- 两侧形成接触：1。

它是离散的接触形成奖励，不是连续几何距离奖励。不要新增点到网格距离、SDF、夹持体积判定或碰撞可行性模块来实现它。

### 6.3 部分抬升与成功抬升

在现有 `lift_end`，即 A 时刻，继续读取真实锤体高度增量：

$$
\Delta z=z_h(t_A)-z_h(0).
$$

新增：

$$
Q_{\rm lift}
=I_{\rm bilateral}(t_A)
\operatorname{clip}\left(\frac{\Delta z}{h_{\rm lift}},0,1\right).
$$

写入 `lift_progress`。

`lift_success` 继续使用原有定义：

$$
L=I[\Delta z\geq h_{\rm lift}]\ I_{\rm bilateral}(t_A).
$$

不要改为按夹爪抬升量评分；不要增加新的持续时间门槛或额外抬升动作。

### 6.4 朝向 `orientation`

删除旧 `max(c·n_h,0)` 评分，改为下面的定义，仍在 A 时刻计算并乘 `L`。

使用：

- `h_A`：A 时刻锤柄轴，当前旧锤子模型为锤体局部 `+X` 转到世界系。
- `c_A`：A 时刻从 narrow 指尖到 wide 指尖的单位向量，沿用现有计算。
- `h_0`：缓存中抓取前锤体姿态对应的锤柄轴。
- `a`：本次最终抓姿的接近方向，即 `final_quat` 对应的 mount 局部 `+Z`。
- `n_t`：当前桌面法向，沿用场景世界 `+Z`。
- `flat`：沿用 `bank["placement"][scene_id] == 0`。

横向夹柄分：

$$
Q_\perp=\operatorname{clip}\left(1-(c_A^\top h_A)^2,0,1\right).
$$

平放侧向分：

$$
d_{\rm side}=\operatorname{normalize}(n_t\times h_0),
\qquad
Q_{\rm side}=\operatorname{clip}\left((a^\top d_{\rm side})^2,0,1\right).
$$

最终：

$$
O=L\begin{cases}
0.4Q_\perp+0.6Q_{\rm side},&\text{flat},\\
Q_\perp,&\text{其他摆放}.
\end{cases}
$$

实现要求：

- 复用已有 `quat_apply`、`torch.cross`、`torch.nn.functional.normalize` 和逐环境 mask。
- 对非平放样本，侧向分不参与奖励。
- 不要求窄指/宽指朝向锤头某个固定正方向。
- 不把上述平方改回单向点积。
- 不新增侧向通道、碰撞或可达性检查。
- 不改变接近轨迹；该分只是抬升成功后的低权重偏好。
- 两侧几何分可以相同，不代表不对称夹爪在受力上等价；动态保持分负责区分实际表现。

### 6.5 包络 `wrap`

复用 `compute_enclosure_score()`、现有掌部/顶靠传感器及三次 `seat_*` 结果，不改 `generation/enclosure.py` 的机构判定。

设 `E` 为现有闭合包络分，`G` 为 B 时刻的现有有效抓持快照，最终形式为：

$$
W=
\frac{
L(E+0.10\,seat_{\rm closure}+0.15\,seat_A)
+0.25\,G\,seat_B
}{1.5}.
$$

- A 时刻写入前半部分。
- B 时刻在得到该时刻实际 `G` 后追加后一部分。
- 保留原有外弯负分。
- 不把历史构型奖励反复乘除归一化尺度。
- 只把 B 时刻顶靠奖励的资格改为 `G`，不要为此改变 `G` 的原有定义。

### 6.6 抓取距离 `distance`

在 A 时刻复用现有抓取中心、锤头中心和柄轴，改为：

$$
\ell=(p_{\rm grip}-p_{\rm head})^\top h_A,
\qquad
D=L\,\operatorname{clip}\left(\frac{\ell}{distance_{\rm ref}},0,1\right).
$$

- `distance` 使用轴向投影，不再以横向偏移增加奖励。
- `grip_distance` 保留原有欧氏距离诊断语义，并在文档中区分。
- 不新增通用工具几何描述、柄端范围检测或有效力臂求解器。
- 不切换到 `assets/tool_hammer_a03/hammer.xml` 中的另一套锤子模型。
- 不改变任何后续轨迹以配合该奖励。

### 6.7 持续保持和掉落

将现有“成功抬升后，双侧接触连续缺失 20 ms 锁存掉落”的评价区间，延长至当前环境既有卸载结束。

复用：

- `lost_contact_steps`
- `dropped`
- 现有 20 ms 对应的子步数
- `bilateral`

只在当前环境尚未 `returning`、且处于 A 之后到 `unload_end` 的评价区间更新；进入撤离后冻结。

不要新增另一套失联计数、重抓恢复、接触滞回或掌指组合判据。`dropped` 一旦锁存，本次尝试不恢复。

记录：

$$
drop=I_{\rm dropped}.
$$

B 时刻仍按原代码计算并锁存 `grasp_valid`，保留 `grasp_before_swing` 的历史阶段事件含义。

定义一个局部的实时评分条件：

$$
H(t)=G\land\neg dropped(t)\land bilateral(t).
$$

该条件只用于评分，不回写 `grasp_valid` 或其他控制状态。最后一帧的 `bilateral` 必须进入终局保持评分，避免“刚失联、尚未累计到 20 ms”时把正分冻结到回合末。

### 6.8 全评价时段稳定性 `stability`

保持 A 时刻保存的相对位姿基准。复用 `_relative_pose()` 和 `quat_error_magnitude()`。

在 A 之后、撤离之前的评价区间，累计：

$$
e_p^{\max}=\max_t\|p_{gh}(t)-p_{gh}(t_A)\|,
$$

$$
e_R^{\max}=\max_t\operatorname{angle}(q_{gh}(t_A),q_{gh}(t)).
$$

分别写入现有 `translation_drift`、`rotation_drift`。不保存全部时间序列，不计算排序分位数。

保留原有稳定性曲线，只替换误差统计和参数读取：

$$
x=\sqrt{\frac12\left[
\left(\frac{e_p^{\max}}{\sigma_p}\right)^2+
\left(\frac{e_R^{\max}}{\sigma_R}\right)^2
\right]},
$$

$$
S=I_H\frac{252}{252+48x-43x^2+23x^3}.
$$

其中 `σ_p`、`σ_R` 分别读取配置中的稳定性尺度，不再硬编码。

- 在 B 之前可以累计最大漂移，但不要把尚未锁存的 B 资格当成已成功搬运。
- B 时刻及之后，根据当前 `H` 更新评分。
- 暂时失联时当前评分为零；未锁存掉落且接触恢复时可重新依据累计最大漂移评分。
- 已锁存掉落后保持零分。
- 撤离前最后一帧的评分为终值。

### 6.9 首次有效命中与法向冲量

复用 `head_contact` 和现有压缩法向力投影：

$$
F_n(t)=\max(-F_{\rm head\rightarrow cap}(t)^\top n_{\rm target},0).
$$

本轮继续使用已有“锤头 geom—端盖 geom”匹配，不新增工作面分割、入射速度阈值或接触几何校验。

本帧有效命中条件：

$$
V(t)=H(t)\land touching(t)\land[F_n(t)>0].
$$

仅在原挥击开始之后、当前环境撤离之前的评价时段使用它。

- `values["valid_strike"]` 记录是否曾发生 `V(t)`；命中后即使后来掉落，也保留该历史事件。
- 第一次 `V(t)` 发生时写入 `first_hit_step`，之后不覆盖。
- 保持 `self.impact_started` 原有控制计算，与该纯评分事件分开。

本轮积分窗口明确采用：

> 首次有效命中起的 40 ms 内，累计仍满足实时持锤条件的压缩法向力；包含窗口内短暂的接触反复；撤离开始后停止。首次命中之后超过此窗口的接触不再增加冲量。

因此无需增加 `window_closed`、分离检测或滞回状态。

设：

$$
N_{\rm window}=\operatorname{round}
\left(\frac{impact_{\rm window\_s}}{\Delta t}\right).
$$

只在 `first_hit_step > 0`、`0 <= k-first_hit_step < N_window`、`H` 成立且仍处于评价区间时累计：

$$
J\leftarrow J+F_n\Delta t.
$$

保存到 `impact_impulse`，单位 N·s。首次命中的当前子步必须计入。

旧 `force` 键继续存在，但其明确新语义为：

$$
F=I_H\frac{J}{J+impulse_{\rm scale}}.
$$

注意：

- 不把力 N 和冲量 N·s 混用。
- 不按物理子步重复累加 `force` 奖励；子步只更新原始冲量与最终评分缓存。
- 掉落后保留已发生的原始 `impact_impulse` 诊断，但将 `force` 评分置零。
- 无有效命中时 `J=0`，评分自然为零。
- 不对整段 1 s 卸载过程积分，不计撤离接触。
- 本轮完整成功没有额外冲量达标阈值；冲击输出大小由连续 `force` 项区分。

### 6.10 峰值力诊断必须与有效命中统计一致

保留 `peak_force` 的 N 单位以及已有绘图入口。将其累计资格与本轮“当时有效持锤的命中”一致，不能累计脱手飞锤的峰值。

必须保持：

$$
valid_{\rm strike}=0 \Longrightarrow peak_{\rm force}=0.
$$

已有 `visualization/hammer_rl_training.py` 使用平均峰值除以命中比例，依赖这个统计关系。

- 后来掉落时，不追溯清零 `valid_strike` 或 `peak_force`。
- 后来掉落由 `drop`、`task_success`、`force` 和 `stability` 体现。
- `peak_force` 是原有效评分时段内的有效命中峰值，不要求与 40 ms 冲量窗口完全相同。
- `touch_peak`、`return_touch_peak` 保持诊断用途；不得用于替换本轮奖励。
- 不改动卸载控制依赖的原始 touch 读数、计数及原 `impact_started` 条件。

### 6.11 完整任务成功 `task_success`

在挥击开始后到当前环境卸载结束前，更新：

$$
task_{\rm success}
=
I_H\,I[valid_{\rm strike}>0].
$$

进入撤离后冻结该值。只有卸载结束前最终仍满足持锤条件的有效命中，才能获得 50 分。

这里的“完整成功”是当前固定端盖场景中的“抓取—保持—命中—卸载结束仍持有”。不要把它描述为钉入成功、钉入深度达标、机械做功达标或冲量阈值达标。

## 7. 配置、日志与评估集成

### 7.1 奖励配置

在现有 `env_cfg.py` 中注册第 5 节的奖励键与权重，继续使用现有 `rewards.score`。

关键要求：

> 每个出现在 `cfg.rewards` 中的键，都必须同时注册同名 `Episode_Metrics`。

现有训练入口通过同名 metric 乘权重生成 `Episode_Reward`。漏注册会使现有接口失配，不应新增 fallback 来掩盖它。

`impact_impulse` 是新增原始指标，需要注册 metric，但不单独再加一遍奖励。

### 7.2 现有评估与展示

- 复用 `evaluate.py` 自动导出全部 `values` 的能力，不另写评估程序。
- 复用 `compare_hammering.py` 的动态奖励求和。
- 保持 `valid_strike` 与 `peak_force` 的统计对应，使训练力曲线脚本无需改动。
- `compare_evaluations.py` 继续保留现有布局；只更新受影响的说明和必要标题。
- 将旧 `force` 描述改为“归一化法向冲量评分”，不要把它的单位写成 N。
- 将 `valid_strike` 表述为“当时有效持锤的命中事件”，与 `task_success` 分开。
- 新指标由现有 NPZ 和控制台输出提供，本轮不新增图表系统。
- 不为旧评估 NPZ 增加字段补齐、版本推断或兼容迁移。新旧奖励比较应在同一修改后评估器下重新生成结果。

### 7.3 文档同步

更新现有 `TRAINING_METRICS.md`，只同步本轮相关内容：

- 总奖励和权重。
- 新增指标定义与单位。
- 朝向公式、距离投影、包络归一化。
- 稳定性覆盖时段、最大漂移及实际生效尺度。
- 冲量窗口和 `force` 新语义。
- `valid_strike` 与 `task_success` 的区别。
- 卸载结束冻结评分，撤离不纳入任务评分。
- 40 ms 和 1 N·s 是待实验标定的首轮超参数。
- 旧总奖励上限、旧五项奖励说明及过期局部注释。

不要为本轮另建项目 README、设计报告或第二份指标文档。

## 8. 实施顺序

1. 确认当前源码、已有本地修改、奖励计算与控制状态依赖关系。
2. 在 `actions.py` 中复用接触计算，加入必要评分项和一个首次命中计时缓存。
3. 先实现接触/抬升/朝向/包络/距离评分，再实现持续保持、最大漂移、冲量及终值。
4. 保留原控制变量与控制流程，尤其保留原 `impact_started` 对卸载的作用。
5. 在 `env_cfg.py` 中统一权重、有效参数与同名 metrics。
6. 同步既有指标文档和确实受影响的展示说明。
7. 执行下面的最小验证，完成后停止扩展。

## 9. 最小验证，不新增测试框架

验证是开发过程中的核对，不是在项目中添加运行时检查或安全代码。

### 9.1 静态与接口核对

使用项目已有工具即可，至少确认：

- 修改文件能够通过 Python 语法编译。
- 奖励键都有同名 metric，所有新增缓存都随原 reset 路径清零。
- `force_scale` 不再作为闲置参数残留。
- `process_actions()`、固定轨迹函数、运动控制语句、阶段时长及控制标志含义没有改变。
- 新评分只更新到撤离之前，并在现有评估回调执行前写入。

可复用：

~~~bash
python -m compileall -q hammer_rl/mdp/actions.py hammer_rl/env_cfg.py
git diff --stat
git diff -- hammer_rl/mdp/actions.py hammer_rl/env_cfg.py hammer_rl/TRAINING_METRICS.md hammer_rl/compare_evaluations.py
~~~

不要安装新的格式化、lint 或测试依赖，不要批量格式化项目。

### 9.2 小样本运行

使用本机已有环境和现有缓存，从已有运行配置或项目实际文件确定数据路径，不把历史说明中的示例路径当成实际路径。

复用 `python -m hammer_rl.evaluate`：

- 使用 `--policy zero`。
- 少量样本，例如 `--episodes 4 --num-envs 2`。
- 优先覆盖 `flat` 和一个非平放类型，最多两组小样本即可。
- 复用现有 `--headless`；如确需查看已有回放，使用 `--viewer native`。
- 不生成新数据集，不重新调用 AnyGrasp，不启动长期 PPO 训练。

确认能够输出新奖励与指标，尤其是 `capture`、`lift_progress`、`impact_impulse`、`task_success`、`drop`；打印原始冲量及归一化分数，识别是否明显全部零或饱和，但本轮不据此自动调参。

如本机缺少仿真依赖、已有缓存或授权，仅完成可执行的验证并据实报告缺项，不新增 stub、mock、自动下载、兼容层或“临时通过”的绕过实现。

### 9.3 评分语义核对

核对以下业务关系即可，不为每条关系新建测试文件：

- 闭合接触形成分只有 0、0.5、1。
- 已持锤但抬升未到 15 cm 时，可以获得部分抬升进展分。
- 平放侧抓不再由于与锤头局部 `+Z` 不平行而必然零分。
- 挥击中脱手的接触不产生有效命中分。
- 先有效命中、后掉落：历史命中和原始输出可以保留，完整成功及最终保持/冲量评分为零。
- 最后一帧无双侧接触时，即使失联尚未累计 20 ms，也不能保留正的终局保持评分。
- 未命中时冲量和输出评分为零。
- 进入撤离后，任务分数冻结，撤离 touch 仅用于原有诊断。
- 新的奖励不改变原卸载与撤离时机。

## 10. 完成后的回复要求

完成修改后，用简明中文报告：

1. 实际修改了哪些文件，以及每个文件的必要改动。
2. 最终奖励公式、权重和新增参数。
3. 如何保持固定执行流程、控制标志和一次决策结构不变。
4. 实际运行了哪些最小验证，以及结果。
5. 未能运行的验证及具体环境原因。
6. 原始冲量的观察结果，以及首轮尺度是否存在明显饱和/全零现象。
7. 仍然存在的范围限制：动作空间和候选覆盖未改，奖励不能凭空产生候选之外的侧抓模式。

不要声称未运行的训练已经收敛，不以新的总奖励更大作为性能提升证据。

## 11. 设计依据，供理解而非扩展任务

以下论文只用于说明本次取舍，本地修改无需重新检索或移植其完整算法：

- [OmniReset：Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning](https://arxiv.org/html/2603.15789v3)：有效探索与状态覆盖很重要，不能只按奖励绝对数值判断引导强弱。
- [A Balanced Data Diet](https://arxiv.org/html/2610.12465v1)：任务成功与训练配置采样需要区分；本轮不实现 SGS。
- [Grasp to Act](https://arxiv.org/html/2602.20466v1)：动态载荷下的滑移和保持能力可以评价初始抓姿；本轮不增加在线手指 RL。
- [SimToolReal](https://arxiv.org/html/2602.16863v1)：接近/抓取进展、阶段事件和最终目标具有不同作用；本轮复用现有接触与抬升量，不扩展成全程臂手控制。
- [已审计 actions.py](https://github.com/dishenliu2004-hub/hammering-task/blob/f91fd72395c7438f4a490e75709aaaf3d9cbbccb/hammer_rl/mdp/actions.py)
- [已审计 env_cfg.py](https://github.com/dishenliu2004-hub/hammering-task/blob/f91fd72395c7438f4a490e75709aaaf3d9cbbccb/hammer_rl/env_cfg.py)

本轮不新增碰撞惩罚、不新增安全检查、不改动作空间、不换模型、不改采样、不做物理标定。后续是否实施这些独立实验由用户另行决定。

