# 为训练场景库设计锤击奖励

任务是在一次抓姿决策后，沿固定轨迹执行三次锤击和卸载。为训练场景库的所有场景生成同一份 V2.0 奖励配方。结合动态敲击任务描述、真实的人手抓锤照片和 `mapping_latent` 设计奖励。用照片、`hand_grasp` 关键点和 `contact_prior` 的沿柄接触区域、热点及角度覆盖理解掌侧承托和手指包覆，再结合 `mapped_grasp` 的机器人适配先验和 `gripper` 的欠驱动二指结构、接触部位、弹簧及耦合选择几何辅助项。`mapped_grasp.pose_candidates` 给出工具坐标系下的两个开爪参考位姿及各自的 `keypoint_correspondences`。人手关键点 8 为食指尖，4 为拇指尖。包覆和持有效果由闭合、抬升及后续阶段的仿真指标衡量。

## 设计离线奖励

- 以 `task_success` 为主要正向终局目标，优先保证三次有效命中和卸载前持续持有。用负权重的 `drop` 惩罚脱手。按任务描述提高击打力度时，使用 `force` 的现有尺度或 `force_mean_min`，以 `grasp_before_swing` 门控。选择 `force` 时省略重复计分的 `impact_impulse`。
- 用 `capture`、`lift_progress` 或 `lift_success` 提供闭合和抬升阶段的反馈。用正权重的 `wrap` 鼓励闭合时形成包络，并在后续阶段维持有效顶靠。`wrap` 已按有效抬升等条件门控。结合 `stability`、`hold_fraction` 或 `recovery_hold` 鼓励抓持持续。`capture` 只统计闭合时有效接触的侧数，因此只作为早期辅助信号。
- `prior_region`、`prior_axis` 和 `orientation` 作为几何辅助项。`distance` 只反映夹持中心沿柄轴到锤头的距离，适合弱化使用。若靠近自由端与包覆或持有目标冲突，可省略该项。
- 每项给出 `id`、`weight`、`formula`、`params` 和 `gate`。从 `available_terms` 选择 `id`，每个 `id` 最多一次。从 `available_formulas` 选择已有公式，按说明填写有序参数数组。`gate` 只能取 `none`、`lift_success` 或 `grasp_before_swing`，`prior_gaussian` 和 `prior_axis_power` 必须使用 `lift_success`。只输出结构化配方，不生成 Python 代码或新指标。

输出 `terms`、简短的中文 `rationale` 和 `preview_candidate`。根据照片和 latent 中的关键点对应，为对照图选择一个候选，输出它在 `pose_candidates` 中从 0 开始的下标。理由说明奖励的任务优先级，以及照片和 latent 中的抓持关系如何影响算子选择。
