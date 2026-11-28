# 根据当前 Teacher 的物理反馈修正锤击奖励

任务是在一次抓姿决策后沿固定轨迹完成三次锤击和卸载。结合原始人手照片、`mapping_latent` 中的 `hand_grasp` 关键点和 `contact_prior` 的接触区域、热点及角度覆盖，以及 `gripper` 的欠驱动二指结构、接触部位、弹簧和耦合关系，判断当前抓取如何完成这项任务。参考 `current_recipe` 中正在使用的奖励、机器人适配先验 `mapping_prior` 和上一轮的修改理由 `rationale`。

先读 `metrics_summary` 的均值、中位数和 p90。再用 `frame_index` 的 `episode` 在 `episode_metrics` 各数组中定位同一回合，结合试次结果与图像诊断。图像依照索引顺序附在人手照片之后。`closure_end` 展示闭合后的接触、包络和承托，`lift_end` 展示抬升后的接触位置、方向和稳定性，`swing_start` 展示挥击前持有和相对位姿，`swing_end` 展示挥击后的抓持状态。判断应同时基于数值和多个阶段的图像。

- 用 `capture`、`wrap`、`seat_closure`、`seat_lift`、`seat_before_swing` 判断接触、包络和承托。若接触位置符合先验但包络不足，考虑调整 `wrap` 的权重或公式。若包络较好但承托不足，结合三个阶段的承托值和图像调整现有抓取奖励。
- 用 `grasp_u_rl`、`grasp_axis_cos2`、`prior_region` 和 `prior_axis` 判断抓取区域与闭合方向是否适合夹爪和任务。若有效区域偏离人手先验，调整 recipe 的 `u_mean_RL`、正的 `u_sigma_RL` 或工具坐标系中的单位向量 `closing_axis_O`。原始 `contact_prior` 保持固定。若先验匹配较好但完整敲击失败，检查几何辅助奖励是否占比过高。
- 用 `hold_fraction`、`stability`、`translation_drift`、`rotation_drift` 和 `drop` 判断持有与滑移。挥击前漂移较大时，可调整 `stability` 或使用 `hold_drift`。
- `task_success` 表示三次有效命中且卸载前仍持有。`valid_strike` 表示至少一次有效命中。结合 `hit_progress` 和各次 `strike_force_*` 判断失败发生在哪次击打。若持有稳定但敲击力不足，可调整 `force_mean_min` 的权重或参数。若击打成功后难以恢复持有，考虑调整 `recovery_hold`。

输出一份完整的下一轮配方，包含 `terms`、`mapping_prior` 和不超过 300 字符的中文 `rationale`。理由只概括当前反馈中的主要现象和关键调整依据，不逐项展开或复述输入。可以增删已有奖励项、调整权重、切换已有公式、调整公式参数或调整机器人适配先验。若当前反馈没有明显问题，输出一份等价配方。

每个奖励项使用 `id`、`weight`、`formula`、`params` 和 `gate`。从 `available_terms` 选择 `id`，每个 `id` 最多一次。从 `available_formulas` 选择公式并按说明填写有序参数数组。`gate` 只能取 `none`、`lift_success` 或 `grasp_before_swing`。`prior_gaussian` 和 `prior_axis_power` 必须使用 `lift_success`，敲击力奖励使用 `grasp_before_swing`。以 `task_success` 为主要终局目标，`drop` 使用负权重，仿生先验服务于完整敲击任务。只输出一份结构化配方，不输出候选列表、新指标、新算子或 Python 代码。
