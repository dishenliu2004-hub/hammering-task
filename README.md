# Hammering Task

基于 MuJoCo 与 Mjlab 的锤击任务项目。

## 本地依赖

weight/ 未上传到仓库。
使用相关功能前，需要自行配置预训练权重、运行库及授权文件。

## 强化学习流程

训练流程：**离线准备数据 → 教师 PPO → 教师向学生蒸馏 → 学生 PPO 微调 → 固定 actor 校准评分器**。

- 离线准备 256 个稳定 A 场景和 40 个独立悬空 B 目标，缓存深度及 AnyGrasp 候选；actor 学习 A 抓取 6 维残差与 B 最终调姿 3 维动作。
- 教师 actor 和 critic 使用特权观测；SGS 按每个训练场景最近 100 次完整任务成功率动态分配采样概率，优先采样成功率接近 0.5 的场景。
- 蒸馏及学生 PPO 使用深度图和公开候选信息，并开启物理随机化和深度噪声；学生 actor 和 critic 使用各自的 CNN。
- 部署前固定 actor，以执行后的总奖励校准 critic；推理时由 critic 选择候选，再由 actor 微调位姿。

具体命令及指标见[强化学习操作指南](hammer_rl/README.md)和[训练日志指标说明](hammer_rl/TRAINING_METRICS.md)。
