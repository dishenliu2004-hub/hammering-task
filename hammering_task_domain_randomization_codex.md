# Hammering-Task：教师—学生统一域随机化最小修改操作文档（Codex 执行规范）

> 目标仓库：<https://github.com/dishenliu2004-hub/hammering-task.git>  
> 分析基准：`main`，Git tree `e3ce549e28ba608f1df8bc9da111174b7c43f5ec`（执行前以工作区现有代码为准，不要回退用户修改）  
> 运行栈：MuJoCo / MuJoCo Warp 3.11、mjlab 1.6、RSL-RL 5.4  
> 本文是 **实施要求**，不是已实施变更的报告。**只修改该仓库；不要安装/复制新的随机化框架。**

## 0. Codex 的执行指令（优先级最高）

1. 直接按本文改代码。**最小修改、无额外抽象、无新依赖、无新随机化管理器、无防御性代码、无额外安全检查。** 不添加兼容旧缓存的 `try/except`、`hasattr`、`get(..., default)`、形状/范围校验、回退逻辑、重复的随机化工具、额外命令行参数或配置文件。
2. 继续使用已有的 `mujoco.MjSpec`、`mjlab.managers.event_manager.EventTermCfg`、`requires_model_fields`、PyTorch Tensor 和 `generation.enclosure`。不要迁移到 Isaac Lab，不要修改任何第三方包、site-packages、RSL-RL 或 mjlab 源码。
3. 实现三类变化：**原有质量/惯量/指面摩擦随机化**、**镜像驱动关节限位随机化**、**锤柄 capsule 半径及长度的离线几何预随机化**。
4. **教师 PPO、蒸馏、学生 PPO 使用同一物理随机化分布；教师不添加无用的深度噪声，蒸馏/学生保留原有深度噪声。**
5. **锤柄几何必须在离线生成深度和 AnyGrasp 候选之前确定**；把该场景对应的锤柄几何与几何包络一并存入 reset bank，训练/评估时从同一条 bank 恢复。**禁止**单纯在线改变锤柄几何，却继续使用与该几何不符的旧深度图、旧候选和名义包络。
6. 不改变动作、奖励权重、SGS 算法、观测维度、网络、蒸馏器、接触传感器命名、锤头几何、机器人机构及 XML 原始名义参数。
7. 不添加新的测试程序；按此前项目约束，**不执行代码测试或训练**。完成后只给出变更文件清单、修改摘要、新缓存生成命令、训练命令和旧缓存/权重的影响；不声称已经验证训练效果。

## 1. 为什么要采用“离线形态 + 在线物理”而非全部在线随机化

仓库的数据链路为：

```
prepare.py: 编译 MuJoCo 场景
  → 稳定位姿沉降筛选
  → 相机采集 depth
  → AnyGrasp 检测并生成 local_pos/local_quat、open_ctrl 等候选
  → 保存 reset_bank_*.npz

resets.py: reset_scene 从 bank 恢复场景、depth、候选
  → actions.py 执行抓取/敲击
  → generation/enclosure.py 利用编译时生成的凸包评价抓持包络
```

若仅在训练 reset 修改 `hammer_handle` 的 `geom_size`，那么 `(1,120,160)` 的深度仍来自名义锤柄，抓取候选也来自旧图像；而 `build_enclosure_geometry()` 的 capsule 凸包仍是名义几何。这会让学生视觉、真实接触和抓持奖励相互不一致。因此几何变化在 `prepare.py` 生成 **每场景固定的随机形态**；训练中保持该形态与 bank 一一对应。质量、摩擦、驱动关节限位则仍 **每次 reset 独立随机**。

此方案不追求每次 reset 为相同场景重新生成相机图像。它利用现有离线稳定场景生成管线，避免增添运行时相机/AnyGrasp 推理服务，也与逐场景 SGS 兼容。

## 2. 优先复用的成熟开源实现（请阅读对应代码，不要照搬框架）

| 来源 | 直接参考位置 | 本项目的使用方式 |
|---|---|---|
| [mjlab v1.6.0](https://github.com/mujocolab/mjlab/tree/v1.6.0) | [`dr/geom.py`](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/envs/mdp/dr/geom.py) 中 `geom_size` / `_recompute_geom_bounds` | 按其 capsule 公式更新 `geom_size`、`geom_rbound`、`geom_aabb`，复用原生 EventManager |
| mjlab v1.6.0 | [`dr/joint.py`](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/envs/mdp/dr/joint.py) 中 `joint_limits` | 直接按原生字段 `jnt_range` 写入；不自造关节控制器 |
| mjlab v1.6.0 | [`event_manager.py`](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/managers/event_manager.py) | 继续使用仓库已有 `@requires_model_fields` 和 `mode="reset"` |
| [Phys2Real，ICRA 2026](https://github.com/phys2real/phys2real) | [`push_env_cfg.py`](https://github.com/phys2real/isaaclab-phys2real/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/push/push_env_cfg.py) | 借鉴锤子质量、摩擦、质心等物理量选择，不移植 Isaac Lab/在线适应网络 |
| [GoFlow，ICML 2025](https://github.com/aidan-curtis/goflow) | 开源域随机化算法 | 理论参考；此次不引入 Flow 分布学习器，不增加第二套 SGS |

**关键架构限制：** 当前项目的 `SceneCfg` 使用 `spec_fn=_attach_scene` 挂入完整原始场景，未在 `SceneCfg.entities` 注册可被 `SceneEntityCfg("robot")` 定位的实体。因此不能机械地在这里直接调用 `dr.geom_size(..., asset_cfg=SceneEntityCfg("robot", ...))` 或 `dr.joint_limits(...)`；这不是一个有效的实体绑定。为保持现有完整场景边界，**复用 mjlab 原生 EventManager 和它公开的模型字段更新约定，仅在现有事件函数中做极少量字段写入；不复制私有 `_core.py`，不伪造 Entity，不重构 SceneCfg。**

文档依据：<https://mujocolab.github.io/mjlab/v1.6.0/source/randomization.html>。其中 capsule 的 `geom_rbound = radius + half_length`，`geom_aabb` 半尺寸为 `(radius, radius, radius + half_length)`，且 `jnt_range` 可以在 MuJoCo Warp 中按世界修改。

## 3. 随机化范围（本次固定为温和初值）

| 参数 | 单位 / 分布 | 默认及作用域 |
|---|---|---|
| 锤子质量与惯量倍率 | `U(0.90, 1.10)` | **原有** `HammerEnvCfg.mass_range`，每个 reset、每个环境独立；质量与惯量同倍率 |
| 指尖摩擦倍率 | `U(0.80, 1.20)` | **原有** `HammerEnvCfg.friction_range`，同一环境的各指尖共用抽样倍率 |
| 驱动关节开口端扩展量 `delta_open` | `U(0, 0.02)` rad | 每个 reset、每个环境独立；镜像两侧共用 |
| 驱动关节闭合端收紧量 `delta_close` | `U(0, 0.06)` rad | 每个 reset、每个环境独立；镜像两侧共用 |
| 锤柄 capsule 半径 | 名义值 `0.025 m × U(0.95, 1.05)` | **离线**每个稳定场景采样一次，训练不再重采样 |
| 锤柄 capsule 半圆柱长度 | 名义值 `0.05 m × U(0.95, 1.05)` | **离线**每个稳定场景采样一次，训练不再重采样 |
| 深度噪声 | 使用已有 `DepthNoise` | 仅蒸馏与学生训练；不可替代几何一致的离线深度 |

说明：MuJoCo capsule 两参数分别是 **半径** 与 **圆柱段半长**，其总端到端长度为 `2 × (radius + half_length)`，不是简单的 `2 × half_length`。当前形态随机化只改变这两个几何参数，不改变 capsule 形状类别，不生成椭圆/锥形/网格或新的机械实体。参数均为工程初值，不宣称已经实验调优。

## 4. 逐文件改动要求

### 4.1 `hammer_rl/env_cfg.py`：三个训练阶段统一物理随机化

在 `make_env_cfg()` 保留原事件结构，只调整：

```python
# 原代码：randomize = stage in ("distill", "student")
randomize = True
```

仍采用现有：

```python
events["physics"] = EventTermCfg(func=resets.randomize_physics, mode="reset")
events["scene"] = EventTermCfg(func=resets.reset_scene, mode="reset")
```

并将配置中的深度噪声从原先与 `randomize` 绑定改为：

```python
randomize_physics=randomize,
visual_noise=stage in ("distill", "student"),
```

保留现有事件字典插入顺序（`physics` 在 `scene` 之前）。本次不增加新 EventTerm。`evaluate.py` 仍沿用其显式关闭物理随机化的现有逻辑；**几何由指定 bank 决定，不受该开关影响**。

### 4.2 `hammer_rl/prepare.py`：在编译前随机化真实锤柄几何

修改 `create_scene(rng, hammer_head_extra_mass=...)`：在已有 `spec = mujoco.MjSpec.from_file(...)`、`add_head_mass(...)` 之后、`spec.compile()` 之前，定位 `spec.geom("hammer_handle")`，用 `rng` 分别为 `size[0]` 与 `size[1]` 乘上 `U(0.95,1.05)` 缩放系数。**每次 `create_scene` 只采一次，在后面的 `while True` 沉降重采样中保持该形态不变。**

伪代码（按当前 MjSpec API 合并到原函数，不抽取新模块）：

```python
handle = spec.geom("hammer_handle")
handle.size[0] *= rng.uniform(0.95, 1.05)
handle.size[1] *= rng.uniform(0.95, 1.05)
model = spec.compile()
```

其余沉降与稳定性筛选不变。项目原有 `create_scene` 已使用编译后的 capsule `geom.size[:2]` 来计算支撑高度，因此不用再写一套几何落点算法。原有 `camera.capture`、`generate_grasps` 自然基于形态变化后的实际模型运行。

在 `prepare_scene(...)` 构建 `sample` 时，新增两个 **scene 级**字段：

```python
sample["handle_size"] = model.geom("hammer_handle").size[:2].copy()
```

并复用下节 `generation.enclosure.build_enclosure_geometry(model, "cpu")` 已有的凸包构建结果，从中取出当前真实 `hammer_handle` 的 convex planes，保存为 `sample["handle_convex_planes"]`（NumPy 数组，形状 `(P,4)`）。复用原函数而不是自行重写 `trimesh.creation.capsule`、ConvexHull、坐标变换。为取出正确的 capsule 子件，下节在返回结构中明确加入 `handle_part_index`；不要依赖“锤柄必定是第一个部件”的隐含假设。

沿用既有 `np.stack` / `np.savez_compressed`，把上述字段放入新的 `reset_bank_*.npz`；**不改变** `offsets`、`local_pos`、`local_quat`、`widths`、`open_ctrl`、`depth`、`split`、现有状态字段与 AnyGrasp 管线。

**必须重新生成 reset bank**，不能把新代码直接配合旧 bank 使用；不得添加旧 bank 兼容分支或自动修复代码。

### 4.3 `generation/enclosure.py`：复用已有凸包计算，但支持按场景选择真实锤柄包络

现有 `build_enclosure_geometry(model, device)` 枚举 `hammer_Object` 的接触 geom，计算每个 geom 的局部凸包。保持这一算法和其他部件不变：

1. 在枚举过程中记录 `geom_id == model.geom("hammer_handle").id` 对应的 `parts` 下标 `handle_part_index`。
2. 在返回的 `SimpleNamespace` 中增加 `handle_part_index`；保持原 `convex_parts` 字段与既有属性。
3. 使 `_segment_union_inside_length(...)` 所接收的 `equations` 为统一批量形状 `(num_envs, P, 4)`，只把当前静态平面方程计算改成带 batch 维度的向量化点积：

```python
values = (origin[:, :, None, :] * equations[:, None, :, :3]).sum(-1) + equations[:, None, :, 3]
slopes = (direction[:, :, None, :] * equations[:, None, :, :3]).sum(-1)
```

其中 `origin` / `direction` 沿用现有 `(B,2,3)`；最终中间结果应保持 `(B,2,P)`，后续截断、并集与评分公式全部不变。不得增加逐环境 Python 循环。

4. 将 `compute_enclosure_score` 增加必需参数 `handle_planes`，形状为 `(B,P,4)`；它在计算前将原 `geometry.convex_parts` 中除 `handle_part_index` 外的静态方程通过 `unsqueeze(0).expand(B,-1,-1)` 批量展开，再替换锤柄所在部件为传入的 `handle_planes`。传入统一批量凸包列表给 `_segment_union_inside_length`。
5. 保留 `ENCLOSURE_BALANCE`、`ENCLOSURE_SHAPE_EXPONENT`、`penetration <= 0.003`、contact、角度、奖励等所有原判据，不重新设计奖励。

此项是必要的一致性修正：若几何发生改变，不能继续把名义半径的 convex hull 当作真实锤柄来评估抓持包络。凸包算法仍直接复用仓库已有开源实现；仅增加 scene 级缓存与 GPU 批量索引。

### 4.4 `hammer_rl/mdp/actions.py`：抓持包络评分读取当前 scene 的缓存凸包

仅修改 `compute_enclosure_score(...)` 的调用参数，在原有 `data`、`self.enclosure_geometry`、`self.palm_frames >= 5` 之后，传入：

```python
self.task.bank["handle_convex_planes"][self.task.scene_id]
```

维度为 `(num_envs,P,4)`；每个环境通过原始 `scene_id` 取得离线几何包络。不要改变动作时间线、接触槽、冲量收集和奖励值计算，不要在物理子步中运行 CPU `trimesh`。

### 4.5 `hammer_rl/mdp/resets.py`：恢复几何 + 新增关节限位随机化

**A. `initialize_task(env)` 只缓存必要的原始索引：**

- `handle_geom_id = model.geom("hammer_handle").id`
- `narrow_driver_id = model.joint("gripper_hand_narrow1_joint").id`
- `wide_driver_id = model.joint("gripper_hand_wide1_joint").id`
- 从 `model.jnt_range.copy()` 保存一份名义 `nominal_joint_range`（Torch，设备与模型已有 nominal 缓存一致）。

可直接将这几个字段加到已有 `SimpleNamespace`；不新建数据类。

**B. 扩展现有 `randomize_physics(env, env_ids)`：**

在现有装饰器的模型字段中补充 `"jnt_range"`，继续使用已有 `recompute=RecomputeLevel.set_const`。保留质量、惯量、摩擦原代码，随后写入两侧主驱动关节限位。按原始名义范围计算，绝不在上一次随机结果上累加：

```python
# 基准范围（来自 XML）：narrow1=[-0.75,0.05]，wide1=[-0.05,0.75]
# 实现时从 task.nominal_joint_range[...] 读取基准值，不再重复硬编码基准端点。
open_delta  = torch.empty(count, device=env.device).uniform_(0.0, 0.02)
close_delta = torch.empty(count, device=env.device).uniform_(0.0, 0.06)

narrow_base = task.nominal_joint_range[task.narrow_driver_id]
wide_base = task.nominal_joint_range[task.wide_driver_id]
model.jnt_range[env_ids, task.narrow_driver_id, 0] = narrow_base[0] - open_delta
model.jnt_range[env_ids, task.narrow_driver_id, 1] = narrow_base[1] - close_delta
model.jnt_range[env_ids, task.wide_driver_id, 0] = wide_base[0] + close_delta
model.jnt_range[env_ids, task.wide_driver_id, 1] = wide_base[1] + open_delta
```

**只随机化这个驱动镜像关节对。** `gripper` 的近端驱动通过 `symmetric_driver` 等式和 `driver_split` tendon 耦合；`narrow2/wide2` 被动弹簧、`narrow3/wide3` 末端硬限位、闭环 `connect`、loop joints、actuator `ctrlrange` / `forcerange` 均不改。尤其当前 XML 的 `closed` keyframe 中 distal 角度与其名义限位本来就不完全一致，盲目统一缩放所有关节上下界会改变闭环求解与开爪初态。镜像采样是必须保持的机构约束，不是新增防御性逻辑。

**C. 扩展现有 `reset_scene(env, env_ids)` 恢复每场景真实几何：**

将该函数装饰器中的 `requires_model_fields(...)` 由 `("body_pos", "body_quat")` 扩展为再包含 `"geom_size"`、`"geom_rbound"`、`"geom_aabb"`，保留原 `recompute=RecomputeLevel.set_const_0`。在确定 `scene` 后、绑定候选前，按下列字段恢复：

```python
sizes = bank["handle_size"][scene]   # (count, 2)
radius, half_length = sizes[:, 0], sizes[:, 1]
h = task.handle_geom_id
model.geom_size[env_ids, h, :2] = sizes
model.geom_rbound[env_ids, h] = radius + half_length
model.geom_aabb[env_ids, h, 1, 0] = radius
model.geom_aabb[env_ids, h, 1, 1] = radius
model.geom_aabb[env_ids, h, 1, 2] = radius + half_length
```

`geom_aabb[...,0,:]` 的 capsule 局部包围盒中心保持编译默认值，不需重写。上述是 mjlab v1.6.0 `dr.geom_size` 所使用的 **capsule 专用公式**；只对现有 `hammer_handle` 使用，不能用于 mesh 型 `hammer_head`。将此写在 `reset_scene` 是因为 geometry **必须由已选 scene_id 决定**，并非新的独立随机事件。无论采样来自 SGS 还是评估时固定 `task.selection`，都取同一条 bank 的几何。

**D. 保留原始恢复及观测顺序：** 原 `STATE_FIELDS` 写回、`data.time=0`、`bank["depth"][scene]`、学生阶段 `DepthNoise`、`bind_candidates` 全部保留。教师的 `privileged` 仍为 87 维，学生 `public_seed` 仍为 50 维，深度仍为 `(1,120,160)`；不增加输入特征。形态变化由离线观测和真实物理接触体现。

### 4.6 `hammer_rl/README.md`：只补充必要的使用说明

补充：新 bank 含 `handle_size`、`handle_convex_planes`，几何预随机化在 prepare 期确定；教师也使用在线物理与关节限位随机化；蒸馏/学生保留深度噪声；老 bank 必须重建；新训练/校准/评估需指向同一版 bank。不要重写整份 README。

## 5. 处理原则与边界

### 必须保持

- **数据一致**：同场景 `handle_size`、`handle_convex_planes`、沉降后的真实状态、深度图、抓取候选相互匹配。
- **评估有意义**：`evaluate.py` 默认关闭 reset 物理随机化，但依照评估 bank 恢复每个场景的离线形态；因而“名义物理参数评估”不等于“名义锤柄形态评估”。
- **SGS 不变**：仍以 `scene_id` 统计最近成功率，场景内部是固定形态，episode 内不修改几何。不要把形态比例或 joint range 加进 SGS 参数，也不要修改 teacher checkpoint 的课程状态结构。
- **训练一致**：Teacher、Distill、Student 共享固定随机化分布和同一个新 bank；不要求三个阶段使用完全相同随机数。初始权重结构不变，但旧教师权重并不等于已经学到了新随机化下的鲁棒性。
- **动作不变**：不更改 `open_ctrl` 离线标定、`symmetric_driver`、夹爪闭合时序、`attempt_substeps`、击打指标及已有奖励系数。限位变化由现有物理引擎执行。

### 明确禁止

- 不添加任何安全检查、参数范围检查、尺寸检查、格式兼容分支、`try/except` 回退或静默跳过。
- 不修改 `assets/scenes/*.xml`、`assets/grippers/omnipicker/ominipicker.xml`、任何 STL/mesh、AnyGrasp SDK、训练算法配置、网络维度或 `train.py`。
- 不添加 `dr.py`、`domain_randomization.py`、新随机化 class、新注册表、新配置文件或第三方安装指令。
- 不引入在线渲染、在线 AnyGrasp、随机外力、重新求解 IK、改变锤头形状、随机质心或再训练检测器。
- 不添加自动异常处理、自动回滚或额外测试文件，不运行测试或训练。

## 6. 生成缓存与训练指令（供用户在代码修改后自行执行，不由 Codex 运行）

在项目根目录、现有 `hammer` 环境运行：

```bash
# 重新生成带锤柄形态变化、对应深度/抓姿及 convex planes 的数据；勿覆盖旧数据
python -m hammer_rl.prepare \
  --output weight/hammer_rl/reset_bank_dr_256.npz \
  --num-scenes 256 \
  --candidates-per-scene 32 \
  --seed 42 \
  --workers 4

# 教师 PPO：启用物理 + 驱动限位随机化，使用新形态场景
python -m hammer_rl.train \
  --stage teacher \
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
  --num-envs 256 --max-iterations 3000 --device cuda:0 --seed 42 \
  --log-dir weight/hammer_rl/runs/teacher/domain_randomization_01

# 蒸馏：将 teacher checkpoint 路径替换为实际产出的权重
python -m hammer_rl.train \
  --stage distill \
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
  --teacher-checkpoint <teacher_checkpoint.pt> \
  --num-envs 128 --max-iterations 1000 --device cuda:0 --seed 42 \
  --log-dir weight/hammer_rl/runs/distill/domain_randomization_01

# 学生 PPO：将蒸馏权重路径替换为实际产出的权重
python -m hammer_rl.train \
  --stage student \
  --data-file weight/hammer_rl/reset_bank_dr_256.npz \
  --init-from <distill_checkpoint.pt> \
  --num-envs 128 --max-iterations 3000 --device cuda:0 --seed 42 \
  --log-dir weight/hammer_rl/runs/student/domain_randomization_01
```

上面尖括号占位符必须由用户替换成真实路径；不在程序中实现自动搜索。若训练采用了 `--hammer-head-extra-mass`，则数据准备及训练仍必须按项目原约定使用相同质量参数。

## 7. 完成定义（静态交付说明，不要求执行测试）

Codex 完成后，应在最终说明中逐项确认其代码修改能够做到：

1. `prepare.py` 为每条场景在编译前采样一次锤柄 capsule 半径、长度，稳定沉降/深度采集/AnyGrasp 使用相同的编译模型，bank 新增 `handle_size`、`handle_convex_planes`。
2. `reset_scene` 依据 scene_id 恢复 capsule size，同时同步 `geom_rbound`、`geom_aabb`，不会把几何留在上一个 scene 的值。
3. 原 `randomize_physics` 的质量、惯量、指面摩擦保持；新增镜像 proximal driver 两侧联动的限位随机化，并始终从名义限位计算。
4. `generation.enclosure` 的凸包奖励按 scene 的真实 hammer_handle 凸包计算，而非固定名义锤柄；评分数学和权重不变。
5. teacher、distill、student 均启用物理/限位随机化；teacher 没有额外深度噪声，distill/student 继续使用已有深度噪声。
6. SGS、模型输入维度、动作与奖励接口、训练/蒸馏 runner、权重格式、评估入口保持不变。
7. 未新增包、测试、检查或防御性代码；未运行测试、训练或评估。

## 8. 权重、评估与客观限制

- **旧缓存不能直接用于新代码**：因为 `handle_size` / `handle_convex_planes` 为新的必需字段；本次禁止写兼容逻辑。
- **旧权重张量格式本身不因本方案发生变化**；但新形态、新摩擦和关节限位分布属于新的训练域。建议用新 bank 开始全新 teacher → distill → student 训练流水线；不把旧权重效果当作新随机化效果。
- **不能声称对任意锤柄形状泛化**：此次仅覆盖 capsule 的连续半径/长度变化，不覆盖锥形、扁平、椭圆截面或新网格。对这些变化需要几何资产或网格变体，超出本次最小修改范围。
- **不能声称夹爪所有关节限位都已随机化**：此次仅随机化两侧主驱动 hinge 限位；被动关节、闭环约束和执行器限制保持名义配置。
- **不能声称训练精度已提高**：本方案只是更一致的训练域随机化实现；泛化效果须由用户之后自行运行训练及 held-out 评估确定。

---

**最终交付给用户的内容：** 修改文件清单、每个文件的简要代码变更、所借鉴的 mjlab v1.6.0 源码位置、新旧 bank 兼容性说明、上述缓存生成和三阶段训练命令。不要执行测试，也不要报告未实际取得的实验数据。
