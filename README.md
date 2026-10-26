# Hammering Task

基于 MuJoCo 的锤击任务项目。

## 目录结构

- assets/：场景、相机和夹爪资源
- generation/：任务生成相关代码
- hammer_rl/：强化学习相关代码
- visualization/：可视化工具

## 本地依赖

weight/ 未上传到仓库。
使用相关功能前，需要自行配置预训练权重、运行库及授权文件。

## 强化学习流程

- 进行第一阶段的教师模型训练，在该阶段提供全部的特权信息内容，同时在该阶段也进行课程学习，逐渐调整其中的数据（锤子的放置姿态），该阶段是基于anygrasp权重生成的抓取位姿来学习抓取位姿的生成
- 进行第二阶段的学生模型训练，在该阶段用实际场景信息替换第一阶段的特权信息内容，同时引入预随机化进行训练

## 强化学习考虑的奖励
- 夹爪的包络程度
- 夹爪的抓取相对于锤头的敲击面的方位
- 夹爪的抓取相对于锤头的距离
- 夹爪抓取并被提升后的与锤子的相对位姿，和敲击前一帧与锤子相对位姿的区别，区别越小越好
- 最后敲击的力的大小

## 一次抓姿决策的最小实现

策略只在抓前调用一次，输出 `[N,6]`。以**转换后的 seed mount** 为原点：
`dp=0.02*tanh(a[:3])` 米，`dtheta=(pi/16)*tanh(a[3:])` 弧度，
`p=p_seed+R_seed*dp`，`R=R_seed*Exp(dtheta)`。四元数统一为 wxyz。
零策略是六维全零，执行原候选；pregrasp 是最终 mount 沿接近轴后退 0.10 m。
开度使用原 AnyGrasp 标定值，闭合指令固定为 1.0。

一次原生 `ManagerBasedRlEnv.step()` 完成整个尝试。预抓/接近/闭合/提升为
1.0/0.6/0.8/0.8 s，提升 0.25 m；搬运/调姿/下降为 2/4/4/2 s，settle 1 s。
挥击沿 `generation/knock_path.py` 的抓取中心抛物线，60°、竖直 0.30 m、水平
0.40 m、切向加速度 10 m/s²、carry_z=0.75 m。保留原 DT=0.02 s 外层采样及
物理子步插值、释放 weld、保速/重力补偿、实际基座对齐后恢复 weld、外法向撤离 3 cm。
卸载最多评价 1.0 s；有命中时沿用连续三子步卸载且精确接触对消失的结束规则。
撤离结束的环境保持最终目标，评价量已锁存。该有限时窗是任务定义，不增加异常中止。
物理步长取总场景编译值（本机 0.002 s）。`auto_reset=True`、有限 horizon=1、
正常 terminal、reward 不乘 dt、rollout 长度=1；策略不控制任何后续轨迹或时序。

初始实现约定，**未经调参**：

| reward | 定义与有效时点 | 权重 |
|---|---|---:|
| wrap | 抬升结束，同一帧两指六个碰撞 geom 对真实柄的接触；每 geom 最强 2 个 slot。转换到柄 geom 局部 XY，覆盖角 C=2π−最大环形角间隙；双侧有效且有效抬升时 clamp((C−π)/π,0,1) | 1 |
| orientation | 有效抬升时，真实两指 contact_tip 连线 c 与锤体局部 +Z 敲击面法向 n 的 abs(dot(c,n)) | 1 |
| distance | 有效抬升时 clamp(d/0.315,0,1)，d 为两指 tip 中点到锤体局部 (-0.24,0,0) 的距离，偏向较长力臂 | 1 |
| stability | 比较抬升后 A 与挥击前 B 的真实 gripper_base_link→hammer_Object 变换；exp(−(平移误差/0.01m)²−(SO(3)角误差/5°)²)，A/B 均有效抓持才计分 | 2 |
| force | 首次挥击及卸载期间，hammer_head→目标端盖精确接触对的世界系 netforce，沿端面压缩方向的峰值 F；F/(F+100N) | 4 |

有效抬升要求锤体真实上升至少 0.15 m 且两侧手指与柄接触。挥击前锁存抓持有效性，
不会因卸载后松脱抹掉先前的有效冲击。力峰值在每个 physics substep 更新小型 buffer；
撤离峰值与原 touch（区域内所有接触的非负法向力之和）单列，不进入 force reward。
距离参考 0.315 m 来自头中心到柄 capsule 远端的几何距离；优化方向、上述尺度、权重、
动作尺度、卸载时窗均为本次选择，不代表原 README 已规定或已经最优。

## 数据与观测

`prepare` 复用原四类摆放公式及 mjlab 的 2 s 落稳，保留每条场景自身的模型
body_pos/body_quat 参考和完整动力学初态，避免把六个 slide/hinge 换成 freejoint 或
统一 Euler 参考。初始深度、相机标定、摆放难度及候选通过场景索引绑定；每类最后
20% 场景（至少 1 个）为 eval，同一场景的所有候选属于同一 split。
每场景需有实际 SDK 候选；不按抓取/敲击结果筛选，不补齐候选数。小规模验证建议
至少 8 场景以覆盖每类的 train/eval。缓存只在环境构造时加载到 GPU。

候选来自固定相机，工作区固定为世界坐标 `[0.14,-0.16,0.11]` 到
`[0.46,0.41,0.72]` m，不用真实锤子位姿或颜色 mask 裁剪；保留 SDK 原 nms/sort，
最多 32 个。复用原 subprocess、开度标定及 mount 转换（包括 depth、TCP 偏移、额外
30 mm 深入量），并保存物体局部候选。默认演示路径保持原行为。

在线 reset 按 env_ids 独立采样并恢复，使用缓存的真实初态与局部候选复合 seed。
夹爪实际自由基座和 mocap 同时置于 seed pregrasp，保持开爪，未抓持锤子。
无 reset 内物理推进，无锤子与夹爪 weld。
这是参考 [UWLab ObjectRestingEEGrasped 的离线采样机制](https://github.com/UW-Lab/UWLab/blob/2b40afb9693d46bc1b1a91c7ba95e03c04daa60d/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py)
独立实现的 mjlab near-object 适配，未移植其 IK、Isaac 运行时或已抓持手指状态。

三组观测冻结在本次决策前，后续 get_observations 不重新采噪声：

- `public_seed [N,50]`：世界系 seed 位姿、SDK width/标定开度、已知 pregrasp
  位姿、8 个手指关节位置及速度、光学相机位姿/内参、标定目标位姿。
- `depth [N,1,120,160]`：原固定相机 1280×960 深度每 8 像素取样，单位 m，
  无效值为 0；CNN 输入保留图像维度，不 flatten 成数值观测。
- `privileged [N,87]`：public_seed 加真实锤体位姿及其在 seed 中的相对位姿、
  锤头中心、支架位姿/尺寸、实际锤体质量/惯量和六个指面摩擦系数。

教师 actor/critic 都用 privileged。学生 CNN actor 只用 depth/public_seed，
critic 用 privileged；A/B 快照与冲击结果均不进入初始观测。
深度是**夹爪靠近前的冻结抓前帧**，不是 near-object reset 后的实时遮挡帧。
未添加实时 GPU 相机。改变初始几何、摆放或相机时需重新 prepare。

教师课程只改变摆放样本范围：默认前 600 次 update 选 train flat 最小偏转四分位，
600～1800 次加入 flat 和逐渐增大倾角的两类 lean，此后四类全覆盖。
里程碑保存在 run_config.json，resume 恢复 common_step_counter，追加迭代不重定义课程。
蒸馏/学生使用全部 train 样本，指面三维摩擦按名义值乘 0.8～1.2，锤子质量和惯量
同乘 0.9～1.1，每回合从名义值采样；几何和质心不变。
深度加噪复用 GPU `DepthNoise`，焦距按 1/8 下采样，baseline=0.05 m、范围
0.3～3 m、filter_size=3，沿用 adapter 约定，未经实机标定。

教师使用内置 MLP(256,128,64)/ELU/观测归一化。学生使用内置 CNNModel
(卷积通道16/32/32、核5/3/3、步长4/2/2) 加相同 MLP。
蒸馏直接用 RSL-RL DistillationRunner/Distillation，gradient_length=1，
师生保留相同 GaussianDistribution(init_std=0.3,std_type=scalar) 配置。
学生 PPO 将 student_state_dict 严格加载为 actor，使用新优化器/critic；
同阶段 --resume 恢复完整 checkpoint。PPO 初值为 5 epochs、4 minibatches、
lr=3e-4/adaptive、clip=0.2、entropy=0.005、gamma=0.99、lambda=0.95、
max_grad_norm=1、desired_kl=0.01。日志与存储由现成框架处理。

本阶段仅学生 actor 的初始输入不含特权真值；critic、reward 和固定执行器仍使用仿真状态，
不代表整个系统已经完成真机部署。

## 运行命令

本机解释器 `/home/ubuntu20/anaconda3/envs/hammer/bin/python`：mjlab 1.6.0、
MuJoCo / MuJoCo Warp 3.11.0、Torch 2.9.0+cu128、rsl-rl-lib 5.4.2。
AnyGrasp 独立解释器与原有授权布局未修改。当前机器使用 GLFW；EGL 设备显示初始化失败。
在仓库根目录执行：

```bash
conda activate hammer
export MUJOCO_GL=glfw
export MPLBACKEND=Agg

# 正式数据示例：会覆盖默认缓存，请按需执行；256 scenes 不保证足够泛化。
python -m hammer_rl.prepare --output weight/hammer_rl/reset_bank.npz --num-scenes 256 --candidates-per-scene 32 --seed 42

# 教师正式训练示例；本次只做短验证，不自动执行此长训练。
python -m hammer_rl.train --stage teacher --data-file weight/hammer_rl/reset_bank.npz --num-envs 128 --max-iterations 3000 --device cuda:0 --seed 42
```

缓存/模型/日志均放在已忽略的 weight/。若使用 --hammer-head-extra-mass，prepare 与
训练需采用同一个 run 级设置。评估自动读取 checkpoint 对应 run 的该设置。
以下记录本次短验证路径与结果。

本次短运行命令（路径均为真实本地产物；再次执行会更新对应 smoke 产物）：

```bash
python -m hammer_rl.prepare --output weight/hammer_rl/reset_bank.npz --num-scenes 8 --candidates-per-scene 2 --seed 42
python -m hammer_rl.train --stage teacher --data-file weight/hammer_rl/reset_bank.npz --num-envs 16 --max-iterations 2 --device cuda:0 --seed 42 --log-dir weight/hammer_rl/runs/teacher/smoke
python -m hammer_rl.train --stage teacher --data-file weight/hammer_rl/reset_bank.npz --num-envs 16 --max-iterations 1 --device cuda:0 --resume weight/hammer_rl/runs/teacher/smoke/model_1.pt
python -m hammer_rl.train --stage distill --data-file weight/hammer_rl/reset_bank.npz --num-envs 16 --max-iterations 1 --device cuda:0 --seed 42 --teacher-checkpoint weight/hammer_rl/runs/teacher/smoke/model_2.pt --log-dir weight/hammer_rl/runs/distill/smoke
python -m hammer_rl.train --stage student --data-file weight/hammer_rl/reset_bank.npz --num-envs 16 --max-iterations 1 --device cuda:0 --seed 42 --init-from weight/hammer_rl/runs/distill/smoke/model_0.pt --log-dir weight/hammer_rl/runs/student/smoke

python -m hammer_rl.evaluate --data-file weight/hammer_rl/reset_bank.npz --split eval --policy zero --episodes 8 --num-envs 8 --headless --output weight/hammer_rl/eval_zero.npz
python -m hammer_rl.evaluate --data-file weight/hammer_rl/reset_bank.npz --split eval --policy trained --checkpoint weight/hammer_rl/runs/teacher/smoke/model_2.pt --episodes 8 --num-envs 8 --headless --output weight/hammer_rl/eval_teacher.npz
python -m hammer_rl.evaluate --data-file weight/hammer_rl/reset_bank.npz --split eval --policy trained --checkpoint weight/hammer_rl/runs/student/smoke/model_0.pt --episodes 8 --num-envs 8 --headless --output weight/hammer_rl/eval_student.npz
python -m hammer_rl.evaluate --data-file weight/hammer_rl/reset_bank.npz --split eval --policy trained --checkpoint weight/hammer_rl/runs/student/smoke/model_0.pt --episodes 1 --viewer native
```

正式规模的初值为 num-envs=128，教师/蒸馏/学生分别可先用
max-iterations=3000/1000/3000。同阶段继续训练用 --resume；蒸馏需指定实际教师
checkpoint，学生首次 PPO 用 --init-from 指定实际蒸馏 checkpoint。
--max-iterations 始终表示本次追加次数。正式运行应使用充分训练的上阶段权重；
上述 smoke 权重仅用于验证流程，不代表可用的已收敛策略。

评估默认采用名义物理参数；三个策略使用相同 split、scene/candidate 顺序和参数。
--noise 可为各策略启用同样的可复现深度噪声。native 模式在同一执行器的子步回调中
按 DT 同步 CPU 数据及模型参考，不走训练时的 CPU 同步路径。

本次文件变更范围：

| 文件 | 必要性 |
|---|---|
| hammer_rl/prepare.py（新增） | 离线落稳、配对抓前深度和多候选缓存 |
| hammer_rl/train.py（新增） | 调用现成 runner，保存配置及阶段衔接 |
| hammer_rl/env_cfg.py、rl_cfg.py、__init__.py | 一次尝试的 MDP、原生传感器和训练/注册配置 |
| hammer_rl/mdp/resets.py、observations.py、curriculum.py | 按 env_ids 恢复缓存、冻结初始观测、摆放课程及物理随机化 |
| hammer_rl/mdp/actions.py、rewards.py、terminations.py | 一次 6D 解码、固定批量物理评价、五奖励及正常结束；替换旧 8D 适配 |
| hammer_rl/evaluate.py | 同样本对照、原始指标及完整 native 回放 |
| generation/grasp_hammer.py | 狭窄提取候选→mount 接口，增加固定已知 ROI 输入；默认行为不变 |
| README.md | 可运行命令、约定和真实验证记录 |

资产 XML、名义接触参数、连杆/驱动约束、原 generation 独立演示和 AnyGrasp 授权/权重未修改。
未新增算法、模型、runner 或测试框架。reset 位姿 jitter=0。

## 本次实际验证（2026-10-24，北京时间）

离线生成 8 个真实场景、16 个 SDK 候选，各类 1 个 train / 1 个 eval 场景。
prepare/train/evaluate 的 --help 均已核对；依赖安装目录统一为
`/home/ubuntu20/anaconda3/envs/hammer/lib/python3.11/site-packages/`。

- 16 环境 action=[16,6]，privileged=[16,87]、public_seed=[16,50]、
  depth=[16,1,120,160]；decimation=8980、step=17.96 s、max_episode_length=1。
  所有 done=True、timeout=False。16 个不同 scene/candidate 组合中抬升 12/16、有效敲击 7/16。
- 部分 reset 验证中，仅重置 env_ids=[1,7]，其余 time/qpos/qvel/sample/seed 完全相同。
  学生物理随机化的部分 reset 也保持其他环境的状态/图像/模型参考不变。
  重复 get_observations 的冻结加噪观测完全一致；CNN 的数值 normalizer 维度确为 50。
- 零残差与非零局部残差映射已核对。从同一真实抬升快照比较原 knock_path 的 666
  个轨迹点，最大位置误差 4.35e-7 m、旋转误差 3.89e-6 rad；开度/闭合、实际提升、
  weld 释放和重力补偿已记录。首次真实锤头接触的 netforce·外法向为 -65.197 N，
  因此训练使用负投影的压缩分量。未命中正常结束，撤离 touch 单列。
- 教师 PPO 完成 2 次 update，model_1.pt 的环境计数=2、优化器 step=40；
  加载续训 1 次后 model_2.pt 的环境计数=3、优化器 step=60，课程里程碑仍为 600/1800。
- 蒸馏完成 1 次 rollout/update（5 epochs 对应 5 次实际 optimizer step），
  然后严格载入 student_state_dict 到学生 PPO actor；学生完成 1 次 PPO update，
  新优化器 step=20。师生保留 distribution.std_param，未使用 strict=False。
- native viewer 实际回放 1 个完整零残差回合：有效提升 0.2473 m，专属锤头峰值
  63.307 N，原 touch 峰值 77.418 N；通过同一执行器的子步回调显示中间过程。
- Ruff、compileall、git diff --check 均通过。没有新增长期测试或运行时巡检。

下面为同一 4 个 eval 场景、8 个候选、名义物理参数、无视觉噪声的实际对照。
峰值力的均值/分位数包含未命中回合的 0 N。

| 策略 | 抬升成功 | 有效敲击 | F均值 N | F中位 N | F90分位 N | F最大 N | 总reward均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 62.5% | 37.5% | 12.0922 | 0.0000 | 32.5205 | 43.9142 | 1.4722 |
| teacher | 50.0% | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8848 |
| student | 50.0% | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8989 |

| 策略 | wrap | orientation | distance | stability | force |
|---|---:|---:|---:|---:|---:|
| zero | 0.00000066 | 0.58572695 | 0.52235886 | 0.00134635 | 0.09035782 |
| teacher | 0.00000000 | 0.46124254 | 0.42354240 | 0.00000000 | 0.00000000 |
| student | 0.00000000 | 0.47089122 | 0.42796834 | 0.00000000 | 0.00000000 |

评估程序还输出实际抓取距离、相对平移/旋转漂移、lift_height、挥击前有效抓持及独立
原 touch/撤离 touch；逐样本原始值保存在 weight/hammer_rl/eval_{zero,teacher,student}.npz，
可结合 lift_success/grasp_before_swing 区分失去抓持的回合。完整短运行日志保存在
weight/hammer_rl/validation_logs/，原轨迹阶段记录在 trajectory_trace.json。

实际 checkpoint（相对于仓库根目录）：

- 教师：weight/hammer_rl/runs/teacher/smoke/model_2.pt
- 蒸馏：weight/hammer_rl/runs/distill/smoke/model_0.pt
- 学生：weight/hammer_rl/runs/student/smoke/model_0.pt

本次学习策略在这组短验证中未优于零残差基线。未进行正式 256×32 数据准备、128
环境长训练、超参数搜索或收敛/泛化验证；额外锤头增重的非默认 run 本次未单独运行。
深度噪声和奖励尺度未经实机标定。所有模型都是真实短训练产物，无假权重，不能据此
宣称已经学到优于 AnyGrasp 的策略。
