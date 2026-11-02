# 本地 Codex 操作规范：最小修改锤击任务的动态训练分配

> 将本文完整交给本地 Codex，并在 `hammering-task` 项目根目录执行。
>
> 本次任务是直接完成代码修改和必要的一次性验证，不只是提供建议。严格按照下文确定的方案实现，不扩展项目范围，不新增安全检查、防御性代码、通用框架或依赖。

## 1. 任务目标与依据

调整教师训练中四类锤子放置场景的采样概率，使低成功率类别获得更多训练机会，同时让成功率较高但有效敲击力度不足的类别继续得到补强。

四类名称保持原样：

- `flat`
- `lean_head_down`
- `lean_head_up`
- `inverted`

分析基于提交 `17954640c632a6914c5b03d1afad431604cbed42`，标题为“根据效果动态分配训练空间”。该提交仅作为理解依据，不要求本地回退或切换到该提交。以当前工作区的实际代码、已有用户修改和项目指令为准，在相同职责位置做最小编辑。

当前实现的默认权重近似为：

```text
q[k] ∝ (success_rate[k] * (1 - success_rate[k])) ** 0.25
```

它偏向约 50% 成功率，可能压低接近零成功率类别的采样概率。本次用“成功缺口主导、力度不足补充、概率保底和上限”的公式替换它。

本次只能承诺改变训练资源分配，不能把公式核验或短运行宣称为四类成功率、敲击力已经提高。

## 2. 严格执行边界

### 2.1 只允许修改三个文件

| 文件 | 允许修改的内容 |
| --- | --- |
| `hammer_rl/mdp/curriculum.py` | 分配公式、该公式需要的历史统计、更新节奏、状态保存恢复和少量现有日志返回项 |
| `hammer_rl/train.py` | 一个显式重置课程参数，以及本规范指定的两处续训分支 |
| `hammer_rl/README.md` | 仅更新教师动态分配说明、参数与对应续训用法 |

不新增源代码、配置模块或永久测试文件，不把本操作规范复制成新的项目运行模块。不全文件格式化，不顺手修改其他文档中的历史问题。一次性验证产生的 Python 缓存，以及现有 weight/ 体系下新目录中的短运行日志和 checkpoint，不计入这三个源码/文档文件的编辑范围。

保留这些已有名称与连接方式：

```text
PlacementCurriculum
SGS_PARAMS
config["placement_sgs"]
infos["placement_sgs"]
_TeacherSgsRunner
task.sampling_weights
```

### 2.2 保持现有项目流程

以下部分不属于本次修改范围：

- `resets.py` 的场景采样与候选采样。
- `env_cfg.py`、`actions.py`、`rewards.py`、`terminations.py`。
- `prepare.py`、AnyGrasp、候选生成、数据划分和缓存格式。
- 抓姿残差范围、夹爪开度、固定执行轨迹、时序、物理参数和成功判定。
- 奖励项目与权重，包括当前 `force`、`task_success` 奖励。
- PPO、网络、优化器、学习率、观测、随机化和训练规模。
- 学生或蒸馏阶段的课程开关。
- `evaluate.py`、`compare_evaluations.py` 和可视化脚本。
- 已有 checkpoint 的模型权重格式、runner 类型和目录语义。

只在现有教师课程中生效。不要将上一轮分析中的奖励实验、学生课程、评估扩展等后续建议一起实现。

### 2.3 不新增安全检查或防御性代码

不得新增：

- 参数合法性校验、额外 `assert`、数据完整性检查。
- NaN/Inf 巡检、`nan_to_num`、异常值替换。
- 为“以防万一”添加的 `try/except`、默认兜底、自动降级。
- 旧 checkpoint 格式自动识别、字段缺失兼容、迁移框架。
- 额外输入检查、自动重试、自动回滚、自动修复。
- 通用调度器、通用投影器、统计管理器、replay buffer、参数注册系统。
- 新依赖、永久测试套件、运行时巡检和新的日志文件体系。

现有代码中的检查和框架行为保持原样，不为满足本条而删除现有机制。

下文明确规定的热身、成功样本门槛、全零优先级时均分、概率上限与平滑，是分配算法本身的正常分支，必须实现；不要围绕这些分支再添加异常保护代码。

一次性语法检查、公式核对和短运行用于确认修改正确，不写入项目的正常运行路径。

## 3. 本次采用的最小实现版本

相较上一轮完整建议，本次明确做以下收敛，不让 Codex自行追加功能：

1. **保留现有 multinomial 采样。** 12.5% 是目标采样概率下限；128 个环境时，每类期望至少约 16 次，实际数量会波动。不实现每批整数硬配额、环境绑定或跨批配额欠额。
2. **保留每类定长历史。** 把现有每类最近 100 条扩为最近 512 条，不实现统一 40 批时间窗口、时间戳或新的缓存结构。
3. **力度使用固定归一化尺度。** 不引入离线标定、日志搜索或自动目标校准流程；具体公式见第 5 节。
4. **旧模型切换使用显式参数。** 保留模型、优化器和迭代恢复，仅重置课程；不做旧课程状态迁移。
5. **保持教师阶段的现有边界。** 不改奖励、不扩展学生、不改评估程序。

逐类 512 条历史覆盖的实际时间长度可能不同，这是保留现有结构的取舍。本次不再为此增加时间窗口或多层平滑。

## 4. 先阅读现有代码，再编辑

在当前项目根目录中，阅读已有项目指令，以及以下文件：

```text
hammer_rl/mdp/curriculum.py
hammer_rl/mdp/resets.py
hammer_rl/mdp/actions.py
hammer_rl/env_cfg.py
hammer_rl/train.py
hammer_rl/rl_cfg.py
hammer_rl/README.md
```

用现有 `git status` / `git diff` 理解工作区已有修改。不要覆盖用户修改，不执行仓库重置、自动拉取、自动切分支、提交或推送。

需要确认的现有连接关系，仅用于定位编辑位置，不添加成运行时检查：

- 课程在旧回合指标清空前读取结果。
- `task_success` 是完整任务成功；`peak_force` 是 N；`impact_impulse` 是 N·s。
- `force` 当前是归一化冲量奖励，不是牛顿单位峰值力。
- 一次环境 step 完成一次尝试，当前 `num_steps_per_env=1`。
- `reset_scene()` 通过 `task.sampling_weights` 抽取场景。
- 当前 runner 已调用课程的 `state_dict()` / `load_state_dict()`。
- `--resume` 从 checkpoint 同目录配置读取 `placement_sgs`，并在该目录继续保存。

不重新实现这些已存在的能力。

## 5. 修改 curriculum.py

### 5.1 参数集中放在现有 SGS_PARAMS

保留常量名，替换其内容；同步修改现有 `__call__` 的参数列表，不保留已不用的旧参数：

```python
SGS_PARAMS = {
    "history_size": 512,
    "warmup_samples_per_class": 256,
    "update_interval": 10,
    "target_success": 0.9,
    "force_scale_n": 100.0,
    "force_mix": 0.7,
    "strength_weight": 0.5,
    "priority_power": 2.0,
    "min_probability": 0.125,
    "max_probability": 0.45,
    "probability_smoothing": 0.3,
    "min_success_samples": 32,
}
```

本次固定四类以及上述概率上下限，不泛化为任意类别数量、任意边界的分配框架。

`force_scale_n=100.0` 是本次提出的初始归一化尺度，未经本项目当前训练结果调优。它不是“已经验证合理的目标力”，也不是成功判定阈值。

冲量尺度直接读取现有 `env.cfg.impulse_scale`，不在课程中再创建另一份冲量尺度配置，不修改该环境参数。

### 5.2 只存必要状态

保留现有 `self.task`、训练 mask、类别索引和每类场景数量计算。

把四类 boolean 历史扩为四个位于 `env.device` 的 `float32` 张量，形状均为 `[n, 3]`，固定列顺序：

```text
第 0 列：task_success，0 或 1
第 1 列：peak_force，N
第 2 列：impact_impulse，N·s
```

每类仍使用已有的 `torch.cat(...)[-history_size:]` 追加、截断方式。三项指标必须从同一批已完成尝试一起追加，保持逐行对应。

只新增必要的两个持久成员：

```text
self.probabilities：形状 [4]，初始值均为 0.25
self.completed_batches：Python 整数，初始值为 0
```

热身通过四类历史长度均达到 256 判断。默认历史容量 512 大于热身门槛，因此达到以后历史长度不会再低于门槛，不额外保存热身计数器或布尔状态。

不存可由历史即时计算的重复统计状态，不新增 EMA 历史。

### 5.3 沿用完成回合过滤和结果读取时机

保留：

```python
completed = env.episode_length_buf[env_ids] >= 1
```

只用 `env_ids[completed]` 对应的旧场景类别和旧动作结果：

```python
values = env.action_manager.get_term("grasp").values
```

追加上述三项历史。

仅在本次课程调用实际含有已完成尝试时，执行：

```python
self.completed_batches += 1
```

这里利用当前项目正常教师训练“一次完整尝试一 step”的结构，不新增通用异步 reset 去重或 rollout 管理器。

初始化 reset、恢复后的零长度 reset：

- 不追加历史。
- 不增加完成批次计数。
- 不执行一次新的概率平滑。
- 仍需将当前 `self.probabilities` 写回场景权重。

不要将结果采集搬到 `reset_scene()` 之后，也不要在动作指标清零后读取。

### 5.4 计算成功率与力度不足程度

对每类历史计算：

```text
s[k] = task_success 的均值
n_success[k] = task_success 为 1 的行数
```

历史为空时，沿用现有成功率初值 0.5。热身阶段实际分配保持四类均分。

仅当该类历史中完整成功的回合数达到 `min_success_samples=32` 时，使用 PyTorch 现有 `median` 计算：

```text
F[k] = 成功回合 peak_force 的中位数
J[k] = 成功回合 impact_impulse 的中位数
```

不要混入 `task_success=0` 的历史峰值。当前代码在后来掉落时可能保留正的 `peak_force`，这些值不代表完整成功回合的力度。

力度不足程度定义为：

```python
d_force = force_scale_n / (F + force_scale_n)
d_impulse = env.cfg.impulse_scale / (J + env.cfg.impulse_scale)
d_strength = force_mix * d_force + (1.0 - force_mix) * d_impulse
```

成功样本不足 32 时，该类 `d_strength=0`，只由成功率缺口驱动资源分配。

这里的零表示“暂不启用力度项”，不能在说明中写成“已掌握力度”或“实测力为零”。

这是一处有意的最小化调整：上一轮的绝对目标缺口需要目标标定，本次改为固定尺度的连续有界评分。力度越大，该项越小；不自动改变尺度，不新增校准流程。

### 5.5 计算优先级

直接在现有课程中计算：

```python
d_success = ((target_success - success_rate) / target_success).clamp(0.0, 1.0)
priority = d_success + strength_weight * (1.0 - d_success) * d_strength
```

固定参数下：

```text
priority[k] = d_success[k] + 0.5 * (1 - d_success[k]) * d_strength[k]
```

含义：

- 零成功率类别的优先级为 1。
- 相同力度不足程度下，成功率越低，优先级越高。
- 成功率达到 90% 后，力度不足仍可以获得额外份额。
- 不使用总 reward、学习进步率或回合长度重新加权。

### 5.6 热身和分配更新节奏

满足以下条件时才更新概率：

1. 本次调用实际包含已完成尝试。
2. 四类历史长度都达到 `warmup_samples_per_class`。
3. `self.completed_batches % update_interval == 0`。

其他情况下保持已有概率。

初始均分持续到实际四类样本数均满足热身条件。保留随机采样后，不承诺固定 8 批就完成热身。

### 5.7 概率下限、上限和一次平滑

先计算优先级平方与原始目标概率：

```python
score = priority.pow(priority_power)
```

当 `score.sum() == 0` 时，目标概率为四类均分。这是算法定义，不要另加 epsilon、NaN 兜底或失败降级。

其他情况：

```python
raw = min_probability + (1.0 - 4.0 * min_probability) * score / score.sum()
```

对本次固定的四类、下限 0.125、上限 0.45，最多只有一类会超过上限。因此使用一次现有张量运算重新分配超出量即可：

```python
excess = (raw - max_probability).clamp_min(0.0)
receivers = (raw <= max_probability).to(raw.dtype)
target = raw - excess + excess.sum() * receivers / receivers.sum()
```

没有超限时 `excess` 为零，结果保持原样。该公式依赖本次明确固定的四类和边界，不包装成通用工具函数或投影求解器。

最后：

```python
self.probabilities = (
    (1.0 - probability_smoothing) * self.probabilities
    + probability_smoothing * target
)
```

只做这一层概率平滑。不要给成功率再叠 EMA，不新增变化率限制，不在平滑后追加 clamp 和重新归一化。

### 5.8 保留现有场景权重写法

每次课程调用都应用当前概率，包括没有完成尝试的 reset：

```python
self.task.sampling_weights.zero_()
self.task.sampling_weights[self.train_mask] = (
    self.probabilities[self.train_placement]
    / self.scene_counts[self.train_placement]
)
```

这段写回必须位于“本次是否存在完成尝试”的条件块之外。

不要因零长度 reset 直接提前返回，否则加载课程状态后，第一批可能仍使用构造时的均分采样权重。

不修改 `resets.py`，不改变类内场景均匀、场景内候选均匀的现有行为。

### 5.9 状态保存与日志

通过现有 `state_dict()` 保存：

```python
{
    "history": self.history,
    "probabilities": self.probabilities,
    "completed_batches": self.completed_batches,
}
```

`load_state_dict()` 按上述结构直接恢复。继续复用原 runner 的 checkpoint 保存、设备映射和加载，不增加格式探测、兼容 fallback 或自定义序列化。

保留现有日志键：

```text
success_rate_<placement>
probability_<placement>
```

只补充能直接解释力度门控的两个返回项：

```text
strength_deficit_<placement>
success_samples_<placement>
```

这些统计从当前历史重算，不额外保存一份日志状态，不新建日志文件或监控器。

## 6. 修改 train.py：只增加显式课程重置

### 6.1 新增一个 argparse 参数

```python
parser.add_argument(
    "--reset-placement-sgs",
    action="store_true",
    help="teacher续训时使用当前课程配置并清空课程状态，保留模型和优化器",
)
```

这个参数只用于教师续训时主动切换或重启课程。本次不为参数组合增加校验或报错分支。

### 6.2 第一处改动：环境构造前替换课程配置

在现有 `if args.resume:` 分支中，读取配置并恢复 `args.stage` 等值以后，加入：

```python
if args.stage == "teacher" and args.reset_placement_sgs:
    config["placement_sgs"] = SGS_PARAMS.copy()
```

必须位于现有写入 `run_config.json` 和构造环境之前。

这样旧配置中的 `concentration`、`temperature` 等参数不会被传给新课程，也不会发生“修改默认参数但续训仍使用旧配置”的问题。

### 6.3 第二处改动：只跳过课程状态加载

保留原有：

```text
runner.load(...)
runner.current_learning_iteration += 1
env.reset()
```

只将教师课程状态加载条件改为：

```python
if args.stage == "teacher" and not args.reset_placement_sgs:
    term = env.unwrapped.curriculum_manager.get_term_cfg("placement").func
    term.load_state_dict(infos["placement_sgs"])
```

不改模型、优化器、迭代次数和其他阶段的恢复逻辑。

显式重置时，新课程构造函数本来就已经建立空历史、均分概率和零计数，跳过课程加载即可；不要再写 `reset_state()` 或旧状态迁移方法。

### 6.4 三种运行语义写清楚

| 用法 | 行为 |
| --- | --- |
| 新 teacher run | 当前新参数；空历史、均分概率；从头训练 |
| 新策略 checkpoint 普通 `--resume` | 恢复保存的参数、历史、概率和计数，继续训练 |
| 旧 SGS checkpoint 加 `--reset-placement-sgs` | 恢复模型、优化器和迭代；采用当前新课程参数，重新均分热身 |

旧格式课程 checkpoint 不做无参数自动兼容。旧模型切换必须显式加 `--reset-placement-sgs`；后续恢复新 checkpoint 时不要再加，否则会再次重置课程。

现有 `_TeacherSgsRunner.save()` 已经保存课程 `state_dict()`，无需另改 runner 或增加 checkpoint 层。

## 7. README 只更新本次相关内容

把教师课程一节更新为本规范中的实际公式、参数和三种运行语义。

明确写入：

- 教师阶段仍从开始开放四类。
- 仅使用训练样本统计更新课程。
- 12.5%～45% 是目标概率边界，不是每批整数名额。
- 类内场景、场景内候选采样保持原样。
- 力度只统计完整成功回合，至少 32 个成功样本才启用。
- 100 N 是峰值力归一化初值，冲量尺度复用环境配置。
- 普通新策略续训恢复全部课程状态。
- 从旧 SGS 切换需要显式重置课程。
- 改分配不等于已经证明成功率和力度提升。

不展开奖励修改、学生训练扩展、评估重构或长期实验计划。其他文档及 README 中无关段落不顺手修改。

## 8. 只做必要的一次性验证

### 8.1 语法与补丁核验

使用项目原有 Python 环境；项目文档中的环境名为 `hammer`，若本地实际不同，仅使用已有正确环境，不创建环境、不安装新依赖。

```bash
python -m compileall -q hammer_rl/mdp/curriculum.py hammer_rl/train.py
git diff --check
git diff --stat
```

确认项目文件变更仅限第 2.1 节列出的三个文件。不要把这一确认写成新的运行时检查。

### 8.2 核对实际新公式

在临时会话中用实际修改后的计算段核对以下语义，不新增永久测试文件或第二份调度器实现：

- 相同成功率、相同力度统计且力度项启用状态相同的四类，目标概率为均分。
- 零成功类别的优先级为 1，不因力值未知而失去份额。
- 四类目标概率之和为 1，且落在 0.125～0.45。
- 仅一类优先级为正时，限幅后的目标为该类 0.45，其他各约 0.183333。
- 未完成热身时保持均分。
- 零长度 reset 不改变课程计数和概率，但会应用已恢复的场景权重。

参考数值例子：成功样本门槛已满足的前三类取

```text
成功率：          [0.8,       0.5,        0.2,   0.0]
成功峰值力中位：  [400.0,     233.333333, 100.0, 未启用] N
成功冲量中位：    [4.0,       2.333333,   1.0,   未启用] N·s
```

使用 `force_scale_n=100`、`env.cfg.impulse_scale=1`：

```text
d_strength ≈ [0.2, 0.3, 0.5, 0.0]
priority   ≈ [0.2, 0.527778, 0.833333, 1.0]
target_q   ≈ [0.134935, 0.194188, 0.297490, 0.373386]
```

这组 target_q 未触发上限。它是平滑前的目标概率，不是首次更新后的最终概率。

若初始概率为四类均分，第一次以 0.3 平滑后约为：

```text
q_next ≈ [0.215481, 0.233256, 0.264247, 0.287016]
```

以上是计算参考值，不是训练结果。

### 8.3 短运行与原有续训路径

已有依赖和训练缓存可用时，复用现有训练入口，在新的短运行目录做一次 2 次更新，再从其实际输出 checkpoint 普通续训 1 次。

示例仅在其中数据路径与本地现有缓存一致时使用；用已有缓存，不运行 `prepare`：

```bash
SMOKE_RUN="weight/hammer_rl/runs/teacher/placement_deficit_smoke_$(date +%Y%m%d_%H%M%S)"
python -m hammer_rl.train --stage teacher --data-file weight/hammer_rl/reset_bank_256.npz --num-envs 16 --max-iterations 2 --device cuda:0 --seed 42 --log-dir "$SMOKE_RUN"
```

续训命令使用程序实际打印的 checkpoint 路径，不猜 `model_*.pt` 的编号。

查看本次实际保存的配置和课程状态，确认新参数以及三项课程状态已经通过原接口保存、恢复。短运行通常仍处于均衡热身，不能把它声称为已经验证了长期动态分配效果。

不启动正式长训练，不运行整套基线评估，不新增测试框架。若现有依赖或数据使短运行不能执行，保留已完成代码修改，报告具体阻塞，不自动安装、补数据或增加兜底代码。

## 9. 旧教师模型切换说明

完成实现后，将下面的用法写入 README 并向用户说明；没有用户选定的源 checkpoint 时，不擅自选择“最佳”权重或启动迁移训练。

现有 `--resume` 以 checkpoint 所在目录作为输出目录，同时传 `--log-dir` 不会换目录。本次不修改该行为。

要保留旧实验，应先将选定 checkpoint 与它旁边的 `run_config.json` 复制到一个新的实验目录，再从副本续训。不要移动或改写源实验。

以下变量必须替换为本地实际值后使用：

```bash
SOURCE_CHECKPOINT=/实际旧教师训练目录/model_实际编号.pt
DEST_RUN=weight/hammer_rl/runs/teacher/placement_deficit_本次唯一标识
mkdir -p "$DEST_RUN"
cp "$SOURCE_CHECKPOINT" "$DEST_RUN/"
cp "$(dirname "$SOURCE_CHECKPOINT")/run_config.json" "$DEST_RUN/"
```

从副本切换：

```bash
python -m hammer_rl.train --stage teacher --resume "$DEST_RUN/$(basename "$SOURCE_CHECKPOINT")" --reset-placement-sgs --num-envs 128 --max-iterations 追加更新次数 --device cuda:0 --seed 42
```

这里的追加次数由用户的训练安排决定，Codex实施本次代码修改时不自动启动这条长训练命令。

之后恢复新产生的 checkpoint，使用原有 `--resume`，不再带 `--reset-placement-sgs`。

## 10. 完成标准与交付回复

完成修改后，直接交付结果，不停在方案描述。

回复必须简洁说明：

1. 实际修改的三个文件及必要性。
2. 新公式、关键参数和“概率保底”的含义。
3. 原 reset、奖励、动作、PPO、学生/蒸馏等均保持原职责。
4. 新策略普通续训及旧模型显式重置的准确用法。
5. 实际执行的语法检查、数值核对、短运行及真实结果；没有执行的项目如实说明。
6. 实际短运行目录和 checkpoint 路径，如有。

不要声称四类成功率或敲击力已经明显提升，除非确实完成了独立训练与评估。不要在交付时继续追加奖励调参、扩动作空间、学生课程或新的安全检查。

## 11. 源码依据

以下链接固定到分析时的提交，便于核对原职责；本地实现仍以当前工作区为准。

- [原四类课程、公式与状态接口](https://github.com/dishenliu2004-hub/hammering-task/blob/17954640c632a6914c5b03d1afad431604cbed42/hammer_rl/mdp/curriculum.py)
- [原场景和候选采样](https://github.com/dishenliu2004-hub/hammering-task/blob/17954640c632a6914c5b03d1afad431604cbed42/hammer_rl/mdp/resets.py)
- [原指标采集与一次抓姿动作](https://github.com/dishenliu2004-hub/hammering-task/blob/17954640c632a6914c5b03d1afad431604cbed42/hammer_rl/mdp/actions.py)
- [原环境奖励、冲量尺度和课程启用阶段](https://github.com/dishenliu2004-hub/hammering-task/blob/17954640c632a6914c5b03d1afad431604cbed42/hammer_rl/env_cfg.py)
- [原训练配置与 checkpoint 保存恢复](https://github.com/dishenliu2004-hub/hammering-task/blob/17954640c632a6914c5b03d1afad431604cbed42/hammer_rl/train.py)

