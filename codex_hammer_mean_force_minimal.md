# Codex 操作任务书：将锤击力峰值统计改为固定窗口内的逐帧平均力

## 0. 任务身份与目标

你是直接在现有仓库中修改代码的 Codex。请检查当前工作树后，以仓库现有实现为准执行**最小必要修改**，不要另起一套力传感器或敲击统计框架。

- 仓库：`https://github.com/dishenliu2004-hub/hammering-task`
- 目标：将对外使用的“最大敲击力”改为“**锤头首次有效接触测力器端面之后，固定 40 ms 窗口内每个物理子步法向力的算术平均值**”。
- 统计窗口：沿用现有 `hammer_rl/env_cfg.py` 中的 `impact_window_s = 0.04`；单次敲击脚本也使用同一数值 `0.04 s`。不新增配置系统。
- 现有物理步长为 `0.002 s`，故窗口对应 20 个物理子步。时间窗口从**首次命中的那一帧开始，包含该帧**。
- 未接触或力为零的帧计入分母，**不只平均非零接触帧**。未命中时结果为 `0 N`。
- 统计的是**法向力标量的平均**，不是三维向量范数、冲量、渲染帧均值，也不是多个回合的峰值均值。

数学定义：

\[
N=\operatorname{round}(T/\Delta t),\qquad
F_{\mathrm{mean}}=\frac{1}{N}\sum_{i=0}^{N-1}F(t_{\mathrm{first}}+i\Delta t)
\]

其中 `T = 0.04 s`；`Δt` 使用项目自身物理步长。对于原始接触检测/奖励门控之外的帧，沿用现有代码的有效力定义，不重写物理判据。

## 1. 必须遵守的代码边界

1. **只改以下 6 个文件**：
   - `hammer_rl/mdp/actions.py`
   - `hammer_rl/env_cfg.py`
   - `hammer_rl/mdp/curriculum.py`
   - `hammer_rl/compare_evaluations.py`
   - `generation/impact_force.py`
   - `generation/main.py`
2. **不要修改** `generation/knock_path.py`、`assets/` 目录中的 XML、传感器定义、物理步长、接触匹配、抓取/挥击轨迹、控制过程、reset/termination/奖励权重、仿真后端及其他文件。
3. **禁止添加任何安全检查或防御性代码**：不要加 `try/except`、额外断言、参数校验、空值判断、分母保护、`max(1, ...)`、回退路径、超时机制、兜底状态或异常吞没。
4. **不要重复造轮子**：不要引入新类、统一统计模块、缓冲数组、逐帧历史列表、额外 GPU/CPU 同步、第三方库或抽象层。
5. 沿用现有 `hit`、`first_hit_step`、`impact_impulse`、`head_contact`、`touch` 传感器和现有调用时序。不得改变抓取有效性门控、成功率和掉落逻辑的含义。
6. 只做与本次平均力需求直接相关的变量改名、计算替换与输出字段同步；不进行重构、格式化全文件、代码“顺手优化”。
7. 遇到仓库实际代码与本任务书片段不一致时，定位对应语义做最小修改，不机械套用行号。

## 2. 修改强化学习主路径

### 2.1 `hammer_rl/mdp/actions.py`：复用已有冲量积分

现有 `observe_substep()` 已做这些事情：

- `head_contact` 给出锤头与端盖接触合力；
- `force = (-(contact.force[:, 0] * self.normal).sum(-1)).clamp_min(0)` 得到压缩法向力；
- `hit` 和 `first_hit_step` 记录首次**有效**命中；
- `window_steps = round(self._env.cfg.impact_window_s / self._env.physics_dt)` 定义窗口；
- `impact_impulse += force * physics_dt` 在窗口内进行积分。

**不重写以上接触、计时、冲量逻辑**。在当前项目原有 `evaluating & holding & in_window` 门控下，直接按 `平均力 = 现有累计冲量 / 统计窗口时长` 计算。这样未计入冲量的窗口帧等价于零力帧，同时保持原始有效敲击评分口径。

**修改 A：** 在 `self.values` 的指标名称元组里，把：

```python
"translation_drift", "rotation_drift", "peak_force",
```

替换为：

```python
"translation_drift", "rotation_drift", "mean_force",
```

**修改 B：** 保留 `first_hit_step`、`valid_strike` 与 `impact_impulse` 的现有代码，删除对 `values["peak_force"]` 做 `torch.maximum(...)` 的更新行。不添加逐帧样本列表。

在现有的：

```python
values["impact_impulse"].add_(torch.where(
    evaluating & holding & in_window,
    force * self._env.physics_dt, 0
))
```

后面增加：

```python
values["mean_force"].copy_(
    values["impact_impulse"] / (window_steps * self._env.physics_dt)
)
```

这里 `window_steps * physics_dt` 是实际离散窗口时长；**不要使用已经过了多少帧作为动态分母**。

**修改 C：** 将原来根据峰值力计算奖励的语句：

```python
values["force"].copy_(torch.where(
    evaluating, values["peak_force"] / 10, values["force"]
))
```

替换为：

```python
values["force"].copy_(torch.where(
    evaluating, values["mean_force"] / 10, values["force"]
))
```

奖励权重、`/ 10` 的尺度、`torch.where(evaluating, ...)` 门控均不改变。

**修改 D：** 在 `observe_substep()` 末尾，将：

```python
return values["peak_force"]
```

替换为：

```python
return values["mean_force"]
```

注意：`collect` 是框架对子步返回值的跨子步归约，不应将 `Episode_Metrics/collect` 误读为最终单次敲击的平均力。最终单次敲击值以 `Episode_Metrics/mean_force` 为准；不要为此修改 metrics 框架。

### 2.2 `hammer_rl/env_cfg.py`：同步注册输出名称

原有 `impact_window_s: float = 0.04` **保持原样**。

仅在 `metrics` 的名称列表中，将：

```python
"translation_drift", "rotation_drift", "peak_force", "valid_strike",
```

改成：

```python
"translation_drift", "rotation_drift", "mean_force", "valid_strike",
```

`rewards` 列表及其权重不变。最终训练指标为：

- `Episode_Metrics/mean_force`：每次尝试的 40 ms 窗口平均法向力，单位 N；
- `Episode_Metrics/force`：平均力除以 10 后的原有力度评分，其他门控不变；
- `Episode_Metrics/impact_impulse`：原先 40 ms 法向冲量，单位 N·s，不变。

### 2.3 `hammer_rl/mdp/curriculum.py`：同步课程采样的力度字段

在 `PlacementCurriculum.__call__()` 的历史采样中，将：

```python
("task_success", "peak_force", "impact_impulse")
```

改为：

```python
("task_success", "mean_force", "impact_impulse")
```

三列的顺序和后续 `successful[:, 1].median()`、冲量统计、课程参数及公式全部保持不变。此处仅改变第 2 列“力度”含义。

### 2.4 `hammer_rl/compare_evaluations.py`：同步评估 CSV

将：

```python
METRICS = (("valid_strike", 100.0, np.mean),
           ("peak_force", 1.0, np.mean))
```

改为：

```python
METRICS = (("valid_strike", 100.0, np.mean),
           ("mean_force", 1.0, np.mean))
```

并且将 CSV 表头三列：

```python
"anygrasp_peak_force_N", "trained_peak_force_N", "delta_peak_force_N",
```

替换为：

```python
"anygrasp_mean_force_N", "trained_mean_force_N", "delta_mean_force_N",
```

其他评估行为、`np.mean` 跨回合汇总方法、对比逻辑与文件路径不变。这里跨回合的 `np.mean` 作用于**每个回合自己的 40 ms 平均力**，不是重新跨回合汇总物理帧。

## 3. 修改单次敲击脚本

这一路径不是强化学习主路径：`generation/main.py` 经 `generation/knock_path.py` 调用 `generation/impact_force.py`，其力来自现有 `cylinder_endface_normal_force` touch 传感器。**不要与强化学习路径的 `head_contact` 传感器进行不必要的合并**。

### 3.1 `generation/impact_force.py`：在原类中加最少量状态

保留现有 `PeakImpactForce` 类名、`peak` 属性、`swing_peak`/`return_peak` 统计及接触状态控制，避免迫使 `knock_path.py` 变更。只添加对外使用的平均力结果。

在 `__init__(self, model, simulation)` 现有统计状态附近增加：

```python
self.window_steps = round(0.04 / model.opt.timestep)
self.force_sum = 0.0
self.force_frames = 0
```

在现有 `peak` 属性后面新增：

```python
@property
def mean_force(self):
    return self.force_sum / self.window_steps
```

原 `__call__()` 继续用 `hammer_head` 与 `cylinder_cylinder_endface_collision` 的接触配对、`force > 0` 与 `impact_started` 来确定首次接触；**不修改已有首次接触判定**。

在 `__call__()` 的**末尾、原有最终 `return self.peak` 之前**加入（与最外层 `if self.phase == "return":` 同一级缩进）：

```python
if self.impact_started and self.force_frames < self.window_steps:
    self.force_sum += force
    self.force_frames += 1
```

注意放置位置：这三行必须在 `if self.phase == "return": ... elif not self.impact_finished: ...` 两个分支**之后**，而非嵌入 `elif` 中。这样：

- 首次接触当帧会被计入；
- 后续挥击/卸载帧会被计入；
- 如果统计窗口跨入撤离阶段，仍按现有逐物理子步调用继续采集；
- 即使接触后某帧 `force == 0`，仍记为一个窗口帧；
- 未命中时不启动累计；
- 未在窗口内取得的数据对应零贡献，分母仍固定为 `window_steps`。

不修改 `generation/knock_path.py`，也不要修改该类其他返回值或控制逻辑。

### 3.2 `generation/main.py`：只输出新的最终统计值

把现有三条“挥击峰值、撤离峰值、撞击峰值”的 `print(...)` 语句替换为一条：

```python
print(f"首次接触后40ms平均法向敲击力：{impact_force.mean_force:.6f} N")
```

原有“抓取中心到锤头中心距离”输出保持不变。不要新增日志模块、持久化模块或实时显示组件。

## 4. 明确禁止的过度修改

不要做以下事情：

- 为平均力创建另一个 `ImpactForceTracker`、`ForceWindow`、`deque`、Tensor 缓冲区或专门的窗口管理器；
- 为强化学习重复积分一次 `force * dt`：已有 `impact_impulse` 即可使用；
- 为单次敲击改动接触传感器 XML 或重新推导接触法线；
- 把物理子步换成渲染帧或外层 `dt=0.02 s`；
- 在 40 ms 内只对非零帧平均，或在接触消失时缩短平均窗口分母；
- 偷改抓取质量、命中门控、卸载终止、有效成功判定、奖励权重及课程算法；
- 增添安全/防御代码、测试框架、演示脚本、命令行参数、环境依赖；
- 大面积批量替换文档中历史的 `peak_force` 描述，或额外修改 README/训练说明；本任务以运行代码与直接输出保持一致为验收范围。

## 5. 运行检查与验收标准（不新增防御性代码）

修改后只执行现有工程环境允许的轻量检查；不要为了检查引入新测试文件或依赖。

1. 使用 `git diff --stat`、`git diff` 核对，确认只出现上述 6 个文件的必要改动。
2. 执行：

   ```bash
   python -m compileall -q \
     hammer_rl/mdp/actions.py \
     hammer_rl/env_cfg.py \
     hammer_rl/mdp/curriculum.py \
     hammer_rl/compare_evaluations.py \
     generation/impact_force.py \
     generation/main.py
   ```

3. 在已具备项目依赖和数据的环境中，运行**已有**的最短评估或单次敲击入口检查输出；若现有运行环境不满足条件，直接报告未运行原因，**不要**为运行补环境兼容层。
4. 静态核对：
   - 强化学习指标/奖励/课程/CSV 使用 `mean_force`，不再使用 `peak_force` 作为有效力度指标；
   - `impact_impulse` 原有定义不变；
   - `generation/main.py` 输出 `impact_force.mean_force`，而不是三类峰值；
   - 40 ms 窗口在 2 ms 物理步长下为 20 帧；首帧被统计，零力帧仍占分母。
5. 可用以下数学例子核对语义（**不必新增测试代码**）：
   - 20 帧全为 `50 N` → 平均 `50 N`；
   - 前 5 帧各 `100 N`，后 15 帧各 `0 N` → 平均 `25 N`；
   - 未发生有效接触 → 平均 `0 N`。
6. 现有可见的训练曲线、历史 CSV 或旧 checkpoint 不会自动转成新语义；不要悄悄修改旧结果，也不要声称旧结果可直接与新值比较。

## 6. Codex 完成时的交付格式

请直接修改工作树中的代码，不要只回答建议，也不要生成替代项目。完成后仅汇报：

1. 实际修改的文件与每个文件的必要变化；
2. 使用的平均力公式和 `40 ms / 20 个物理子步` 统计口径；
3. 执行过的检查与结果、未执行的运行测试及原因；
4. `git diff --stat` 概要。

**不要创建提交，不要修改任务范围外的文件，不要添加任何安全检查、防御性代码或额外架构。**
