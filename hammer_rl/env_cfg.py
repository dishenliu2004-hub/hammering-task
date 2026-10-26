"""原场景、一次尝试的 MDP，以及集中配置的初始实现约定。"""

from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg

from generation.hammer_inertia import add_head_mass
from .mdp.actions import GraspResidualActionCfg, attempt_substeps, collect_attempt
from .mdp import observations, rewards, resets, terminations, curriculum

SCENE = Path(__file__).resolve().parents[1] / "assets/scenes/scene.xml"


@dataclass(kw_only=True)
class HammerEnvCfg(ManagerBasedRlEnvCfg):
    data_file: str = "weight/hammer_rl/reset_bank.npz"
    stage: str = "teacher"  # teacher / distill / student
    split: str = "train"

    randomize_physics: bool = False
    visual_noise: bool = False
    # 质量/惯量与指面摩擦相对名义值的随机倍率。
    mass_range: tuple = (0.9, 1.1)
    friction_range: tuple = (0.8, 1.2)

    curriculum_milestones: tuple = (600, 1800)  # 课程切换步数

    lift_threshold: float = 0.15  # 有效抬升高度，m
    distance_ref: float = 0.315  # 距离归一化尺度，m
    stability_position_scale: float = 0.01  # 相对平移尺度，m
    stability_rotation_scale: float = math.pi / 36  # 相对旋转尺度，5°
    force_scale: float = 60.0  # 力归一化尺度，N


def _attach_scene(spec: mujoco.MjSpec, hammer_head_extra_mass=0.0) -> None:
    # 挂接原场景，沿用既有物理设置和控制约束。
    source = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(source, hammer_head_extra_mass)
    source.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    source.option.noslip_iterations = 0
    source.geom("hammer_head").condim = 6
    source.compile()
    spec.modelname = source.modelname
    spec.option = source.option
    spec.visual = source.visual
    spec.stat = source.stat
    spec.attach(source, prefix="", frame=spec.worldbody.add_frame())


def make_env_cfg(hammer_head_extra_mass=0.0, *, data_file="weight/hammer_rl/reset_bank.npz",
                 stage="teacher", num_envs=128, split="train", seed=42,
                 training=True, curriculum_milestones=(600, 1800)) -> HammerEnvCfg:
    # 使用总场景编译后的物理步长。
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    opt = model.opt
    action = GraspResidualActionCfg(entity_name="gripper_mount")
    # 一个环境 step 完成一次抓取和敲击尝试。
    decimation = attempt_substeps(opt.timestep, action.unload_time)

    sensors = (
        # 指面与锤柄接触：抓持判定。
        ContactSensorCfg(name="handle_contact",
            primary=ContactMatch(mode="geom", pattern=resets.FINGER_GEOMS),
            secondary=ContactMatch(mode="geom", pattern="hammer_handle"),
            fields=("found", "force"), reduce="maxforce", num_slots=2),
        # 掌面与整个锤子的接触：闭合末帧的严格真包络判定。
        # 保留接触槽逐点判断法向力，槽数与 nconmax 一致。
        ContactSensorCfg(name="palm_contact",
            primary=ContactMatch(mode="geom", pattern=("gripper_base_collision_part1", "gripper_base_collision_part2")),
            secondary=ContactMatch(mode="subtree", pattern="hammer_Object"),
            fields=("found", "force"), reduce="none", num_slots=128),
        # 锤柄在基座前端支撑面的有效顶靠。
        ContactSensorCfg(name="handle_seat_contact",
            primary=ContactMatch(mode="geom", pattern="gripper_base_collision_part1"),
            secondary=ContactMatch(mode="geom", pattern="hammer_handle"),
            fields=("found", "force", "pos", "normal"), reduce="none", num_slots=128),
        # 锤头与端盖合力：有效敲击评分。
        ContactSensorCfg(name="head_contact",
            primary=ContactMatch(mode="geom", pattern="hammer_head"),
            secondary=ContactMatch(mode="geom", pattern="cylinder_cylinder_endface_collision"),
            fields=("found", "force"), reduce="netforce"),
    )

    # 蒸馏和学生阶段默认启用物理及深度随机化。
    randomize = stage in ("distill", "student")
    events = {}
    if randomize:
        events["physics"] = EventTermCfg(func=resets.randomize_physics, mode="reset")
    events["scene"] = EventTermCfg(func=resets.reset_scene, mode="reset")

    cfg = HammerEnvCfg(data_file=str(data_file), stage=stage, split=split, seed=seed,
        randomize_physics=randomize, visual_noise=randomize,
        curriculum_milestones=tuple(curriculum_milestones),
        scene=SceneCfg(num_envs=num_envs, sensors=sensors,
            spec_fn=partial(_attach_scene, hammer_head_extra_mass=hammer_head_extra_mass)),
        sim=SimulationCfg(nconmax=128, njmax=1024, contact_sensor_maxmatch=128, mujoco=MujocoCfg(
            timestep=opt.timestep,
            integrator=mujoco.mjtIntegrator(opt.integrator).name.removeprefix("mjINT_").lower(),
            cone="elliptic",
            jacobian=mujoco.mjtJacobian(opt.jacobian).name.removeprefix("mjJAC_").lower(),
            solver=mujoco.mjtSolver(opt.solver).name.removeprefix("mjSOL_").lower(),
            impratio=opt.impratio, iterations=opt.iterations, tolerance=opt.tolerance,
            ls_iterations=opt.ls_iterations, ls_tolerance=opt.ls_tolerance,
            ccd_iterations=opt.ccd_iterations, gravity=tuple(opt.gravity), disableflags=("multiccd",))),
        decimation=decimation, episode_length_s=decimation * opt.timestep,
        # 单次尝试结束即 terminal，奖励不乘 dt。
        auto_reset=True, is_finite_horizon=True, scale_rewards_by_dt=False,
        actions={"grasp": action}, events=events,
        # 教师/critic 使用特权状态；学生 actor 使用公开数值和深度。
        observations={
            "privileged": ObservationGroupCfg(terms={"initial": ObservationTermCfg(func=observations.privileged)}),
            "public_seed": ObservationGroupCfg(terms={"initial": ObservationTermCfg(func=observations.public_seed)}),
            "depth": ObservationGroupCfg(terms={"initial": ObservationTermCfg(func=observations.depth)},
                flatten_history_dim=False),
        },
        # 包络、方位、距离、稳定、敲击力的权重依次为 1/1/0.5/2/5。
        rewards={name: RewardTermCfg(func=rewards.score, weight=weight, params={"name": name})
                 for name, weight in zip(("wrap", "orientation", "distance", "stability", "force"), (1, 1, 0.5, 2, 5))},
        terminations={"attempt_complete": TerminationTermCfg(func=terminations.attempt_complete)},
        # 子步收集快照和力峰值，回合末记录最终指标。
        metrics={"collect": MetricsTermCfg(func=collect_attempt, per_substep=True),
                 **{name: MetricsTermCfg(func=rewards.score, params={"name": name}, reduce="last")
                    for name in ("lift_success", "wrap", "orientation", "distance", "stability", "force",
                        "translation_drift", "rotation_drift", "peak_force", "valid_strike", "touch_peak",
                        "return_touch_peak", "lift_height", "grasp_before_swing", "grip_distance",
                        "seat_closure", "seat_lift", "seat_before_swing")}},
        # 仅教师训练启用摆放课程。
        curriculum={"placement": CurriculumTermCfg(func=curriculum.placement_curriculum)}
                   if stage == "teacher" and training else {},
    )
    return cfg
