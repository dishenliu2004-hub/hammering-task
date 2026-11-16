"""原场景抓取起点的 reset 和冻结观测。"""

from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_apply

from visualization.noise.depth_noise import DepthNoise
from generation.hammer_geometry import HEAD_CENTER_LOCAL

STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active",
                "qfrc_applied", "xfrc_applied", "qacc_warmstart")
FINGER_GEOMS = tuple(f"gripper_hand_{side}_fingertip_{part}"
                     for side in ("narrow", "wide") for part in ("proximal", "distal", "tip"))


def initialize_task(env):
    """仅在环境构造时加载一次；reset 事件与动作项共享场景缓存。"""
    with np.load(env.cfg.data_file) as source:
        bank = {key: torch.as_tensor(source[key], device=env.device,
                  dtype=torch.float32 if source[key].dtype.kind == "f" else None)
                for key in source.files}
    model = env.sim.mj_model
    ids = torch.arange(env.num_envs, device=env.device)
    task = SimpleNamespace(bank=bank, scene_id=ids * 0,
        depth=torch.zeros((env.num_envs, 1, 120, 160), device=env.device),
        sampling_weights=((bank["split"] == (env.cfg.split == "eval")).float()),
        selection=None, hammer_id=model.body("hammer_Object").id,
        handle_geom_id=model.geom("hammer_handle").id,
        spring_joint_ids=torch.tensor([model.joint(f"gripper_hand_{side}2_joint").id
                                       for side in ("narrow", "wide")], device=env.device),
        mocap_id=int(model.body_mocapid[model.body("gripper_mount").id]),
        base_qadr=int(model.joint("gripper_base_free").qposadr[0]),
        base_vadr=int(model.joint("gripper_base_free").dofadr[0]),
        finger_ids=torch.tensor([model.geom(name).id for name in FINGER_GEOMS], device=env.device),
        nominal_mass=torch.as_tensor(model.body_mass.copy(), dtype=torch.float32, device=env.device),
        nominal_inertia=torch.as_tensor(model.body_inertia.copy(), dtype=torch.float32, device=env.device),
        nominal_friction=torch.as_tensor(model.geom_friction.copy(), dtype=torch.float32, device=env.device),
        nominal_stiffness=torch.as_tensor(model.jnt_stiffness.copy(), dtype=torch.float32, device=env.device))
    task.finger_qadr = torch.tensor([int(model.joint(i).qposadr[0]) for i in range(model.njnt)
                                   if model.joint(i).name.startswith("gripper_hand_")], device=env.device)
    task.finger_vadr = torch.tensor([int(model.joint(i).dofadr[0]) for i in range(model.njnt)
                                   if model.joint(i).name.startswith("gripper_hand_")], device=env.device)
    task.noise = DepthNoise(focal_length=float(bank["intrinsics"][0, 0]), baseline=0.05,
                           min_depth=0.3, max_depth=3.0, filter_size=3).to(env.device)
    env.hammer = task
    # 构造 observation manager 时只需确定维度；第一次 reset 才写真实初始观测。
    task.public = torch.zeros((env.num_envs, 34), device=env.device)
    task.privileged = torch.zeros((env.num_envs, 66), device=env.device)


@requires_model_fields("geom_friction", "body_mass", "body_inertia", "jnt_stiffness", recompute=RecomputeLevel.set_const)
def randomize_physics(env, env_ids):
    task, model = env.hammer, env.sim.model
    count, body = len(env_ids), task.hammer_id
    scale = torch.empty((count, 1), device=env.device).uniform_(*env.cfg.mass_range)
    model.body_mass[env_ids, body] = task.nominal_mass[body] * scale[:, 0]
    model.body_inertia[env_ids, body] = task.nominal_inertia[body] * scale
    friction = torch.empty((count, 1, 1), device=env.device).uniform_(*env.cfg.friction_range)
    model.geom_friction[env_ids[:, None], task.finger_ids] = task.nominal_friction[task.finger_ids] * friction
    spring_scale = torch.empty((count, 1), device=env.device).uniform_(*env.cfg.spring_stiffness_range)
    model.jnt_stiffness[env_ids[:, None], task.spring_joint_ids] = (
        task.nominal_stiffness[task.spring_joint_ids] * spring_scale)


def public_observation(*, finger_pos, finger_vel,
                       camera_pos, camera_quat, intrinsics, target_pos, target_quat):
    """仿真缓存和在线公开输入共用的 34 维字段；仅计算张量。"""
    return torch.cat((finger_pos, finger_vel, camera_pos, camera_quat, intrinsics,
        target_pos, target_quat), dim=-1)


def _initial_observations(env, env_ids, scene):
    task, model = env.hammer, env.sim.model
    bank, count = task.bank, len(env_ids)
    # 使用同条缓存的实际世界初态，避免在 forward 之前读取旧 xpos。
    object_pos, object_quat = bank["hammer_pos"][scene], bank["hammer_quat"][scene]
    public = public_observation(
        finger_pos=bank["qpos"][scene[:, None], task.finger_qadr],
        finger_vel=bank["qvel"][scene[:, None], task.finger_vadr],
        **{key: bank[key][scene] for key in
           ("camera_pos", "camera_quat", "intrinsics", "target_pos", "target_quat")})
    # 所有量冻结于本次决策前；不包含执行后的评分。
    head_offset = object_pos.new_tensor(HEAD_CENTER_LOCAL).expand_as(object_pos)
    support_size = torch.tensor([0.42, 0.20, 0.37], device=env.device).expand(count, -1)
    mass = (model.body_mass[env_ids, task.hammer_id] if env.cfg.randomize_physics else
            task.nominal_mass[task.hammer_id].expand(count))
    inertia = (model.body_inertia[env_ids, task.hammer_id] if env.cfg.randomize_physics else
               task.nominal_inertia[task.hammer_id].expand(count, -1))
    friction = (model.geom_friction[env_ids[:, None], task.finger_ids, 0] if env.cfg.randomize_physics else
                task.nominal_friction[task.finger_ids, 0].expand(count, -1))
    privileged = torch.cat((public, object_pos, object_quat, bank["handle_size"][scene],
        object_pos + quat_apply(object_quat, head_offset),
        bank["support_pos"][scene], bank["support_quat"][scene], support_size,
        mass[:, None], inertia, friction), dim=-1)
    return TensorDict({"public_seed": public,
                       "privileged": privileged,
                       "depth": task.depth[env_ids]}, batch_size=[count])


@requires_model_fields("body_pos", "body_quat", "geom_size", "geom_rbound", "geom_aabb",
                       recompute=RecomputeLevel.set_const_0)
def reset_scene(env, env_ids):
    task, data, model = env.hammer, env.sim.data, env.sim.model
    bank, count = task.bank, len(env_ids)
    if task.selection is None:
        scene = torch.multinomial(task.sampling_weights, count, replacement=True)
    else:
        scene = task.selection[env_ids]
    task.scene_id[env_ids] = scene
    sizes = bank["handle_size"][scene]
    radius, half_length = sizes[:, 0], sizes[:, 1]
    h = task.handle_geom_id
    model.geom_size[env_ids, h, :2] = sizes
    model.geom_rbound[env_ids, h] = radius + half_length
    model.geom_aabb[env_ids, h, 1, 0] = radius
    model.geom_aabb[env_ids, h, 1, 1] = radius
    model.geom_aabb[env_ids, h, 1, 2] = radius + half_length
    model.body_pos[env_ids, task.hammer_id] = bank["reference_pos"][scene]
    model.body_quat[env_ids, task.hammer_id] = bank["reference_quat"][scene]
    for name in STATE_FIELDS:
        getattr(data, name)[env_ids] = bank[name][scene]
    data.time[env_ids] = 0
    depth = bank["depth"][scene]
    task.depth[env_ids] = task.noise(depth) if env.cfg.visual_noise else depth
    obs = _initial_observations(env, env_ids, scene)
    task.public[env_ids], task.privileged[env_ids] = obs["public_seed"], obs["privileged"]
