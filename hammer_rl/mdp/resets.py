"""基于 UWLab 离线候选采样机制的 mjlab near-object reset；未移植 IK/已抓持状态。"""

from types import SimpleNamespace

import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from visualization.noise.depth_noise import DepthNoise

STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active",
                "qfrc_applied", "xfrc_applied", "qacc_warmstart")
FINGER_GEOMS = tuple(f"gripper_hand_{side}_fingertip_{part}"
                     for side in ("narrow", "wide") for part in ("proximal", "distal", "tip"))


def initialize_task(env):
    """仅在环境构造时加载一次；reset 事件与动作项共享持久 seed buffer。"""
    with np.load(env.cfg.data_file) as source:
        bank = {key: torch.as_tensor(source[key], device=env.device,
                  dtype=torch.float32 if source[key].dtype.kind == "f" else None)
                for key in source.files}
    model = env.sim.mj_model
    ids = torch.arange(env.num_envs, device=env.device)
    task = SimpleNamespace(bank=bank, scene_id=ids * 0, candidate_id=ids * 0,
        seed_pos=torch.zeros((env.num_envs, 3), device=env.device),
        seed_quat=torch.zeros((env.num_envs, 4), device=env.device),
        open_ctrl=torch.zeros(env.num_envs, device=env.device),
        depth=torch.zeros((env.num_envs, 1, 120, 160), device=env.device),
        sampling_weights=((bank["split"] == (env.cfg.split == "eval")).float()),
        selection=None, hammer_id=model.body("hammer_Object").id,
        mocap_id=int(model.body_mocapid[model.body("gripper_mount").id]),
        base_qadr=int(model.joint("gripper_base_free").qposadr[0]),
        base_vadr=int(model.joint("gripper_base_free").dofadr[0]),
        finger_ids=torch.tensor([model.geom(name).id for name in FINGER_GEOMS], device=env.device),
        nominal_mass=torch.as_tensor(model.body_mass.copy(), dtype=torch.float32, device=env.device),
        nominal_inertia=torch.as_tensor(model.body_inertia.copy(), dtype=torch.float32, device=env.device),
        nominal_friction=torch.as_tensor(model.geom_friction.copy(), dtype=torch.float32, device=env.device))
    task.finger_qadr = torch.tensor([int(model.joint(i).qposadr[0]) for i in range(model.njnt)
                                   if model.joint(i).name.startswith("gripper_hand_")], device=env.device)
    task.finger_vadr = torch.tensor([int(model.joint(i).dofadr[0]) for i in range(model.njnt)
                                   if model.joint(i).name.startswith("gripper_hand_")], device=env.device)
    task.noise = DepthNoise(focal_length=float(bank["intrinsics"][0, 0]), baseline=0.05,
                           min_depth=0.3, max_depth=3.0, filter_size=3).to(env.device)
    env.hammer = task
    # 构造 observation manager 时只需确定维度；第一次 reset 才写真实初始观测。
    task.public = torch.zeros((env.num_envs, 50), device=env.device)
    task.privileged = torch.zeros((env.num_envs, 87), device=env.device)


@requires_model_fields("geom_friction", "body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_physics(env, env_ids):
    task, model = env.hammer, env.sim.model
    count, body = len(env_ids), task.hammer_id
    scale = torch.empty((count, 1), device=env.device).uniform_(*env.cfg.mass_range)
    model.body_mass[env_ids, body] = task.nominal_mass[body] * scale[:, 0]
    model.body_inertia[env_ids, body] = task.nominal_inertia[body] * scale
    friction = torch.empty((count, 1, 1), device=env.device).uniform_(*env.cfg.friction_range)
    model.geom_friction[env_ids[:, None], task.finger_ids] = task.nominal_friction[task.finger_ids] * friction


@requires_model_fields("body_pos", "body_quat", recompute=RecomputeLevel.set_const_0)
def reset_scene(env, env_ids):
    task, data, model = env.hammer, env.sim.data, env.sim.model
    bank, count = task.bank, len(env_ids)
    if task.selection is None:
        scene = torch.multinomial(task.sampling_weights, count, replacement=True)
        start = bank["offsets"][scene]
        candidate = start + (torch.rand(count, device=env.device) *
                            (bank["offsets"][scene + 1] - start)).long()
    else:
        scene, candidate = task.selection
        scene, candidate = scene[env_ids], candidate[env_ids]
    task.scene_id[env_ids], task.candidate_id[env_ids] = scene, candidate
    model.body_pos[env_ids, task.hammer_id] = bank["reference_pos"][scene]
    model.body_quat[env_ids, task.hammer_id] = bank["reference_quat"][scene]
    for name in STATE_FIELDS:
        getattr(data, name)[env_ids] = bank[name][scene]
    data.time[env_ids] = 0
    # 使用同条缓存的实际世界初态，避免在 forward 之前读取旧 xpos。
    object_pos, object_quat = bank["hammer_pos"][scene], bank["hammer_quat"][scene]
    seed_pos = object_pos + quat_apply(object_quat, bank["local_pos"][candidate])
    seed_quat = quat_mul(object_quat, bank["local_quat"][candidate])
    task.seed_pos[env_ids], task.seed_quat[env_ids] = seed_pos, seed_quat
    task.open_ctrl[env_ids] = bank["open_ctrl"][candidate]
    # R_mount = R_grasp Ry(pi/2)，所以接近轴为 mount 的 +Z。
    approach = torch.zeros_like(seed_pos)
    approach[:, 2] = 0.10
    pregrasp = seed_pos - quat_apply(seed_quat, approach)
    data.mocap_pos[env_ids, task.mocap_id] = pregrasp
    data.mocap_quat[env_ids, task.mocap_id] = seed_quat
    qadr, vadr = task.base_qadr, task.base_vadr
    data.qpos[env_ids, qadr:qadr + 3] = pregrasp
    data.qpos[env_ids, qadr + 3:qadr + 7] = seed_quat
    data.qvel[env_ids, vadr:vadr + 6] = 0
    # 保留原开爪手指构型；候选开度由固定预抓段调整。
    depth = bank["depth"][scene]
    task.depth[env_ids] = task.noise(depth) if env.cfg.visual_noise else depth
    public = torch.cat((seed_pos, seed_quat, bank["widths"][candidate, None],
        task.open_ctrl[env_ids, None], pregrasp, seed_quat,
        data.qpos[env_ids[:, None], task.finger_qadr],
        data.qvel[env_ids[:, None], task.finger_vadr],
        bank["camera_pos"][scene], bank["camera_quat"][scene], bank["intrinsics"][scene],
        bank["target_pos"][scene], bank["target_quat"][scene]), dim=-1)
    # 所有量冻结于决策前；结果快照绝不进入 actor/critic。
    head_offset = torch.zeros_like(object_pos)
    head_offset[:, 0] = -0.24
    support_size = torch.tensor([0.42, 0.20, 0.37], device=env.device).expand(count, -1)
    mass = (model.body_mass[env_ids, task.hammer_id] if env.cfg.randomize_physics else
            task.nominal_mass[task.hammer_id].expand(count))
    inertia = (model.body_inertia[env_ids, task.hammer_id] if env.cfg.randomize_physics else
               task.nominal_inertia[task.hammer_id].expand(count, -1))
    friction = (model.geom_friction[env_ids[:, None], task.finger_ids, 0] if env.cfg.randomize_physics else
                task.nominal_friction[task.finger_ids, 0].expand(count, -1))
    privileged = torch.cat((public, object_pos, object_quat,
        quat_apply_inverse(seed_quat, object_pos - seed_pos),
        quat_mul(quat_conjugate(seed_quat), object_quat),
        object_pos + quat_apply(object_quat, head_offset),
        bank["support_pos"][scene], bank["support_quat"][scene], support_size,
        mass[:, None], inertia, friction), dim=-1)
    task.public[env_ids], task.privileged[env_ids] = public, privileged
