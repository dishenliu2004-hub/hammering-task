"""原场景抓取起点与离线真实抓持快照共用的 reset 和冻结观测。"""

from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
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
    strike_bank = None
    if env.cfg.strike_start_prob > 0:
        with np.load(env.cfg.strike_data_file) as source:
            strike_bank = {key: torch.as_tensor(source[key], device=env.device,
                dtype=torch.float32 if source[key].dtype.kind == "f" else None)
                for key in source.files}
    model = env.sim.mj_model
    ids = torch.arange(env.num_envs, device=env.device)
    task = SimpleNamespace(bank=bank, scene_id=ids * 0, candidate_id=ids * 0,
        start_mode=ids * 0, strike_bank=strike_bank,
        strike_reference_id=ids * 0, strike_selected_id=ids * 0,
        strike_hold_pos=torch.zeros((env.num_envs, 3), device=env.device),
        strike_hold_quat=torch.zeros((env.num_envs, 4), device=env.device),
        strike_hold_ctrl=torch.zeros_like(env.sim.data.ctrl),
        seed_pos=torch.zeros((env.num_envs, 3), device=env.device),
        seed_quat=torch.zeros((env.num_envs, 4), device=env.device),
        open_ctrl=torch.zeros(env.num_envs, device=env.device),
        depth=torch.zeros((env.num_envs, 1, 120, 160), device=env.device),
        sampling_weights=((bank["split"] == (env.cfg.split == "eval")).float()),
        selection=None, hammer_id=model.body("hammer_Object").id,
        handle_geom_id=model.geom("hammer_handle").id,
        narrow_driver_id=model.joint("gripper_hand_narrow1_joint").id,
        wide_driver_id=model.joint("gripper_hand_wide1_joint").id,
        mocap_id=int(model.body_mocapid[model.body("gripper_mount").id]),
        base_qadr=int(model.joint("gripper_base_free").qposadr[0]),
        base_vadr=int(model.joint("gripper_base_free").dofadr[0]),
        finger_ids=torch.tensor([model.geom(name).id for name in FINGER_GEOMS], device=env.device),
        nominal_mass=torch.as_tensor(model.body_mass.copy(), dtype=torch.float32, device=env.device),
        nominal_inertia=torch.as_tensor(model.body_inertia.copy(), dtype=torch.float32, device=env.device),
        nominal_friction=torch.as_tensor(model.geom_friction.copy(), dtype=torch.float32, device=env.device),
        nominal_joint_range=torch.as_tensor(model.jnt_range.copy(), dtype=torch.float32, device=env.device))
    task.finger_qadr = torch.tensor([int(model.joint(i).qposadr[0]) for i in range(model.njnt)
                                   if model.joint(i).name.startswith("gripper_hand_")], device=env.device)
    task.finger_vadr = torch.tensor([int(model.joint(i).dofadr[0]) for i in range(model.njnt)
                                   if model.joint(i).name.startswith("gripper_hand_")], device=env.device)
    task.noise = DepthNoise(focal_length=float(bank["intrinsics"][0, 0]), baseline=0.05,
                           min_depth=0.3, max_depth=3.0, filter_size=3).to(env.device)
    env.hammer = task
    if strike_bank is not None:
        task.strike_scenes = ((bank["split"] == 0) &
            (strike_bank["strike_offsets"][1:] > strike_bank["strike_offsets"][:-1])).nonzero().flatten()
    # 构造 observation manager 时只需确定维度；第一次 reset 才写真实初始观测。
    task.public = torch.zeros((env.num_envs, 51), device=env.device)
    task.privileged = torch.zeros((env.num_envs, 88), device=env.device)


@requires_model_fields("geom_friction", "body_mass", "body_inertia", "jnt_range", recompute=RecomputeLevel.set_const)
def randomize_physics(env, env_ids):
    task, model = env.hammer, env.sim.model
    count, body = len(env_ids), task.hammer_id
    scale = torch.empty((count, 1), device=env.device).uniform_(*env.cfg.mass_range)
    model.body_mass[env_ids, body] = task.nominal_mass[body] * scale[:, 0]
    model.body_inertia[env_ids, body] = task.nominal_inertia[body] * scale
    friction = torch.empty((count, 1, 1), device=env.device).uniform_(*env.cfg.friction_range)
    model.geom_friction[env_ids[:, None], task.finger_ids] = task.nominal_friction[task.finger_ids] * friction
    open_delta = torch.empty(count, device=env.device).uniform_(0.0, 0.02)
    close_delta = torch.empty(count, device=env.device).uniform_(0.0, 0.06)
    narrow_base = task.nominal_joint_range[task.narrow_driver_id]
    wide_base = task.nominal_joint_range[task.wide_driver_id]
    model.jnt_range[env_ids, task.narrow_driver_id, 0] = narrow_base[0] - open_delta
    model.jnt_range[env_ids, task.narrow_driver_id, 1] = narrow_base[1] - close_delta
    model.jnt_range[env_ids, task.wide_driver_id, 0] = wide_base[0] + close_delta
    model.jnt_range[env_ids, task.wide_driver_id, 1] = wide_base[1] + open_delta


def public_observation(seed_pos, seed_quat, widths, open_ctrl, *, finger_pos, finger_vel,
                       camera_pos, camera_quat, intrinsics, target_pos, target_quat):
    """仿真缓存和在线公开输入共用的 50 维字段；仅计算张量。"""
    approach = torch.zeros_like(seed_pos)
    approach[:, 2] = 0.10
    pregrasp = seed_pos - quat_apply(seed_quat, approach)
    return torch.cat((seed_pos, seed_quat, widths[:, None], open_ctrl[:, None],
        pregrasp, seed_quat, finger_pos, finger_vel, camera_pos, camera_quat, intrinsics,
        target_pos, target_quat), dim=-1)


def candidate_observations(env, env_ids, candidate):
    """env_ids 可重复，用同一决策前场景批量计算不同候选，不绑定状态。"""
    task = env.hammer
    bank = task.bank
    scene = task.scene_id[env_ids]
    # 使用同条缓存的实际世界初态，避免在 forward 之前读取旧 xpos。
    object_pos, object_quat = bank["hammer_pos"][scene], bank["hammer_quat"][scene]
    seed_pos = object_pos + quat_apply(object_quat, bank["local_pos"][candidate])
    seed_quat = quat_mul(object_quat, bank["local_quat"][candidate])
    return _initial_observations(env, env_ids, candidate, seed_pos, seed_quat,
        object_pos, object_quat,
        bank["qpos"][scene[:, None], task.finger_qadr],
        bank["qvel"][scene[:, None], task.finger_vadr], task.depth[env_ids], mode=0)


def _initial_observations(env, env_ids, candidate, seed_pos, seed_quat,
                          object_pos, object_quat, finger_pos, finger_vel, depth, *, mode):
    task, model = env.hammer, env.sim.model
    bank, count = task.bank, len(env_ids)
    scene = task.scene_id[env_ids]
    public = public_observation(seed_pos, seed_quat, bank["widths"][candidate],
        bank["open_ctrl"][candidate],
        finger_pos=finger_pos, finger_vel=finger_vel,
        **{key: bank[key][scene] for key in
           ("camera_pos", "camera_quat", "intrinsics", "target_pos", "target_quat")})
    # 所有量冻结于本次决策前；不包含执行后的评分。
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
    start_mode = public.new_full((count, 1), mode)
    return TensorDict({"public_seed": torch.cat((public, start_mode), dim=-1),
                       "privileged": torch.cat((privileged, start_mode), dim=-1),
                       "depth": depth}, batch_size=[count])


def bind_candidates(env, env_ids, candidate):
    """绑定决策前候选及完整夹爪初态；不重置场景、深度或随机化参数。"""
    task, data = env.hammer, env.sim.data
    obs = candidate_observations(env, env_ids, candidate)
    public = obs["public_seed"]
    seed_pos, seed_quat, pregrasp = public[:, :3], public[:, 3:7], public[:, 9:12]
    task.candidate_id[env_ids] = candidate
    task.seed_pos[env_ids], task.seed_quat[env_ids] = seed_pos, seed_quat
    task.open_ctrl[env_ids] = public[:, 8]
    task.public[env_ids], task.privileged[env_ids] = public, obs["privileged"]
    data.mocap_pos[env_ids, task.mocap_id] = pregrasp
    data.mocap_quat[env_ids, task.mocap_id] = seed_quat
    qadr, vadr = task.base_qadr, task.base_vadr
    data.qpos[env_ids, qadr:qadr + 3] = pregrasp
    data.qpos[env_ids, qadr + 3:qadr + 7] = seed_quat
    data.qvel[env_ids, vadr:vadr + 6] = 0
    # 保留缓存的开爪手指构型；候选开度由固定预抓段调整。
    scene = task.scene_id[env_ids]
    data.qpos[env_ids[:, None], task.finger_qadr] = task.bank["qpos"][scene[:, None], task.finger_qadr]
    data.qvel[env_ids[:, None], task.finger_vadr] = task.bank["qvel"][scene[:, None], task.finger_vadr]
    return obs


def restore_strike_state(env, env_ids, rows):
    """只恢复仿真数据，保留场景模型和本回合的物理随机化。"""
    task, data = env.hammer, env.sim.data
    for name in STATE_FIELDS:
        getattr(data, name)[env_ids] = task.strike_bank[name][rows]
    data.time[env_ids] = 0
    task.strike_selected_id[env_ids] = rows
    task.strike_hold_pos[env_ids] = data.mocap_pos[env_ids, task.mocap_id]
    task.strike_hold_quat[env_ids] = data.mocap_quat[env_ids, task.mocap_id]
    task.strike_hold_ctrl[env_ids] = data.ctrl[env_ids]


def bind_strike_reference(env, env_ids, rows):
    task, bank = env.hammer, env.hammer.strike_bank
    restore_strike_state(env, env_ids, rows)
    task.strike_reference_id[env_ids] = rows
    candidate = bank["candidate_id"][rows]
    task.candidate_id[env_ids] = candidate
    object_pos, object_quat = bank["hammer_pos"][rows], bank["hammer_quat"][rows]
    seed_pos = object_pos + quat_apply(object_quat, bank["rel_grasp_pos"][rows])
    seed_quat = quat_mul(object_quat, bank["rel_grasp_quat"][rows])
    task.seed_pos[env_ids], task.seed_quat[env_ids] = seed_pos, seed_quat
    task.open_ctrl[env_ids] = task.bank["open_ctrl"][candidate]
    depth = bank["depth"][rows]
    task.depth[env_ids] = task.noise(depth) if env.cfg.visual_noise else depth
    obs = _initial_observations(env, env_ids, candidate, seed_pos, seed_quat,
        object_pos, object_quat,
        bank["qpos"][rows[:, None], task.finger_qadr],
        bank["qvel"][rows[:, None], task.finger_vadr], task.depth[env_ids], mode=1)
    task.public[env_ids], task.privileged[env_ids] = obs["public_seed"], obs["privileged"]


@requires_model_fields("body_pos", "body_quat", "geom_size", "geom_rbound", "geom_aabb",
                       recompute=RecomputeLevel.set_const_0)
def reset_scene(env, env_ids):
    task, data, model = env.hammer, env.sim.data, env.sim.model
    bank, count = task.bank, len(env_ids)
    task.start_mode[env_ids] = 0
    if task.selection is None:
        scene = torch.multinomial(task.sampling_weights, count, replacement=True)
        start = bank["offsets"][scene]
        candidate = start + (torch.rand(count, device=env.device) *
                            (bank["offsets"][scene + 1] - start)).long()
        if env.cfg.strike_start_prob > 0:
            is_strike = torch.rand(count, device=env.device) < env.cfg.strike_start_prob
            task.start_mode[env_ids] = is_strike.long()
            scene[is_strike] = task.strike_scenes[torch.randint(
                len(task.strike_scenes), (int(is_strike.sum()),), device=env.device)]
    else:
        scene, candidate = task.selection
        scene, candidate = scene[env_ids], candidate[env_ids]
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
    grasp_ids = env_ids[task.start_mode[env_ids] == 0]
    if len(grasp_ids):
        depth = bank["depth"][task.scene_id[grasp_ids]]
        task.depth[grasp_ids] = task.noise(depth) if env.cfg.visual_noise else depth
        bind_candidates(env, grasp_ids, candidate[task.start_mode[env_ids] == 0])
    strike_ids = env_ids[task.start_mode[env_ids] == 1]
    if len(strike_ids):
        strike_scene = task.scene_id[strike_ids]
        offsets = task.strike_bank["strike_offsets"]
        start = offsets[strike_scene]
        rows = start + (torch.rand(len(strike_ids), device=env.device) *
                        (offsets[strike_scene + 1] - start)).long()
        bind_strike_reference(env, strike_ids, rows)
