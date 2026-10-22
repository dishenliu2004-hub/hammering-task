"""原夹爪位姿控制与搬运、敲击轨迹；物理由 manager-based 环境推进。"""

from dataclasses import dataclass
from functools import partial
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase

from generation.knock_path import hammer_strike as run_hammer_strike

from .observations import sync_data


@dataclass(kw_only=True)
class MocapActionCfg(ActionTermCfg):
    def build(self, env):
        return MocapAction(self, env)


class MocapAction(ActionTerm):
    def __init__(self, cfg, env):
        # 完整 MJCF 通过 SceneCfg.spec_fn 挂接，直接控制场景中的 mocap 和 ctrl。
        ManagerTermBase.__init__(self, env)
        self.cfg = cfg
        model = env.sim.mj_model
        self._mocap_id = model.body_mocapid[model.body(cfg.entity_name).id]
        self._actuator_id = model.actuator("gripper_gripper_actuator").id
        self._raw_action = torch.zeros((env.num_envs, 8), dtype=torch.float64, device=env.device)
        self.on_physics_step = None
        self.impact_peak = 0.0

    @property
    def action_dim(self):
        return 8

    @property
    def raw_action(self):
        return self._raw_action

    def reset(self, env_ids):
        self._raw_action[env_ids] = 0
        self._substep = 0
        self.on_physics_step = None
        self.impact_peak = 0.0

    def process_actions(self, actions):
        self._raw_action.copy_(actions)
        state = self._env.sim.data
        self._start_pos = state.mocap_pos[0, self._mocap_id].cpu().numpy().copy()
        quat = state.mocap_quat[0, self._mocap_id].cpu().numpy()
        self._start_rot = Rotation.from_quat(quat[[1, 2, 3, 0]])
        target = actions[0].cpu().numpy()
        self._delta_pos = target[:3] - self._start_pos
        target_rot = Rotation.from_quat(target[3:7][[1, 2, 3, 0]])
        self._delta_rot = (target_rot * self._start_rot.inv()).as_rotvec()
        state.ctrl[0, self._actuator_id] = actions[0, 7]
        self._substep = 0

    def apply_actions(self):
        self._substep += 1
        u = self._substep / self._env.cfg.decimation
        pos = self._start_pos + u * self._delta_pos
        rot = Rotation.from_rotvec(u * self._delta_rot) * self._start_rot
        state = self._env.sim.data
        state.mocap_pos[0, self._mocap_id] = torch.as_tensor(
            pos, dtype=state.mocap_pos.dtype, device=self.device,
        )
        state.mocap_quat[0, self._mocap_id] = torch.as_tensor(
            rot.as_quat()[[3, 0, 1, 2]], dtype=state.mocap_quat.dtype, device=self.device,
        )


def impact_peak_force(env):
    term = env.action_manager.get_term("mocap")
    if term.on_physics_step is not None:
        term.impact_peak = term.on_physics_step()
    return torch.full((env.num_envs,), term.impact_peak, dtype=torch.float64, device=env.device)


def step_mocap(env, data, pos, quat, gripper_action, dt=0.02, viewer=None, on_step=None,
               *, on_physics_step=None):
    action = torch.as_tensor(
        np.concatenate((pos, quat, [gripper_action])),
        dtype=torch.float64, device=env.device,
    ).unsqueeze(0)
    term = env.action_manager.get_term("mocap")
    decimation = env.cfg.decimation
    env.cfg.decimation = round(dt / env.sim.mj_model.opt.timestep)
    term.on_physics_step = on_physics_step
    try:
        env.step(action)
    finally:
        env.cfg.decimation = decimation
        term.on_physics_step = None
    sync_data(env, data)
    if viewer is not None:
        viewer.sync()
        time.sleep(dt)
    if on_step is not None:
        on_step()


def move_to(env, data, viewer, target_pos, target_quat, duration, grip, dt=0.02, on_step=None):
    model = env.sim.mj_model
    mocap_id = model.body_mocapid[model.body("gripper_mount").id]
    start_pos = data.mocap_pos[mocap_id].copy()
    start_rot = Rotation.from_quat(data.mocap_quat[mocap_id][[1, 2, 3, 0]])
    target_rot = Rotation.from_quat(np.asarray(target_quat)[[1, 2, 3, 0]])
    rot_delta = (target_rot * start_rot.inv()).as_rotvec()
    steps = round(duration / dt)

    for i in range(steps):
        u = (i + 1) / steps
        s = u * u * (3.0 - 2.0 * u)
        pos = start_pos + s * (target_pos - start_pos)
        rot = Rotation.from_rotvec(s * rot_delta) * start_rot
        step_mocap(env, data, pos, rot.as_quat()[[3, 0, 1, 2]], grip, dt, viewer, on_step)


def hammer_strike(
    env,
    data,
    target_pos,
    hammer_tip_offset,
    impact_quat,
    gripper_action=1.0,
    dt=0.02,
    viewer=None,
    on_step=None,
    *,
    placement=None,
    hammer_head_center_offset=None,
    on_swing_step=None,
    on_physics_step=None,
):
    model = env.sim.mj_model
    mount = data.body("gripper_base_link")
    return run_hammer_strike(
        model, data, mount.xpos.copy(), mount.xquat.copy(),
        target_pos, hammer_tip_offset, impact_quat,
        gripper_action, dt, viewer, on_step,
        simulation=env.sim, placement=placement,
        hammer_head_center_offset=hammer_head_center_offset,
        on_swing_step=on_swing_step, on_physics_step=on_physics_step,
        step_fn=partial(step_mocap, env, data),
    )
