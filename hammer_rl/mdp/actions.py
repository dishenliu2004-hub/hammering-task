"""一次 seed 局部 6D 残差，以及 knock_path.py 固定规则的 GPU 批量表达。"""

from dataclasses import dataclass
from functools import partial
import math

import numpy as np
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import (axis_angle_from_quat, quat_apply, quat_apply_inverse,
    quat_box_plus, quat_conjugate, quat_error_magnitude, quat_mul)

from generation.enclosure import build_enclosure_geometry, compute_enclosure_score, compute_handle_seating
from generation.hammer_geometry import HEAD_CENTER_LOCAL, STRIKE_FACE_CENTER_LOCAL
from .resets import initialize_task

DT = 0.02
PRE_SWING_MOVE_DURATIONS = (1.0, 0.6, 0.50, 0.40, 0.50, 1.5, 1.5, 0.60)
PRE_SWING_HOLD_DURATION = 0.5


def swing_schedule(physics_dt):
    # 仅构造时使用 NumPy；与原脚本完全相同的 1001 点弧长表。
    s = np.linspace(0.0, 1.0, 1001)
    points = np.column_stack((0.40 * s, 0.30 * (1 - s * s) + 0.10 * s * (1 - s)))
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    duration = math.sqrt(2.0 * length[-1] / 10.0)
    steps = round(duration / DT)
    return s, length, duration, steps, round(duration / steps / physics_dt)


def _pre_swing_substeps(physics_dt):
    move_substeps = sum(round(duration / physics_dt) for duration in PRE_SWING_MOVE_DURATIONS)
    return move_substeps + round(PRE_SWING_HOLD_DURATION / physics_dt)


def attempt_substeps(physics_dt, unload_time, num_strikes=3, recover_time=0.8):
    *_, steps, substeps = swing_schedule(physics_dt)
    swing_len = steps * substeps
    recover_len = round(recover_time / physics_dt)
    return (
        _pre_swing_substeps(physics_dt)
        + num_strikes * swing_len
        + (num_strikes - 1) * recover_len
        + round(unload_time / physics_dt)
        + round(0.45 / DT) * round(DT / physics_dt)
    )


def _rotation_delta(start, end):
    return axis_angle_from_quat(quat_mul(end, quat_conjugate(start)))


def _rotate_by(quat, delta):
    return quat_box_plus(quat.expand(*delta.shape[:-1], 4), delta)


def interpolate_quat(start, end, s):
    return _rotate_by(start, _rotation_delta(start, end) * s)


def sampled_progress(substep, duration, substeps, quintic=False):
    """原 DT 外层平滑采样，内层物理子步线性插值。"""
    index = (substep - 1) // substeps
    fraction = ((substep - 1) % substeps + 1) / substeps
    count = round(duration / DT)
    u0, u1 = index / count, (index + 1) / count
    def smooth(u):
        return 10 * u**3 - 15 * u**4 + 6 * u**5 if quintic else u * u * (3 - 2 * u)
    return smooth(u0) + fraction * (smooth(u1) - smooth(u0))


@dataclass(kw_only=True)
class GraspResidualActionCfg(ActionTermCfg):
    position_scale: float = 0.10
    rotation_scale: float = math.pi / 5
    unload_time: float = 1.0
    num_strikes: int = 3
    recover_time: float = 0.8

    def build(self, env):
        return GraspResidualAction(self, env)


def refine_pose(seed_pos, seed_quat, actions, cfg):
    """训练动作项和候选推理共用的局部残差合成。"""
    delta = torch.tanh(actions)
    identity = torch.zeros_like(seed_quat)
    identity[:, 0] = 1
    final_pos = seed_pos + quat_apply(seed_quat, delta[:, :3] * cfg.position_scale)
    local_rotation = quat_box_plus(identity, delta[:, 3:] * cfg.rotation_scale)
    final_quat = quat_mul(seed_quat, local_rotation)
    approach = torch.zeros_like(seed_pos)
    approach[:, 2] = 1
    pregrasp = final_pos - 0.10 * quat_apply(final_quat, approach)
    return final_pos, final_quat, pregrasp


class GraspResidualAction(ActionTerm):
    def __init__(self, cfg, env):
        # 整场景 spec_fn，无虚构 Entity 或机械臂。
        ManagerTermBase.__init__(self, env)
        self.cfg = cfg
        initialize_task(env)
        self.task = env.hammer
        model = env.sim.mj_model
        self.mocap = self.task.mocap_id
        self.base = model.body("gripper_base_link").id
        self.hammer = self.task.hammer_id
        self.enclosure_geometry = build_enclosure_geometry(model, self.device)
        self.narrow = model.site("gripper_hand_narrow_contact_tip").id
        self.wide = model.site("gripper_hand_wide_contact_tip").id
        self.target = model.site("cylinder_endface_force_frame").id
        self.weld = model.equality("gripper_target_weld").id
        self.actuator = model.actuator("gripper_gripper_actuator").id
        self.touch_address = int(model.sensor("cylinder_endface_normal_force").adr[0])
        self.moving = torch.tensor(np.flatnonzero(np.isin(model.body_rootid,
            [model.body("gripper_dynamic_base").id, self.hammer])), device=self.device)
        self.gravity = torch.tensor(model.opt.gravity.copy(), dtype=torch.float32, device=self.device)
        self._raw_action = torch.zeros((env.num_envs, 6), device=self.device)
        self.head_local = self._raw_action.new_tensor(HEAD_CENTER_LOCAL).expand(env.num_envs, -1)
        self.face_local = self._raw_action.new_tensor(STRIKE_FACE_CENTER_LOCAL).expand(env.num_envs, -1)
        self.unit = torch.eye(3, device=self.device).expand(env.num_envs, -1, -1)
        self.identity = torch.zeros((env.num_envs, 4), device=self.device)
        self.identity[:, 0] = 1
        self.substeps = round(DT / env.physics_dt)
        s, length, self.swing_duration, self.swing_steps, self.swing_substeps = swing_schedule(env.physics_dt)
        self.path_s = torch.tensor(s, dtype=torch.float32, device=self.device)
        self.path_length = torch.tensor(length, dtype=torch.float32, device=self.device)
        self.enclosure_sample_steps = round(1 / (250 * env.physics_dt))
        self.palm_frames = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.seat_frames = torch.zeros_like(self.palm_frames)
        self.closure_enclosure = torch.zeros(env.num_envs, device=self.device)
        self.swing_start = _pre_swing_substeps(env.physics_dt)
        self.swing_len = self.swing_steps * self.swing_substeps
        self.recovery_steps = round(cfg.recover_time / env.physics_dt)
        self.cycle_steps = self.swing_len + self.recovery_steps
        self.swing_end = (
            self.swing_start
            + cfg.num_strikes * self.swing_len
            + (cfg.num_strikes - 1) * self.recovery_steps
        )
        self.unload_end = self.swing_end + round(cfg.unload_time / env.physics_dt)
        self.return_steps = round(0.45 / DT) * self.substeps
        self.on_physics_step = None  # 仅评估安装，训练路径没有 CPU 同步。
        self._substep = 0
        self._world_mask = None
        self.values = {name: torch.zeros(env.num_envs, device=self.device) for name in (
            "capture", "lift_progress", "lift_success", "wrap", "orientation", "distance", "stability", "force",
            "impact_impulse", "task_success", "drop",
            "translation_drift", "rotation_drift", "mean_force",
            "valid_strike", "lift_height", "grasp_before_swing", "hold_fraction",
            "seat_closure", "seat_lift", "seat_before_swing", "peak_force",
            "hit_progress", "recovery_hold", "hit_count",
            "strike_force_1", "strike_force_2", "strike_force_3")}
        if env.cfg.reward_recipe is not None:
            prior = env.cfg.reward_recipe["mapping_prior"]
            self.prior_u = self._raw_action.new_tensor([prior["u_mean_RL"], prior["u_sigma_RL"]])
            self.prior_axis = self._raw_action.new_tensor(prior["closing_axis_O"])
            self.values.update({name: torch.zeros(env.num_envs, device=self.device)
                                for name in ("prior_region", "prior_axis", "grasp_u_rl", "grasp_axis_cos2")})
        self.grasp_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.relative_closure_pos = torch.zeros((env.num_envs, 3), device=self.device)
        self.relative_closure_quat = self.identity.clone()
        self.relative_a_pos = torch.empty_like(self.relative_closure_pos)
        self.relative_a_quat = torch.empty_like(self.identity)
        self.swing_observed = torch.zeros_like(self.grasp_valid)
        self.dropped = torch.zeros_like(self.grasp_valid)
        self.lost_contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.hold_steps = torch.zeros_like(self.lost_contact_steps)
        self.strike_hits = torch.zeros(
            (env.num_envs, cfg.num_strikes), dtype=torch.bool, device=self.device)
        self.strike_first_hit_step = torch.zeros(
            (env.num_envs, cfg.num_strikes), dtype=torch.long, device=self.device)
        self.strike_impulse = torch.zeros((env.num_envs, cfg.num_strikes), device=self.device)
        self.recovery_ok = torch.zeros(
            (env.num_envs, cfg.num_strikes - 1), dtype=torch.bool, device=self.device)
        self.impact_started = torch.zeros_like(self.grasp_valid)
        self.finished = torch.zeros_like(self.grasp_valid)
        self.returning = torch.zeros_like(self.grasp_valid)
        self.unloaded = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.return_start = torch.zeros_like(self.unloaded)
        self.return_pos = torch.zeros((env.num_envs, 3), device=self.device)
        self.return_quat = self.identity.clone()
        self.final_pos = torch.empty_like(self.return_pos)
        self.final_quat = torch.empty_like(self.identity)
        self.pregrasp = torch.empty_like(self.return_pos)
        self.start_pos = torch.empty_like(self.return_pos)
        self.start_quat = torch.empty_like(self.identity)
        self.lifted_pos = torch.empty_like(self.return_pos)
        # 跨物理步的规划结果必须在图外分配，供整次尝试原地更新。
        self.head_offset = torch.empty_like(self.start_pos)
        self.grip_offset = torch.empty_like(self.start_pos)
        self.normal = torch.empty_like(self.start_pos)
        self.swing_axis = torch.empty_like(self.start_pos)
        self.horizontal = torch.empty_like(self.start_pos)
        self.grip_impact = torch.empty_like(self.start_pos)
        self.strike_pos = torch.empty_like(self.start_pos)
        self.impact_quat = torch.empty_like(self.start_quat)
        self.raised_quat = torch.empty_like(self.start_quat)
        self._carry_segments = [(torch.empty_like(self.start_pos), torch.empty_like(self.start_pos),
                                 torch.empty_like(self.start_quat), torch.empty_like(self.start_pos))
                                for _ in range(4)]
        self.controls, self.samples = self._schedule()

    @property
    def action_dim(self):
        return 6

    @property
    def raw_action(self):
        return self._raw_action

    def reset(self, env_ids):
        self._raw_action[env_ids] = 0
        for value in self.values.values():
            value[env_ids] = 0
        for value in (self.grasp_valid, self.swing_observed, self.dropped, self.lost_contact_steps, self.hold_steps,
                      self.strike_hits, self.strike_first_hit_step, self.strike_impulse, self.recovery_ok,
                      self.impact_started, self.finished, self.returning,
                      self.unloaded, self.return_start, self.palm_frames, self.seat_frames, self.closure_enclosure):
            value[env_ids] = 0
        # 不清除 reset 事件刚写入的场景及候选观测，也不推进其他环境的调度时钟。

    def process_actions(self, actions, env_ids=slice(None)):
        self._raw_action[env_ids] = actions
        self._substep = 0
        pos, quat, pregrasp = refine_pose(
            self.task.seed_pos[env_ids], self.task.seed_quat[env_ids], actions, self.cfg)
        data = self._env.sim.data
        self.final_pos[env_ids] = pos
        self.final_quat[env_ids] = quat
        self.pregrasp[env_ids] = pregrasp
        self.start_pos[env_ids] = data.mocap_pos[env_ids, self.mocap]
        self.start_quat[env_ids] = data.mocap_quat[env_ids, self.mocap]
        self.lifted_pos[env_ids] = pos + 0.15 * self.unit[env_ids, 2]

    def _store(self, dest, value):
        """训练图内只写当前阶段的世界；同步回放时整批写入。"""
        if self._world_mask is not None:
            mask = self._world_mask.view(-1, *([1] * (dest.ndim - 1)))
            value = torch.where(mask, value, dest)
        dest[...] = value

    def _write(self, pos, quat, grip=1.0):
        data = self._env.sim.data
        self._store(data.mocap_pos[:, self.mocap], pos)
        self._store(data.mocap_quat[:, self.mocap], quat)
        self._store(data.ctrl[:, self.actuator], grip)

    def _relative_pose(self):
        data = self._env.sim.data
        q = data.xquat[:, self.base]
        return (quat_apply_inverse(q, data.xpos[:, self.hammer] - data.xpos[:, self.base]),
                quat_mul(quat_conjugate(q), data.xquat[:, self.hammer]))

    def _plan_strike(self):
        data = self._env.sim.data
        mount_pos, mount_quat = data.xpos[:, self.base], data.xquat[:, self.base]
        hammer_pos, hammer_quat = data.xpos[:, self.hammer], data.xquat[:, self.hammer]
        face = hammer_pos + quat_apply(hammer_quat, self.face_local)
        tip_offset = quat_apply_inverse(mount_quat, face - mount_pos)
        self._store(self.head_offset, quat_apply_inverse(mount_quat,
            hammer_pos + quat_apply(hammer_quat, self.head_local) - mount_pos))
        center = (data.site_xpos[:, self.narrow] + data.site_xpos[:, self.wide]) / 2
        self._store(self.grip_offset, quat_apply_inverse(mount_quat, center - mount_pos))
        target_pos = data.site_xpos[:, self.target]
        frame = data.site_xmat[:, self.target].reshape(-1, 3, 3)
        self._store(self.normal, frame[:, :, 2])
        self._store(self.swing_axis, frame[:, :, 1])
        self._store(self.horizontal, torch.nn.functional.normalize(torch.cross(self.swing_axis, self.unit[:, 2], dim=-1), dim=-1))
        target_quat = self.task.bank["target_quat"][self.task.scene_id]
        impact_hammer = quat_mul(target_quat, quat_box_plus(self.identity, math.pi * self.unit[:, 1]))
        self._store(self.impact_quat, quat_mul(impact_hammer, quat_conjugate(self.relative_a_quat)))
        self._store(self.raised_quat, quat_box_plus(self.impact_quat, -math.pi / 3 * self.swing_axis))
        pivot = target_pos - quat_apply(self.impact_quat, tip_offset)
        self._store(self.grip_impact, pivot + quat_apply(self.impact_quat, self.grip_offset))
        head_impact = pivot + quat_apply(self.impact_quat, self.head_offset)
        carry_quat = data.mocap_quat[:, self.mocap].clone()
        head_start = data.mocap_pos[:, self.mocap] + quat_apply(carry_quat, self.head_offset)
        high_start, high_target = head_start.clone(), head_impact.clone()
        high_start[:, 2] = high_target[:, 2] = head_start[:, 2] + 0.20
        self._store(self.strike_pos, self._swing_pose(0.0)[0])
        head_strike = self.strike_pos + quat_apply(self.raised_quat, self.head_offset)
        segments = (
            (head_start, high_start, carry_quat, carry_quat),
            (high_start, high_target, carry_quat, carry_quat),
            (high_target, high_target, carry_quat, self.raised_quat),
            (high_target, head_strike, self.raised_quat, self.raised_quat),
        )
        for (pos, dp, quat, dr), (p0, p1, q0, q1) in zip(self._carry_segments, segments):
            self._store(pos, p0)
            self._store(dp, p1 - p0)
            self._store(quat, q0)
            self._store(dr, _rotation_delta(q0, q1))

    def _swing_pose(self, s):
        angle = -math.pi / 3 * (1 - s)
        height = 0.30 * (1 - s * s) + 0.10 * s * (1 - s)
        distance = 0.40 * (1 - s)
        if isinstance(s, torch.Tensor):
            angle, height, distance = angle.float(), height.float(), distance.float()
        quat = _rotate_by(self.impact_quat, angle * self.swing_axis)
        center = self.grip_impact + height * self.unit[:, 2] - distance * self.horizontal
        return center - quat_apply(quat, self.grip_offset.expand(*quat.shape[:-1], 3)), quat

    def _swing_progress(self, index):
        distance = 5.0 * (index * self.swing_duration / self.swing_steps)**2
        d = torch.full((1,), distance, device=self.device)
        i = torch.searchsorted(self.path_length, d).clamp(1, 1000)
        fraction = (d - self.path_length[i - 1]) / (self.path_length[i] - self.path_length[i - 1])
        return (self.path_s[i - 1] + fraction * (self.path_s[i] - self.path_s[i - 1]))[0]

    def _schedule(self):
        """固定时点只计算一次；GPU 采集和逐帧回放共享时序和计算。"""
        k = np.arange(self._env.cfg.decimation + 1)
        progress = np.zeros_like(k, dtype=float)
        controls, samples = [], []
        move_ends = []
        start = 0
        for i, duration in enumerate(PRE_SWING_MOVE_DURATIONS):
            end = start + round(duration / self._env.physics_dt)
            active = (start < k) & (k <= end)
            progress[active] = sampled_progress(k[active] - start, duration, self.substeps, quintic=i >= 4)
            controls.append((active, partial(self._move_segment, segment=i), self.task.open_ctrl if i < 2 else 1.0))
            move_ends.append(end)
            start = end
        self.closure_start, self.closure_end, self.lift_end = move_ends[1:4]
        controls.append(((start < k) & (k <= self.swing_start), self._hold_raised, 1.0))
        phase = (k - self.swing_start - 1) % self.cycle_steps
        strikes = (self.swing_start < k) & (k <= self.swing_end)
        swing = strikes & (phase < self.swing_len)
        progress[swing] = (phase[swing] % self.swing_substeps + 1) / self.swing_substeps
        self.swing_index = torch.as_tensor(np.where(swing, phase // self.swing_substeps, 0), device=self.device)
        self.swing_parameters = torch.stack([self._swing_progress(i) for i in range(self.swing_steps + 1)])
        controls.append((swing, self._swing, 1.0))
        recovery = strikes & ~swing
        u = (phase[recovery] - self.swing_len + 1) / self.recovery_steps
        progress[recovery] = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
        self.progress = torch.tensor(progress, dtype=torch.float64, device=self.device)
        controls.append((recovery, self._recover, 1.0))
        controls.append((k > self.swing_end, self._withdraw, 1.0))

        spacing = self.enclosure_sample_steps
        snapshots = (self.closure_end, self.lift_end, self.swing_start)
        samples.append((np.isin(k, snapshots), self._refresh))
        samples.append(((self.closure_start <= k) & (k <= self.closure_end)
                        & ((k - self.closure_start) % spacing == 0), self._palm_contact))
        starts = np.array(snapshots) - 4 * spacing
        samples.append((np.isin(k, starts), lambda k: self._store(self.seat_frames, 0)))
        seat_ticks = starts[:, None] + spacing * np.arange(5)
        samples.append((np.isin(k, seat_ticks), self._seat_contact))
        for end, name in zip(snapshots, ("seat_closure", "seat_lift", "seat_before_swing")):
            samples.append((k == end, lambda k, name=name: self._store(self.values[name], self.seat_frames >= 5)))
        samples.append(((k == self.closure_end) | ((self.lift_end <= k) & (k <= self.unload_end)),
                        self._handle_contact))
        samples.append((k == self.closure_end, self._closure_snapshot))
        samples.append(((self.lift_end < k) & (k <= self.unload_end), self._track_drop))
        samples.append(((self.lift_end < k) & (k <= self.swing_start),
                        lambda k: self._store(self.hold_steps, self.hold_steps + (self.bilateral & ~self.dropped))))
        samples.append((k == self.lift_end, self._lift_snapshot))
        samples.append((k == self.swing_start, self._swing_snapshot))
        samples.append(((self.swing_start < k) & (k <= self.unload_end), self._head_contact))
        for i in range(self.cfg.num_strikes):
            start = self.swing_start + i * self.cycle_steps
            last = i == self.cfg.num_strikes - 1
            end = self.unload_end if last else start + self.cycle_steps
            if not last:
                samples.append((k == end, partial(self._recovery_snapshot, i=i)))
            samples.append(((start < k) & (k <= end), partial(self._impact, strike_id=i)))
        samples.append(((self.swing_end < k) & (k <= self.unload_end), self._unload))
        return controls, samples

    def _move_segment(self, k, segment):
        s = self.progress[k].float()
        if segment < 4:
            points = (self.start_pos, self.pregrasp, self.final_pos, self.final_pos, self.lifted_pos)
            q0 = self.start_quat if segment == 0 else self.final_quat
            p0 = points[segment]
            return p0 + s * (points[segment + 1] - p0), interpolate_quat(q0, self.final_quat, s)
        p0, delta_pos, q0, delta_rot = self._carry_segments[segment - 4]
        q = _rotate_by(q0, delta_rot * s)
        return p0 + s * delta_pos - quat_apply(q, self.head_offset.expand(*q.shape[:-1], 3)), q

    def _hold_raised(self, k):
        return self.strike_pos, self.raised_quat

    def _swing(self, k):
        index = self.swing_index[k]
        p0, q0 = self._swing_pose(self.swing_parameters[index])
        p1, q1 = self._swing_pose(self.swing_parameters[index + 1])
        u = self.progress[k].float()
        return p0 + u * (p1 - p0), interpolate_quat(q0, q1, u)

    def _recover(self, k):
        return self._swing_pose(self.progress[k])

    def _withdraw(self, k):
        data = self._env.sim.data
        # 未命中也在固定时窗内正常结束。不同环境可以提前完成卸载及撤离。
        self._store(self.finished, self.finished | (k > self.unload_end))
        new_return = self.finished & ~self.returning
        self._store(self.return_start, torch.where(new_return, k - 1, self.return_start))
        adr = self.task.base_qadr
        self._store(self.return_pos, torch.where(new_return[:, None], data.qpos[:, adr:adr + 3], self.return_pos))
        self._store(self.return_quat, torch.where(new_return[:, None], data.qpos[:, adr + 3:adr + 7], self.return_quat))
        self._store(self.returning, self.returning | new_return)
        progress = (k - self.return_start).clamp(0, self.return_steps)
        s = sampled_progress(progress, round(0.45 / DT) * DT, self.substeps)
        p = self.return_pos + 0.03 * s[:, None] * self.normal
        self._store(data.eq_active[:, self.weld], self.returning)
        masses = (self._env.sim.model.body_mass[:, self.moving] if self._env.cfg.randomize_physics
                  else self.task.nominal_mass[self.moving].expand(self.num_envs, -1))
        compensation = -masses[:, :, None] * self.gravity
        force = compensation * (~self.returning)[:, None, None]
        if self._world_mask is not None:
            force = torch.where(self._world_mask[:, None, None], force, data.xfrc_applied[:, self.moving, :3])
        data.xfrc_applied[:, self.moving, :3] = force
        return (torch.where(self.returning[:, None], p, data.mocap_pos[:, self.mocap]),
                torch.where(self.returning[:, None], self.return_quat, data.mocap_quat[:, self.mocap]))

    def apply_actions(self):
        self._substep += 1
        for active, pose, grip in self.controls:
            if active[self._substep]:
                self._write(*pose(self._substep), grip)

    def _refresh(self, k):
        self._env.sim.forward()

    def _palm_contact(self, k):
        contact = self._env.scene.sensors["palm_contact"].data
        force = contact.force[:, :, 0].abs()
        strong = ((contact.found > 0) & ((force >= 0.1) | (force / 250 >= 0.0001))).any(-1)
        self._store(self.palm_frames, torch.where(strong, self.palm_frames + 1, 0))

    def _seat_contact(self, k):
        seated = compute_handle_seating(self._env.scene.sensors["handle_seat_contact"].data,
                                       self._env.sim.data, self.enclosure_geometry)
        self._store(self.seat_frames, torch.where(seated, self.seat_frames + 1, 0))

    def _handle_contact(self, k):
        contact = self._env.scene.sensors["handle_contact"].data
        # handle_contact 只检测六个指面与锤柄的接触，每个指面最多保留两个接触槽。
        # 12 个槽中前 6 个属于 narrow 侧，后 6 个属于 wide 侧；槽内有接触且法向力 > 0
        # 才算有效。双侧各至少一个有效接触，才能通过 bilateral 判据。
        valid = (contact.found > 0) & (contact.force[:, :, 0] > 0)
        self.left, self.right = valid[:, :6].any(-1), valid[:, 6:].any(-1)
        self.bilateral = self.left & self.right

    def _closure_snapshot(self, k):
        self._store(self.closure_enclosure, compute_enclosure_score(self._env.sim.data, self.enclosure_geometry,
            self.palm_frames >= 5, self.task.bank["handle_convex_planes"][self.task.scene_id]))
        self._store(self.values["capture"], (self.left.float() + self.right.float()) / 2)
        p, q = self._relative_pose()
        self._store(self.relative_closure_pos, p)
        self._store(self.relative_closure_quat, q)

    def _track_drop(self, k):
        values, bilateral = self.values, self.bilateral
        # 从抬升后评价；连续失联 20 ms 锁存掉落。
        evaluating = ~self.returning
        lost = (values["lift_success"] > 0) & ~bilateral
        self._store(self.lost_contact_steps, torch.where(evaluating,
            torch.where(lost, self.lost_contact_steps + 1, 0), self.lost_contact_steps))
        self._store(self.dropped, self.dropped | (evaluating & (self.lost_contact_steps >= self.substeps)))
        self._store(values["drop"], self.dropped)

    def _lift_snapshot(self, k):
        data, values, bilateral = self._env.sim.data, self.values, self.bilateral
        # 抬升结束时写入 A 快照的抬升和几何评分，
        # wrap 在挥击前追加顶靠奖励；保存夹爪系下的锤子位姿，用于规划后续挥击轨迹。
        p, q = self._relative_pose()
        self._store(self.relative_a_pos, p)
        self._store(self.relative_a_quat, q)
        initial = self.task.bank["hammer_pos"][self.task.scene_id, 2]
        self._store(values["lift_height"], data.xpos[:, self.hammer, 2] - initial)
        # 有效抬升 lift：锤子相对本场景初态的真实高度增量 >= lift_threshold，且双侧接触。
        # lift_threshold 当前为 0.15 m；lift_success 是 0/1 日志值，也作为奖励的有效性门控。
        # 抬升失败时几何分为零；有双侧接触时仍可获得部分抬升进展分。
        lift = (values["lift_height"] >= self._env.cfg.lift_threshold) & bilateral
        self._store(values["lift_success"], lift)
        self._store(values["lift_progress"], (values["lift_height"] / self._env.cfg.lift_threshold).clamp(0, 1)
                                      * bilateral)

        # 1. 包覆奖励：抬升成功后，真包络角度评分放大 2 倍，加闭合/抬升顶靠奖励。
        self._store(values["wrap"], torch.where(lift, 2 * self.closure_enclosure
                             + 0.10 * values["seat_closure"] + 0.15 * values["seat_lift"], 0) / 1.5)

        # 横向夹柄；按初态柄轴接近水平的程度，连续加权双向侧抓偏好。
        axis = torch.nn.functional.normalize(data.site_xpos[:, self.wide] - data.site_xpos[:, self.narrow], dim=-1)
        handle_axis = quat_apply(data.xquat[:, self.hammer], self.unit[:, 0])
        transverse = (1 - (axis * handle_axis).sum(-1).square()).clamp(0, 1)
        initial_axis = quat_apply(self.task.bank["hammer_quat"][self.task.scene_id], self.unit[:, 0])
        side = torch.nn.functional.normalize(torch.cross(self.unit[:, 2], initial_axis, dim=-1), dim=-1)
        approach = quat_apply(self.final_quat, self.unit[:, 2])
        side_score = (approach * side).sum(-1).square().clamp(0, 1)
        # 初态倾角从 0° 到 30° 时，平躺权重从 1 衰减至 0。
        flatness = (1.0 - initial_axis[:, 2].abs() / math.sin(math.pi / 6)).clamp(0.0, 1.0)
        self._store(values["orientation"], ((1.0 - 0.6 * flatness) * transverse
                                     + 0.6 * flatness * side_score) * lift)

        # 沿柄轴投影评分。
        center = (data.site_xpos[:, self.wide] + data.site_xpos[:, self.narrow]) / 2
        if self._env.cfg.reward_recipe is not None:
            center_o = quat_apply_inverse(data.xquat[:, self.hammer], center - data.xpos[:, self.hammer])
            u = center_o[:, 0] / self.task.bank["handle_size"][self.task.scene_id, 1]
            axis_o = quat_apply_inverse(data.xquat[:, self.hammer], axis)
            self._store(values["prior_region"], torch.exp(-0.5 * ((u - self.prior_u[0]) / self.prior_u[1]).square()) * lift)
            cos2 = (axis_o * self.prior_axis).sum(-1).square()
            self._store(values["prior_axis"], cos2 * lift)
            self._store(values["grasp_u_rl"], u)
            self._store(values["grasp_axis_cos2"], cos2.clamp(0, 1))
        head = data.xpos[:, self.hammer] + quat_apply(data.xquat[:, self.hammer], self.head_local)
        distance = ((center - head) * handle_axis).sum(-1)
        self._store(values["distance"], (distance / self._env.cfg.distance_ref).clamp(0, 1) * lift)
        # 按 A 快照规划后续固定挥击轨迹，同时准备后面投影接触力所需的目标端面法向。
        self._plan_strike()

    def _swing_snapshot(self, k):
        values, bilateral = self.values, self.bilateral
        # 挥击前快照需已有效抬升、未锁存掉落且当前双侧有效接触。
        self._store(values["hold_fraction"], self.hold_steps.float() / (self.swing_start - self.lift_end))
        self._store(self.grasp_valid, (values["lift_success"] > 0) & bilateral & ~self.dropped)
        self._store(values["grasp_before_swing"], self.grasp_valid)
        self._store(values["wrap"], values["wrap"] + 0.25 * self.grasp_valid * values["seat_before_swing"] / 1.5)
        if self._env.cfg.reward_recipe is not None:
            self._store(self.swing_observed, True)
        # 比较闭合结束与挥击前的相对位姿。
        p, q = self._relative_pose()
        self._store(values["translation_drift"], (p - self.relative_closure_pos).norm(dim=-1))
        self._store(values["rotation_drift"], quat_error_magnitude(self.relative_closure_quat, q))
        x = torch.sqrt(0.5 * ((values["translation_drift"] / self._env.cfg.stability_position_scale).square()
            + (values["rotation_drift"] / self._env.cfg.stability_rotation_scale).square()))
        b, c = math.sqrt(26.0) / 3.0, math.sqrt(10.0)
        # 三个给分节点间线性插值，末段接分数和斜率连续的指数衰减。
        score = torch.where(x <= 1.0, 1.0 - 0.1 * x,
            torch.where(x <= b, 0.9 - 0.15 * (x - 1.0) / (b - 1.0),
                torch.where(x <= c, 0.75 - 0.25 * (x - b) / (c - b),
                    0.5 * torch.exp(-(x - c) / (2.0 * (c - b))))))
        self._store(values["stability"], score * self.grasp_valid.float())

    def _recovery_snapshot(self, k, i):
        data, values, bilateral = self._env.sim.data, self.values, self.bilateral
        pos_error = (data.xpos[:, self.base] - self.strike_pos).norm(dim=-1)
        rot_error = quat_error_magnitude(data.xquat[:, self.base], self.raised_quat)
        ready = (self.grasp_valid & ~self.dropped & bilateral
                 & (pos_error < 0.03) & (rot_error < math.radians(10.0)))
        self._store(self.recovery_ok[:, i], ready)
        self._store(values["recovery_hold"], self.recovery_ok.float().mean(dim=-1))

    def _head_contact(self, k):
        # 挥击和卸载期间保留原控制计算；实时有效命中、冲量及终局保持另行评分。
        # head_contact 仅匹配锤头与目标端盖；netforce 给出这些接触的世界系合力。
        contact = self._env.scene.sensors["head_contact"].data
        # native netforce 为 primary(锤头) -> secondary(端盖) 的作用力；压缩投影沿 -n。
        # self.normal 是目标端面外法向；只保留压缩法向分量，切向力和负投影不计分。
        self.force = (-(contact.force[:, 0] * self.normal).sum(-1)).clamp_min(0)
        self.touching = contact.found[:, 0] > 0
        self.holding = self.grasp_valid & ~self.dropped & self.bilateral

    def _impact(self, k, strike_id):
        values = self.values
        force, touching, holding = self.force, self.touching, self.holding
        strike_start = self.swing_start + strike_id * self.cycle_steps
        strike_end = strike_start + self.swing_len
        window_steps = round(self._env.cfg.impact_window_s / self._env.physics_dt)
        score_end = (strike_end + window_steps
                     if strike_id < self.cfg.num_strikes - 1 else self.unload_end)
        scoring = k <= score_end
        hit = scoring & ~self.returning & holding & touching & (force > 0)
        self._store(self.strike_hits[:, strike_id], self.strike_hits[:, strike_id] | hit)
        self._store(values["valid_strike"], self.strike_hits.any(dim=-1).float())
        self._store(values["hit_count"], self.strike_hits.sum(dim=-1).float())
        self._store(values["hit_progress"], self.strike_hits.float().mean(dim=-1))
        self._store(values["peak_force"], torch.maximum(values["peak_force"], torch.where(hit, force, 0)))
        first = self.strike_first_hit_step[:, strike_id]
        first = torch.where(hit & (first == 0), k, first)
        self._store(self.strike_first_hit_step[:, strike_id], first)
        in_window = ((first > 0) & (k >= first) & (k - first < window_steps))
        self._store(self.strike_impulse[:, strike_id], self.strike_impulse[:, strike_id] + torch.where(
            ~self.returning & holding & touching & in_window,
            force * self._env.physics_dt, 0.0))
        actual_window_s = window_steps * self._env.physics_dt
        per_strike_mean_force = self.strike_impulse / actual_window_s
        self._store(values["impact_impulse"], self.strike_impulse.sum(dim=-1))
        self._store(values["mean_force"], per_strike_mean_force.mean(dim=-1))
        self._store(values["strike_force_1"], per_strike_mean_force[:, 0])
        self._store(values["strike_force_2"], per_strike_mean_force[:, 1])
        self._store(values["strike_force_3"], per_strike_mean_force[:, 2])
        f = torch.log1p(per_strike_mean_force / self._env.cfg.force_ref_n)
        force_reward = 0.5 * f.mean(dim=-1) + 0.5 * f.min(dim=-1).values
        self._store(values["force"], torch.where(~self.returning, force_reward, values["force"]))

        # 仅第三次挥击及最终卸载的接触参与原提前卸载逻辑。
        if strike_id == self.cfg.num_strikes - 1:
            active = ~self.finished & self.grasp_valid
            self._store(self.impact_started, self.impact_started | (active & touching & (force > 0)))

    def _unload(self, k):
        data, values = self._env.sim.data, self.values
        touching, holding = self.touching, self.holding
        # touch 连续为 0 的物理子步数仅在最终卸载时累计，重新非零即清零。
        self._store(self.unloaded, torch.where(data.sensordata[:, self.touch_address] == 0, self.unloaded + 1, 0))
        self._store(self.finished, self.finished | (self.impact_started & (self.unloaded >= 3) & ~touching))
        all_hits = self.strike_hits.all(dim=-1)
        self._store(values["task_success"], torch.where(
            ~self.returning, (all_hits & holding).float(), values["task_success"]))

    def observe_substep(self):
        for active, sample in self.samples:
            if active[self._substep]:
                sample(self._substep)
        if self.on_physics_step is not None:
            # 评估/回放时安装的逐子步回调，可用于显示或捕获最终指标。
            self.on_physics_step(self)
        # 返回值供 collect metric 记录；框架会对子步返回值求平均，该日志不等于最终窗口平均力。
        # 奖励实际由 rewards.score() 读取 values 中各自的缓存，与这里的返回值分开。
        return self.values["mean_force"]


def collect_attempt(env):
    return env.action_manager.get_term("grasp").observe_substep()
