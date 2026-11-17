"""一次完整 6D 抓取生成，以及 knock_path.py 固定规则的 GPU 批量表达。"""

from dataclasses import dataclass
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


def swing_schedule(physics_dt):
    # 仅构造时使用 NumPy；与原脚本完全相同的 1001 点弧长表。
    s = np.linspace(0.0, 1.0, 1001)
    points = np.column_stack((0.40 * s, 0.30 * (1 - s * s) + 0.10 * s * (1 - s)))
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    duration = math.sqrt(2.0 * length[-1] / 10.0)
    steps = round(duration / DT)
    return s, length, duration, steps, round(duration / steps / physics_dt)


def attempt_substeps(physics_dt, unload_time, num_strikes=3, recover_time=0.8):
    *_, steps, substeps = swing_schedule(physics_dt)
    swing_len = steps * substeps
    recover_len = round(recover_time / physics_dt)
    return (
        round(16.2 / DT) * round(DT / physics_dt)
        + num_strikes * swing_len
        + (num_strikes - 1) * recover_len
        + round(unload_time / physics_dt)
        + round(0.45 / DT) * round(DT / physics_dt)
    )


def interpolate_quat(start, end, s):
    delta = axis_angle_from_quat(quat_mul(end, quat_conjugate(start)))
    return quat_box_plus(start, delta * s)


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
class DirectGraspActionCfg(ActionTermCfg):
    unload_time: float = 1.0
    num_strikes: int = 3
    recover_time: float = 0.8

    def build(self, env):
        return DirectGraspAction(self, env)


def direct_grasp_pose(hammer_pos, hammer_quat, handle_size, tcp_offset, actions):
    """由锤柄局部抓持中心和完整轴角构造世界系 TCP/mount 位姿。"""
    a = torch.tanh(actions)
    radius, half_length = handle_size[:, 0], handle_size[:, 1]
    local_pos = torch.stack((half_length * a[:, 0], radius * a[:, 1],
                             radius * a[:, 2]), dim=-1)
    center = hammer_pos + quat_apply(hammer_quat, local_pos)
    rotvec = a[:, 3:]
    rotvec = math.pi * rotvec / rotvec.norm(dim=-1, keepdim=True).clamp_min(1.0)
    identity = torch.zeros_like(hammer_quat)
    identity[:, 0] = 1
    local_quat = quat_box_plus(identity, rotvec)
    final_quat = quat_mul(hammer_quat, local_quat)
    final_pos = center - quat_apply(final_quat, tcp_offset)
    approach = torch.zeros_like(final_pos)
    approach[:, 2] = 1
    pregrasp = final_pos - 0.10 * quat_apply(final_quat, approach)
    return final_pos, final_quat, pregrasp


class DirectGraspAction(ActionTerm):
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
        self.closure_start = round(1.6 / env.physics_dt)
        self.closure_end = round(2.4 / env.physics_dt)
        self.enclosure_sample_steps = round(1 / (250 * env.physics_dt))
        self.palm_frames = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.seat_frames = torch.zeros_like(self.palm_frames)
        self.closure_enclosure = torch.zeros(env.num_envs, device=self.device)
        self.lift_end = round(3.2 / env.physics_dt)
        self.swing_start = round(16.2 / env.physics_dt)
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
        self.values = {name: torch.zeros(env.num_envs, device=self.device) for name in (
            "capture", "lift_progress", "lift_success", "wrap", "orientation", "distance", "stability", "force",
            "impact_impulse", "task_success", "drop",
            "translation_drift", "rotation_drift", "mean_force",
            "valid_strike", "lift_height", "grasp_before_swing",
            "seat_closure", "seat_lift", "seat_before_swing", "peak_force",
            "hit_progress", "recovery_hold", "hit_count",
            "strike_force_1", "strike_force_2", "strike_force_3")}
        self.grasp_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.relative_closure_pos = torch.zeros((env.num_envs, 3), device=self.device)
        self.relative_closure_quat = self.identity.clone()
        self.dropped = torch.zeros_like(self.grasp_valid)
        self.lost_contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
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
        for value in (self.grasp_valid, self.dropped, self.lost_contact_steps,
                      self.strike_hits, self.strike_first_hit_step, self.strike_impulse, self.recovery_ok,
                      self.impact_started, self.finished, self.returning,
                      self.unloaded, self.return_start, self.palm_frames, self.seat_frames, self.closure_enclosure):
            value[env_ids] = 0
        # 不清除 reset 事件刚写入的场景观测，也不推进其他环境的调度时钟。

    def process_actions(self, actions):
        self._raw_action.copy_(actions)
        self._substep = 0
        bank, scene = self.task.bank, self.task.scene_id
        data = self._env.sim.data
        tcp_center_world = 0.5 * (data.site_xpos[:, self.narrow] + data.site_xpos[:, self.wide])
        tcp_offset = quat_apply_inverse(data.xquat[:, self.base],
            tcp_center_world - data.xpos[:, self.base])
        self.final_pos, self.final_quat, self.pregrasp = direct_grasp_pose(
            bank["hammer_pos"][scene], bank["hammer_quat"][scene],
            bank["handle_size"][scene], tcp_offset, actions)
        self.start_pos = data.mocap_pos[:, self.mocap].clone()
        self.start_quat = data.mocap_quat[:, self.mocap].clone()
        self.lifted_pos = self.final_pos + 0.25 * self.unit[:, 2]

    def _write(self, pos, quat, grip=1.0):
        data = self._env.sim.data
        data.mocap_pos[:, self.mocap] = pos
        data.mocap_quat[:, self.mocap] = quat
        data.ctrl[:, self.actuator] = grip

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
        self.tip_offset = quat_apply_inverse(mount_quat, face - mount_pos)
        self.head_offset = quat_apply_inverse(mount_quat,
            hammer_pos + quat_apply(hammer_quat, self.head_local) - mount_pos)
        center = (data.site_xpos[:, self.narrow] + data.site_xpos[:, self.wide]) / 2
        self.grip_offset = quat_apply_inverse(mount_quat, center - mount_pos)
        self.target_pos = data.site_xpos[:, self.target].clone()
        frame = data.site_xmat[:, self.target].reshape(-1, 3, 3)
        self.normal = frame[:, :, 2].clone()
        self.swing_axis = frame[:, :, 1].clone()
        self.horizontal = torch.nn.functional.normalize(torch.cross(self.swing_axis, self.unit[:, 2], dim=-1), dim=-1)
        target_quat = self.task.bank["target_quat"][self.task.scene_id]
        impact_hammer = quat_mul(target_quat, quat_box_plus(self.identity, math.pi * self.unit[:, 1]))
        self.impact_quat = quat_mul(impact_hammer, quat_conjugate(self.relative_a_quat))
        self.raised_quat = quat_box_plus(self.impact_quat, -math.pi / 3 * self.swing_axis)
        pivot = self.target_pos - quat_apply(self.impact_quat, self.tip_offset)
        self.grip_impact = pivot + quat_apply(self.impact_quat, self.grip_offset)
        self.head_impact = pivot + quat_apply(self.impact_quat, self.head_offset)
        self.carry_quat = data.mocap_quat[:, self.mocap].clone()
        self.head_start = data.mocap_pos[:, self.mocap] + quat_apply(self.carry_quat, self.head_offset)
        self.high_start, self.high_target = self.head_start.clone(), self.head_impact.clone()
        self.high_start[:, 2] = self.high_target[:, 2] = 0.75
        self.strike_pos = self._swing_pose(0.0)[0]
        self.head_strike = self.strike_pos + quat_apply(self.raised_quat, self.head_offset)

    def _swing_pose(self, s):
        quat = quat_box_plus(self.impact_quat, -math.pi / 3 * (1 - s) * self.swing_axis)
        center = (self.grip_impact + (0.30 * (1 - s * s) + 0.10 * s * (1 - s)) * self.unit[:, 2]
                  - 0.40 * (1 - s) * self.horizontal)
        return center - quat_apply(quat, self.grip_offset), quat

    def _swing_progress(self, index):
        distance = 5.0 * (index * self.swing_duration / self.swing_steps)**2
        d = torch.full((1,), distance, device=self.device)
        i = torch.searchsorted(self.path_length, d).clamp(1, 1000)
        fraction = (d - self.path_length[i - 1]) / (self.path_length[i] - self.path_length[i - 1])
        return (self.path_s[i - 1] + fraction * (self.path_s[i] - self.path_s[i - 1]))[0]

    def apply_actions(self):
        self._substep += 1
        k, data = self._substep, self._env.sim.data
        if k <= self.lift_end:
            boundaries = ((1.0, self.start_pos, self.pregrasp, self.start_quat, self.final_quat, 0.0),
                (0.6, self.pregrasp, self.final_pos, self.final_quat, self.final_quat, 0.0),
                (0.8, self.final_pos, self.final_pos, self.final_quat, self.final_quat, 1.0),
                (0.8, self.final_pos, self.lifted_pos, self.final_quat, self.final_quat, 1.0))
            for duration, p0, p1, q0, q1, grip in boundaries:
                steps = round(duration / self._env.physics_dt)
                if k <= steps:
                    s = sampled_progress(k, duration, self.substeps)
                    self._write(p0 + s * (p1 - p0), interpolate_quat(q0, q1, s), grip)
                    return
                k -= steps
        elif k <= self.swing_start:
            k -= self.lift_end
            segments = ((2.0, self.head_start, self.high_start, self.carry_quat, self.carry_quat),
                (4.0, self.high_start, self.high_target, self.carry_quat, self.carry_quat),
                (4.0, self.high_target, self.high_target, self.carry_quat, self.raised_quat),
                (2.0, self.high_target, self.head_strike, self.raised_quat, self.raised_quat))
            for duration, p0, p1, q0, q1 in segments:
                steps = round(duration / self._env.physics_dt)
                if k <= steps:
                    s = sampled_progress(k, duration, self.substeps, quintic=True)
                    q = interpolate_quat(q0, q1, s)
                    self._write(p0 + s * (p1 - p0) - quat_apply(q, self.head_offset), q)
                    return
                k -= steps
            self._write(self.strike_pos, self.raised_quat)
        elif k <= self.swing_end:
            elapsed = k - self.swing_start - 1
            phase = elapsed % self.cycle_steps
            if phase < self.swing_len:
                index = phase // self.swing_substeps
                if phase % self.swing_substeps == 0:
                    self.swing_p0, self.swing_q0 = self._swing_pose(self._swing_progress(index))
                    self.swing_p1, self.swing_q1 = self._swing_pose(self._swing_progress(index + 1))
                u = (phase % self.swing_substeps + 1) / self.swing_substeps
                self._write(self.swing_p0 + u * (self.swing_p1 - self.swing_p0),
                            interpolate_quat(self.swing_q0, self.swing_q1, u))
            else:
                j = phase - self.swing_len + 1
                u = j / self.recovery_steps
                h = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                p, q = self._swing_pose(1.0 - h)
                self._write(p, q)
        else:
            if k == self.swing_end + 1:
                data.eq_active[:, self.weld] = False
            # 未命中也在固定时窗内正常结束。不同环境可以提前完成卸载及撤离。
            self.finished |= (k > self.unload_end)
            new_return = self.finished & ~self.returning
            self.return_start = torch.where(new_return, k - 1, self.return_start)
            adr = self.task.base_qadr
            self.return_pos = torch.where(new_return[:, None], data.qpos[:, adr:adr + 3], self.return_pos)
            self.return_quat = torch.where(new_return[:, None], data.qpos[:, adr + 3:adr + 7], self.return_quat)
            self.returning |= new_return
            progress = (k - self.return_start).clamp(0, self.return_steps)
            s = sampled_progress(progress, round(0.45 / DT) * DT, self.substeps)
            p = self.return_pos + 0.03 * s[:, None] * self.normal
            self._write(torch.where(self.returning[:, None], p, data.mocap_pos[:, self.mocap]),
                        torch.where(self.returning[:, None], self.return_quat, data.mocap_quat[:, self.mocap]))
            data.eq_active[:, self.weld] = self.returning
            masses = (self._env.sim.model.body_mass[:, self.moving] if self._env.cfg.randomize_physics
                      else self.task.nominal_mass[self.moving].expand(self.num_envs, -1))
            compensation = -masses[:, :, None] * self.gravity
            data.xfrc_applied[:, self.moving, :3] = compensation * (~self.returning)[:, None, None]

    def observe_substep(self):
        # collect_attempt() 通过 per_substep metric 在每个物理子步后调用本方法。
        # 一次策略动作执行整次尝试；这里按阶段计算评分，写入逐环境的 values 缓存。
        # rewards.score() 在环境计算奖励时读取缓存；本方法不直接加权求和。
        # 各项权重由 env_cfg.py 注册；事件每次尝试只计一次，物理子步只更新评分缓存。
        # reset() 会清零评分和状态，因此未满足条件的尝试不会沿用上一回合的结果。
        k, data, values = self._substep, self._env.sim.data, self.values
        if k in (self.closure_end, self.lift_end, self.swing_start):
            # 固定快照刷新 GPU 派生量，对应原脚本阶段结束后的 forward。
            self._env.sim.forward()
        if self.closure_start <= k <= self.closure_end and (k - self.closure_start) % self.enclosure_sample_steps == 0:
            contact = self._env.scene.sensors["palm_contact"].data
            force = contact.force[:, :, 0].abs()
            strong = ((contact.found > 0) & ((force >= 0.1) | (force / 250 >= 0.0001))).any(-1)
            self.palm_frames.copy_(torch.where(strong, self.palm_frames + 1, 0))
            if k == self.closure_end:
                self.closure_enclosure.copy_(compute_enclosure_score(data, self.enclosure_geometry,
                    self.palm_frames >= 5, self.task.bank["handle_convex_planes"][self.task.scene_id]))
        # 每个关键时点只检查末尾五个 250 Hz 采样帧，包含终态。
        for end, name in ((self.closure_end, "seat_closure"), (self.lift_end, "seat_lift"),
                          (self.swing_start, "seat_before_swing")):
            start = end - 4 * self.enclosure_sample_steps
            if start <= k <= end and (end - k) % self.enclosure_sample_steps == 0:
                if k == start:
                    self.seat_frames.zero_()
                seated = compute_handle_seating(self._env.scene.sensors["handle_seat_contact"].data,
                                               data, self.enclosure_geometry)
                self.seat_frames.copy_(torch.where(seated, self.seat_frames + 1, 0))
                if k == end:
                    values[name].copy_(self.seat_frames >= 5)
        if k == self.closure_end or self.lift_end <= k <= self.unload_end:
            contact = self._env.scene.sensors["handle_contact"].data
            # handle_contact 只检测六个指面与锤柄的接触，每个指面最多保留两个接触槽。
            # 12 个槽中前 6 个属于 narrow 侧，后 6 个属于 wide 侧；槽内有接触且法向力 > 0
            # 才算有效。双侧各至少一个有效接触，才能通过 bilateral 判据。
            valid = (contact.found > 0) & (contact.force[:, :, 0] > 0)
            left, right = valid[:, :6].any(-1), valid[:, 6:].any(-1)
            bilateral = left & right
            if k == self.closure_end:
                values["capture"].copy_((left.float() + right.float()) / 2)
                p, q = self._relative_pose()
                self.relative_closure_pos.copy_(p)
                self.relative_closure_quat.copy_(q)
            if k > self.lift_end:
                # 从抬升后评价；连续失联 20 ms 锁存掉落。
                evaluating = ~self.returning
                lost = (values["lift_success"] > 0) & ~bilateral
                self.lost_contact_steps.copy_(torch.where(evaluating,
                    torch.where(lost, self.lost_contact_steps + 1, 0), self.lost_contact_steps))
                self.dropped |= evaluating & (self.lost_contact_steps >= self.substeps)
                values["drop"].copy_(self.dropped)
        if k == self.lift_end:
            # 抬升结束快照（当前为尝试开始后约 3.2 s）。A 的抬升和几何评分在此写入，
            # wrap 在挥击前追加顶靠奖励；保存夹爪系下的锤子位姿，用于规划后续挥击轨迹。
            self.relative_a_pos, self.relative_a_quat = (value.clone() for value in self._relative_pose())
            initial = self.task.bank["hammer_pos"][self.task.scene_id, 2]
            values["lift_height"].copy_(data.xpos[:, self.hammer, 2] - initial)
            # 有效抬升 lift：锤子相对本场景初态的真实高度增量 >= lift_threshold，且双侧接触。
            # lift_threshold 当前为 0.15 m；lift_success 是 0/1 日志值，也作为奖励的有效性门控。
            # 抬升失败时几何分为零；有双侧接触时仍可获得部分抬升进展分。
            lift = (values["lift_height"] >= self._env.cfg.lift_threshold) & bilateral
            values["lift_success"].copy_(lift)
            values["lift_progress"].copy_((values["lift_height"] / self._env.cfg.lift_threshold).clamp(0, 1)
                                          * bilateral)

            # 1. 包覆奖励：抬升成功后，真包络角度评分放大 2 倍，加闭合/抬升顶靠奖励。
            values["wrap"].copy_(torch.where(lift, 2 * self.closure_enclosure
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
            values["orientation"].copy_(((1.0 - 0.6 * flatness) * transverse
                                         + 0.6 * flatness * side_score) * lift)

            # 沿柄轴投影评分。
            center = (data.site_xpos[:, self.wide] + data.site_xpos[:, self.narrow]) / 2
            head = data.xpos[:, self.hammer] + quat_apply(data.xquat[:, self.hammer], self.head_local)
            distance = ((center - head) * handle_axis).sum(-1)
            values["distance"].copy_((distance / self._env.cfg.distance_ref).clamp(0, 1) * lift)
            # 按 A 快照规划后续固定挥击轨迹，同时准备后面投影接触力所需的目标端面法向。
            self._plan_strike()
        if k == self.swing_start:
            # 挥击前快照（当前约 16.2 s）：需已有效抬升、未锁存掉落且当前双侧有效接触。
            self.grasp_valid.copy_((values["lift_success"] > 0) & bilateral & ~self.dropped)
            values["grasp_before_swing"].copy_(self.grasp_valid)
            values["wrap"].add_(0.25 * self.grasp_valid * values["seat_before_swing"] / 1.5)
            # 比较闭合结束与挥击前的相对位姿。
            p, q = self._relative_pose()
            values["translation_drift"].copy_((p - self.relative_closure_pos).norm(dim=-1))
            values["rotation_drift"].copy_(quat_error_magnitude(self.relative_closure_quat, q))
            x = torch.sqrt(0.5 * ((values["translation_drift"] / self._env.cfg.stability_position_scale).square()
                + (values["rotation_drift"] / self._env.cfg.stability_rotation_scale).square()))
            b, c = math.sqrt(26.0) / 3.0, math.sqrt(10.0)
            # 三个给分节点间线性插值，末段接分数和斜率连续的指数衰减。
            score = torch.where(x <= 1.0, 1.0 - 0.1 * x,
                torch.where(x <= b, 0.9 - 0.15 * (x - 1.0) / (b - 1.0),
                    torch.where(x <= c, 0.75 - 0.25 * (x - b) / (c - b),
                        0.5 * torch.exp(-(x - c) / (2.0 * (c - b))))))
            values["stability"].copy_(score * self.grasp_valid.float())

        for i in range(self.cfg.num_strikes - 1):
            if k == self.swing_start + (i + 1) * self.cycle_steps:
                pos_error = (data.xpos[:, self.base] - self.strike_pos).norm(dim=-1)
                rot_error = quat_error_magnitude(data.xquat[:, self.base], self.raised_quat)
                ready = (self.grasp_valid & ~self.dropped & bilateral
                         & (pos_error < 0.03) & (rot_error < math.radians(10.0)))
                self.recovery_ok[:, i].copy_(ready)
                values["recovery_hold"].copy_(self.recovery_ok.float().mean(dim=-1))
        if self.swing_start < k <= self.unload_end:
            # 挥击和卸载期间保留原控制计算；实时有效命中、冲量及终局保持另行评分。
            # head_contact 仅匹配锤头与目标端盖；netforce 给出这些接触的世界系合力。
            contact = self._env.scene.sensors["head_contact"].data
            # native netforce 为 primary(锤头) -> secondary(端盖) 的作用力；压缩投影沿 -n。
            # self.normal 是目标端面外法向；只保留压缩法向分量，切向力和负投影不计分。
            force = (-(contact.force[:, 0] * self.normal).sum(-1)).clamp_min(0)
            touching = contact.found[:, 0] > 0
            holding = self.grasp_valid & ~self.dropped & bilateral
            if k <= self.swing_end:
                strike_id = (k - self.swing_start - 1) // self.cycle_steps
            else:
                strike_id = self.cfg.num_strikes - 1
            strike_start = self.swing_start + strike_id * self.cycle_steps
            strike_end = strike_start + self.swing_len
            window_steps = round(self._env.cfg.impact_window_s / self._env.physics_dt)
            score_end = (strike_end + window_steps
                         if strike_id < self.cfg.num_strikes - 1 else self.unload_end)
            scoring = strike_start < k <= score_end
            hit = scoring & ~self.returning & holding & touching & (force > 0)
            self.strike_hits[:, strike_id] |= hit
            values["valid_strike"].copy_(self.strike_hits.any(dim=-1).float())
            values["hit_count"].copy_(self.strike_hits.sum(dim=-1).float())
            values["hit_progress"].copy_(self.strike_hits.float().mean(dim=-1))
            values["peak_force"].copy_(torch.maximum(values["peak_force"], torch.where(hit, force, 0)))
            first = self.strike_first_hit_step[:, strike_id]
            first = torch.where(hit & (first == 0), torch.full_like(first, k), first)
            self.strike_first_hit_step[:, strike_id].copy_(first)
            in_window = ((first > 0) & (k >= first) & (k - first < window_steps))
            self.strike_impulse[:, strike_id].add_(torch.where(
                ~self.returning & holding & touching & in_window,
                force * self._env.physics_dt, 0.0))
            actual_window_s = window_steps * self._env.physics_dt
            per_strike_mean_force = self.strike_impulse / actual_window_s
            values["impact_impulse"].copy_(self.strike_impulse.sum(dim=-1))
            values["mean_force"].copy_(per_strike_mean_force.mean(dim=-1))
            values["strike_force_1"].copy_(per_strike_mean_force[:, 0])
            values["strike_force_2"].copy_(per_strike_mean_force[:, 1])
            values["strike_force_3"].copy_(per_strike_mean_force[:, 2])
            f = torch.log1p(per_strike_mean_force / self._env.cfg.force_ref_n)
            force_reward = 0.5 * f.mean(dim=-1) + 0.5 * f.min(dim=-1).values
            values["force"].copy_(torch.where(~self.returning, force_reward, values["force"]))

            # 仅第三次挥击及最终卸载的接触参与原提前卸载逻辑。
            if strike_id == self.cfg.num_strikes - 1:
                active = ~self.finished & self.grasp_valid
                self.impact_started |= active & touching & (force > 0)
            if k > self.swing_end:
                # touch 连续为 0 的物理子步数仅在最终卸载时累计，重新非零即清零。
                self.unloaded = torch.where(data.sensordata[:, self.touch_address] == 0, self.unloaded + 1, 0)
                self.finished |= self.impact_started & (self.unloaded >= 3) & ~touching
                all_hits = self.strike_hits.all(dim=-1)
                values["task_success"].copy_(torch.where(
                    ~self.returning, (all_hits & holding).float(), values["task_success"]))
        if self.on_physics_step is not None:
            # 评估/回放时安装的逐子步回调，可用于显示或捕获最终指标。
            self.on_physics_step(self)
        # 返回值供 collect metric 记录；框架会对子步返回值求平均，该日志不等于最终窗口平均力。
        # 奖励实际由 rewards.score() 读取 values 中各自的缓存，与这里的返回值分开。
        return values["mean_force"]


def collect_attempt(env):
    return env.action_manager.get_term("grasp").observe_substep()
