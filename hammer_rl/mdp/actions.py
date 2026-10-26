"""一次 seed 局部 6D 残差，以及 knock_path.py 固定规则的 GPU 批量表达。"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import (axis_angle_from_quat, quat_apply, quat_apply_inverse,
    quat_box_plus, quat_conjugate, quat_error_magnitude, quat_mul)

from generation.enclosure import build_enclosure_geometry, compute_enclosure_score, compute_handle_seating
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


def attempt_substeps(physics_dt, unload_time):
    *_, steps, substeps = swing_schedule(physics_dt)
    return (round(16.2 / DT) * round(DT / physics_dt) + steps * substeps
            + round(unload_time / physics_dt) + round(0.45 / DT) * round(DT / physics_dt))


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
class GraspResidualActionCfg(ActionTermCfg):
    position_scale: float = 0.02
    rotation_scale: float = math.pi / 16
    unload_time: float = 1.0

    def build(self, env):
        return GraspResidualAction(self, env)


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
        self.swing_end = self.swing_start + self.swing_steps * self.swing_substeps
        self.unload_end = self.swing_end + round(cfg.unload_time / env.physics_dt)
        self.return_steps = round(0.45 / DT) * self.substeps
        self.on_physics_step = None  # 仅评估安装，训练路径没有 CPU 同步。
        self._substep = 0
        self.values = {name: torch.zeros(env.num_envs, device=self.device) for name in (
            "lift_success", "wrap", "orientation", "distance", "stability", "force",
            "translation_drift", "rotation_drift", "peak_force", "touch_peak", "return_touch_peak",
            "valid_strike", "lift_height", "grasp_before_swing", "grip_distance",
            "seat_closure", "seat_lift", "seat_before_swing")}
        self.grasp_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
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
        for value in (self.grasp_valid, self.impact_started, self.finished, self.returning,
                      self.unloaded, self.return_start, self.palm_frames, self.seat_frames, self.closure_enclosure):
            value[env_ids] = 0
        # 不清除 reset 事件刚写入的 sample/seed，也不推进其他环境的调度时钟。

    def process_actions(self, actions):
        self._raw_action.copy_(actions)
        self._substep = 0
        delta = torch.tanh(actions)
        self.final_pos = self.task.seed_pos + quat_apply(self.task.seed_quat, delta[:, :3] * self.cfg.position_scale)
        local_rotation = quat_box_plus(self.identity, delta[:, 3:] * self.cfg.rotation_scale)
        self.final_quat = quat_mul(self.task.seed_quat, local_rotation)
        self.pregrasp = self.final_pos - 0.10 * quat_apply(self.final_quat, self.unit[:, 2])
        data = self._env.sim.data
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
        face = hammer_pos + quat_apply(hammer_quat, -0.24 * self.unit[:, 0] + 0.04 * self.unit[:, 2])
        self.tip_offset = quat_apply_inverse(mount_quat, face - mount_pos)
        self.head_offset = quat_apply_inverse(mount_quat,
            hammer_pos - 0.24 * quat_apply(hammer_quat, self.unit[:, 0]) - mount_pos)
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
            boundaries = ((1.0, self.start_pos, self.pregrasp, self.start_quat, self.final_quat, self.task.open_ctrl),
                (0.6, self.pregrasp, self.final_pos, self.final_quat, self.final_quat, self.task.open_ctrl),
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
            index = (k - self.swing_start - 1) // self.swing_substeps
            if (k - self.swing_start - 1) % self.swing_substeps == 0:
                self.swing_p0, self.swing_q0 = self._swing_pose(self._swing_progress(index))
                self.swing_p1, self.swing_q1 = self._swing_pose(self._swing_progress(index + 1))
            u = ((k - self.swing_start - 1) % self.swing_substeps + 1) / self.swing_substeps
            self._write(self.swing_p0 + u * (self.swing_p1 - self.swing_p0),
                        interpolate_quat(self.swing_q0, self.swing_q1, u))
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
        # env_cfg.py 中当前设置：R = wrap + orientation + distance + 2*stability + 4*force。
        # wrap 沿用 ICRA-BUILD 的有符号评分；其他 values 项用于判据或日志，不单独加入奖励。
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
                                                                    self.palm_frames >= 5))
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
        if k == self.lift_end:
            # A 快照：抬升结束（当前为尝试开始后约 3.2 s）。前三项奖励在此时写入，
            # wrap 在挥击前复核顶靠；保存夹爪系下的锤子位姿，用于之后评价抓持稳定性。
            self.relative_a_pos, self.relative_a_quat = (value.clone() for value in self._relative_pose())
            contact = self._env.scene.sensors["handle_contact"].data
            # handle_contact 只检测六个指面与锤柄的接触，每个指面最多保留两个接触槽。
            # 12 个槽中前 6 个属于 narrow 侧，后 6 个属于 wide 侧；槽内有接触且法向力 > 0
            # 才算有效。双侧各至少一个有效接触，才能通过 bilateral 判据。
            valid = (contact.found > 0) & (contact.force[:, :, 0] > 0)
            bilateral = valid[:, :6].any(-1) & valid[:, 6:].any(-1)
            initial = self.task.bank["hammer_pos"][self.task.scene_id, 2]
            values["lift_height"].copy_(data.xpos[:, self.hammer, 2] - initial)
            # 有效抬升 lift：锤子相对本场景初态的真实高度增量 >= lift_threshold，且双侧接触。
            # lift_threshold 当前为 0.15 m；lift_success 是 0/1 日志值，也作为奖励的有效性门控。
            # 抬升失败时 wrap、orientation、distance 均为 0，没有额外的失败扣分。
            lift = (values["lift_height"] >= self._env.cfg.lift_threshold) & bilateral
            values["lift_success"].copy_(lift)

            # 1. 包覆奖励：原真包络角度评分，乘抬升成功及闭合/抬升顶靠门控。
            values["wrap"].copy_(torch.where(lift, self.closure_enclosure, 0)
                                 * values["seat_closure"] * values["seat_lift"])

            # 2. 朝向奖励 orientation：axis 为从 narrow 指尖指向 wide 指尖的单位向量，
            # normal 为锤体局部 +Z（敲击面法向）旋转到世界系后的单位向量。
            # orientation = max(axis·normal, 0) * lift；同向平行得 1，垂直或反平行得 0。
            axis = torch.nn.functional.normalize(data.site_xpos[:, self.wide] - data.site_xpos[:, self.narrow], dim=-1)
            normal = quat_apply(data.xquat[:, self.hammer], self.unit[:, 2])
            values["orientation"].copy_((axis * normal).sum(-1).clamp_min(0) * lift)

            # 3. 距离奖励 distance：抓取中心取两指尖中点；锤头中心位于锤体原点沿局部 -X
            # 方向 0.24 m 处。d 是两者的三维欧氏距离，grip_distance 保存其原始米制值。
            # distance = clamp(d / distance_ref, 0, 1) * lift：有效抬升后，抓得离锤头越远
            # 分数越高，达到参考距离后饱和。distance_ref 当前为 0.315 m，取一半时得 0.5。
            center = (data.site_xpos[:, self.wide] + data.site_xpos[:, self.narrow]) / 2
            head = data.xpos[:, self.hammer] - 0.24 * quat_apply(data.xquat[:, self.hammer], self.unit[:, 0])
            distance = (center - head).norm(dim=-1)
            values["grip_distance"].copy_(distance)
            values["distance"].copy_((distance / self._env.cfg.distance_ref).clamp(0, 1) * lift)
            # 按 A 快照规划后续固定挥击轨迹，同时准备后面投影接触力所需的目标端面法向。
            self._plan_strike()
        if k == self.swing_start:
            values["wrap"].mul_(values["seat_before_swing"])
            # B 快照：搬运和预备姿态调整结束、即将挥击（当前约 16.2 s）。
            # grasp_valid 要求 A 时刻已有效抬升，且 B 时刻两侧仍各有有效接触；
            # 它在此锁存，供稳定奖励及之后的敲击力累计使用，不在挥击中重新判断双侧接触。
            p, q = self._relative_pose()
            contact = self._env.scene.sensors["handle_contact"].data
            valid = (contact.found > 0) & (contact.force[:, :, 0] > 0)
            self.grasp_valid.copy_((values["lift_success"] > 0) & valid[:, :6].any(-1) & valid[:, 6:].any(-1))
            values["grasp_before_swing"].copy_(self.grasp_valid)

            # 4. 稳定奖励 stability：比较 A/B 两时刻夹爪坐标系下的锤子相对位姿，
            # 因而共同搬运、转动不会直接算作漂移。ep 为相对平移差的模（m），
            # er 为两个相对姿态的旋转角误差（rad），不是四元数各分量的差。
            ep = (p - self.relative_a_pos).norm(dim=-1)
            er = quat_error_magnitude(self.relative_a_quat, q)
            values["translation_drift"].copy_(ep)
            values["rotation_drift"].copy_(er)
            # stability = exp(-(ep/sp)^2 - (er/sr)^2) * grasp_valid。
            # 当前 sp=0.01 m、sr=pi/36（5°）；无漂移得 1，单独漂移 1 cm 或 5° 得 exp(-1)。
            # 尺度越小，对滑移越敏感；无有效抓持时为 0。本项在 B 时刻计算一次。
            values["stability"].copy_(torch.exp(-(ep / self._env.cfg.stability_position_scale)**2
                - (er / self._env.cfg.stability_rotation_scale)**2) * self.grasp_valid)
        if self.swing_start < k <= self.unload_end:
            # 5. 敲击力奖励 force：在挥击开始后至卸载时窗结束之间，逐物理子步维护峰值。
            # head_contact 仅匹配锤头与目标端盖；netforce 给出这些接触的世界系合力。
            contact = self._env.scene.sensors["head_contact"].data
            # native netforce 为 primary(锤头) -> secondary(端盖) 的作用力；压缩投影沿 -n。
            # self.normal 是目标端面外法向；只保留压缩法向分量，切向力和负投影不计分。
            force = (-(contact.force[:, 0] * self.normal).sum(-1)).clamp_min(0)
            touching = contact.found[:, 0] > 0
            # 仅对挥击前有效抓持、且尚未结束计分的环境累计；完成卸载后不再更新峰值。
            # impact_started 锁存是否出现过有效正法向接触；valid_strike 只是 0/1 日志，
            # 不单独加分。peak_force 是奖励使用的峰值（N），并非对子步接触力取平均。
            active = ~self.finished & self.grasp_valid
            self.impact_started |= active & touching & (force > 0)
            values["peak_force"].copy_(torch.maximum(values["peak_force"], torch.where(active, force, 0)))
            # touch_peak 另存原场景 touch 传感器的峰值作诊断；奖励仍使用上面的 head_contact。
            values["touch_peak"].copy_(torch.maximum(values["touch_peak"], torch.where(active,
                data.sensordata[:, self.touch_address], 0)))
            # touch 连续为 0 的物理子步数用于判断卸载；重新非零即清零计数。
            # 挥击结束后，已经有效命中、连续至少 3 步 touch=0 且锤头接触消失，才提前结束计分。
            # 未提前满足条件的环境由固定 unload_end 时窗终止，之后执行撤离。
            self.unloaded = torch.where(data.sensordata[:, self.touch_address] == 0, self.unloaded + 1, 0)
            self.finished |= (k > self.swing_end) & self.impact_started & (self.unloaded >= 3) & ~touching
            values["valid_strike"].copy_(self.impact_started)
            # force = F_peak / (F_peak + force_scale)；当前 force_scale=100 N。
            # 0 N 得 0、100 N 得 0.5、300 N 得 0.75；力越大越接近 1，且边际增益递减。
            # force_scale 是归一化尺度，不是命中阈值；无有效抓持时峰值保持 0，评分也为 0。
            values["force"].copy_(values["peak_force"] / (values["peak_force"] + self._env.cfg.force_scale))
        if k > self.swing_end:
            # 撤离时的 touch 峰值单独记录，只有 returning=True 才累计，不加入五项奖励。
            values["return_touch_peak"].copy_(torch.maximum(values["return_touch_peak"],
                torch.where(self.returning, data.sensordata[:, self.touch_address], 0)))
        if self.on_physics_step is not None:
            # 评估/回放时安装的逐子步回调，可用于显示或捕获最终指标。
            self.on_physics_step(self)
        # 返回值供 collect metric 记录；框架会对子步返回值求平均，该日志不等于最终力峰值。
        # 五项奖励实际由 rewards.score() 读取 values 中各自的缓存，与这里的返回值分开。
        return values["peak_force"]


def collect_attempt(env):
    return env.action_manager.get_term("grasp").observe_substep()
