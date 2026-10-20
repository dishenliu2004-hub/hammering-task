"""mjlab：八种放置姿态的锤头中心轨迹规划与夹爪位姿输出。"""

import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from generation.hammer_generate import sync_data


def step_mocap(model, data, pos, quat, gripper_action, dt=0.02, viewer=None, on_step=None,
               *, simulation):
    mocap_id = model.body_mocapid[model.body("gripper_mount").id]
    start_pos = data.mocap_pos[mocap_id].copy()
    start_rot = Rotation.from_quat(data.mocap_quat[mocap_id][[1, 2, 3, 0]])
    target_rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
    delta_pos = np.asarray(pos) - start_pos
    delta_rot = (target_rot * start_rot.inv()).as_rotvec()
    simulation.data.ctrl[0, model.actuator("gripper_gripper_actuator").id] = gripper_action

    # 每个物理子步更新位姿，不再一次跳到 20 ms 后的位置再保持。
    steps = round(dt / model.opt.timestep)
    for i in range(steps):
        u = (i + 1) / steps
        data.mocap_pos[mocap_id] = start_pos + u * delta_pos
        rot = Rotation.from_rotvec(u * delta_rot) * start_rot
        data.mocap_quat[mocap_id] = rot.as_quat()[[3, 0, 1, 2]]
        simulation.data.mocap_pos[0, mocap_id] = torch.as_tensor(
            data.mocap_pos[mocap_id], dtype=simulation.data.mocap_pos.dtype,
            device=simulation.device,
        )
        simulation.data.mocap_quat[0, mocap_id] = torch.as_tensor(
            data.mocap_quat[mocap_id], dtype=simulation.data.mocap_quat.dtype,
            device=simulation.device,
        )
        simulation.step()
    sync_data(simulation, data)

    if viewer is not None:
        viewer.sync()
        time.sleep(dt)
    if on_step is not None:
        on_step()


def move_to(model, data, viewer, target_pos, target_quat, duration, grip, dt=0.02, on_step=None,
            *, simulation):
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
        step_mocap(model, data, pos, rot.as_quat()[[3, 0, 1, 2]], grip, dt, viewer, on_step,
                   simulation=simulation)


def hammer_strike(
    model,
    data,
    eef_pos,
    eef_quat,
    target_pos,
    hammer_tip_offset,
    impact_quat,
    gripper_action=1.0,
    dt=0.02,
    viewer=None,
    on_step=None,
    *,
    simulation,
    placement=None,
    hammer_head_center_offset=None,
):
    # 两个偏移均以夹爪安装点为原点、在夹爪局部坐标系中表达。
    lift_angle = np.deg2rad(50.0)
    stroke_distance = 0.08
    horizontal_distance = 0.08
    retreat_distance = 0.08
    swing_time = 0.18
    return_time = 0.45
    carry_z = 0.75
    lift_time, carry_time, turn_time, lower_time = 2.0, 4.0, 4.0, 2.0
    settle_time = 1.0

    def to_rotation(q):
        return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])

    def step(pos, rot):
        step_mocap(
            model, data, pos, rot.as_quat()[[3, 0, 1, 2]],
            gripper_action, dt, viewer, on_step,
            simulation=simulation,
        )

    mocap_id = model.body_mocapid[model.body("gripper_mount").id]
    start_pos = data.mocap_pos[mocap_id].copy()
    start_rot = to_rotation(data.mocap_quat[mocap_id])
    impact_rot = to_rotation(impact_quat)
    target_pos = np.array(target_pos, dtype=float, copy=True)
    tip_offset = np.asarray(hammer_tip_offset, dtype=float)
    if hammer_head_center_offset is None:
        hammer_rot = to_rotation(data.body("hammer_Object").xquat)
        hammer_head_center_offset = tip_offset - (start_rot.inv() * hammer_rot).apply([0, 0, 0.04])
    head_offset = np.asarray(hammer_head_center_offset, dtype=float)

    # 固定任务轴，水平正向为 horizontal_axis；左右分别对应负向与正向。
    target_frame = data.site("cylinder_endface_force_frame").xmat.reshape(3, 3).copy()
    swing_axis = target_frame[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    horizontal_axis = np.cross(swing_axis, up)
    horizontal_axis /= np.linalg.norm(horizontal_axis)

    # 未传标签时，从支架坐标系中的模型初始姿态恢复放置类别。
    if placement is None:
        initial_axes = (
            to_rotation(model.body("support_L_support").quat).inv()
            * to_rotation(model.body("hammer_Object").quat)
        ).as_matrix()
        if np.isclose(initial_axes[2, 0], 0.0):
            kind = "flat"
        elif np.isclose(abs(initial_axes[1, 1]), 1.0):
            kind = "inverted"
        else:
            kind = "lean_head_down" if initial_axes[2, 0] > 0 else "lean_head_up"
        placement = kind + ("_left" if initial_axes[0, 0] > 0 else "_right")

    # 八种放置姿态对应的固定水平撤离量；斜靠时朝远离墙的一侧撤离。
    retreat_dx = {
        "flat_left": 0.0,
        "flat_right": 0.0,
        "lean_head_up_left": retreat_distance,
        "lean_head_up_right": -retreat_distance,
        "lean_head_down_left": -retreat_distance,
        "lean_head_down_right": retreat_distance,
        "inverted_left": 0.0,
        "inverted_right": 0.0,
    }[placement]

    # 搬运阶段规划锤头中心，挥击阶段直接规划夹爪安装点。
    head_start = start_pos + start_rot.apply(head_offset)
    pivot = target_pos - impact_rot.apply(tip_offset)
    head_impact = pivot + impact_rot.apply(head_offset)

    def swing_rotation(angle):
        return Rotation.from_rotvec(angle * swing_axis) * impact_rot

    raised_rot = swing_rotation(-lift_angle)

    # 根据目标相对夹爪的水平位置确定挥击前进方向。
    approach_axis = (
        horizontal_axis
        if np.dot(target_pos - pivot, horizontal_axis) >= 0
        else -horizontal_axis
    )

    # 挥击起点：夹爪位于撞击位置上方 8 cm、远离目标 8 cm。
    gripper_strike_start = pivot + stroke_distance * up - horizontal_distance * approach_axis
    head_strike_start = gripper_strike_start + raised_rot.apply(head_offset)

    def move_segment(head0, rot0, head1, rot1, duration):
        delta_rot = (rot1 * rot0.inv()).as_rotvec()
        steps = round(duration / dt)
        for i in range(steps):
            u = (i + 1) / steps
            s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            head_pos = head0 + s * (head1 - head0)
            rot = Rotation.from_rotvec(s * delta_rot) * rot0
            step(head_pos - rot.apply(head_offset), rot)

    high_start = head_start + retreat_dx * horizontal_axis
    high_start[2] = carry_z
    high_target = head_impact.copy()
    high_target[2] = carry_z

    # 1、2：按照该放置姿态的固定路线提升、撤离、搬运；夹爪朝向不变。
    move_segment(head_start, start_rot, high_start, start_rot, lift_time)
    move_segment(high_start, start_rot, high_target, start_rot, carry_time)

    # 3：保持锤头中心不动，独立慢速调姿。
    move_segment(high_target, start_rot, high_target, raised_rot, turn_time)

    # 4：以锤头中心为参考下降至挥击起点。
    move_segment(high_target, raised_rot, head_strike_start, raised_rot, lower_time)
    for _ in range(round(settle_time / dt)):
        step(gripper_strike_start, raised_rot)

    # 5：夹爪沿世界 -Z 下降 8 cm、水平朝目标移动 8 cm，同时完成 50° 挥锤。
    steps = round(swing_time / dt)
    for i in range(steps):
        u = (i + 1) / steps
        s = u * u
        gripper_pos = pivot + (1.0 - s) * (
            stroke_distance * up - horizontal_distance * approach_axis
        )
        rot = swing_rotation(-lift_angle * (1.0 - s))
        step(gripper_pos, rot)

    # 6：保留原地旋转回收，夹爪安装点停在理论撞击位置。
    steps = round(return_time / dt)
    for i in range(steps):
        u = (i + 1) / steps
        s = u * u * (3.0 - 2.0 * u)
        rot = swing_rotation(-lift_angle * s)
        step(pivot, rot)