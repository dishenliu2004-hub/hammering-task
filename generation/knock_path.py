
"""mjlab：锤子搬运、抓取点抛物线挥击及卸载撤离。"""

from functools import partial
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from generation.hammer_generate import sync_data
from generation.impact_force import PeakImpactForce


def step_mocap(model, data, pos, quat, gripper_action, dt=0.02, viewer=None, on_step=None,
               *, simulation, on_physics_step=None):
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
        if on_physics_step is not None:
            on_physics_step()
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
    on_swing_step=None,
    on_physics_step=None,
    step_fn=None,
):
    # 两个偏移均以夹爪安装点为原点、在夹爪局部坐标系中表达。
    lift_angle = np.deg2rad(60.0)
    stroke_distance = 0.30
    horizontal_distance = 0.40
    retreat_distance = 0.00
    gripper_accel = 10.0  # 夹爪沿路径的固定切向加速度，m/s²
    return_time = 0.45
    carry_z = 0.75
    lift_time, carry_time, turn_time, lower_time = 2.0, 4.0, 4.0, 2.0
    settle_time = 1.0

    def to_rotation(q):
        return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])

    if step_fn is None:
        step_fn = partial(step_mocap, model, data, simulation=simulation)

    def step(pos, rot, on_physics_step=None, step_dt=dt):
        step_fn(
            pos, rot.as_quat()[[3, 0, 1, 2]],
            gripper_action, step_dt, viewer, on_step,
            on_physics_step=on_physics_step,
        )

    mocap_id = model.body_mocapid[model.body("gripper_mount").id]
    start_pos = data.mocap_pos[mocap_id].copy()
    start_rot = to_rotation(data.mocap_quat[mocap_id])
    impact_rot = to_rotation(impact_quat)
    target_pos = np.array(target_pos, dtype=float, copy=True)
    tip_offset = np.asarray(hammer_tip_offset, dtype=float)
    mount = data.body("gripper_base_link")
    mount_rot = Rotation.from_matrix(mount.xmat.reshape(3, 3))
    grip_center = (data.site("gripper_hand_narrow_contact_tip").xpos
                   + data.site("gripper_hand_wide_contact_tip").xpos) / 2
    grip_offset = mount_rot.inv().apply(grip_center - mount.xpos)
    if hammer_head_center_offset is None:
        hammer_rot = to_rotation(data.body("hammer_Object").xquat)
        hammer_head_center_offset = tip_offset - (mount_rot.inv() * hammer_rot).apply([0, 0, 0.04])
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

    # 搬运仍以锤头为参考，挥击以抓取中心为参考。
    head_start = start_pos + start_rot.apply(head_offset)
    pivot = target_pos - impact_rot.apply(tip_offset)
    head_impact = pivot + impact_rot.apply(head_offset)
    grip_impact = pivot + impact_rot.apply(grip_offset)

    def swing_rotation(angle):
        return Rotation.from_rotvec(angle * swing_axis) * impact_rot

    raised_rot = swing_rotation(-lift_angle)

    # 挥击方向由任务坐标系固定，不受抓取位置影响。
    approach_axis = horizontal_axis

    def grip_trajectory(s):
        s = np.asarray(s)
        return (grip_impact
                + (stroke_distance * (1.0 - s * s) + 0.10 * s * (1.0 - s))[..., None] * up
                - horizontal_distance * (1.0 - s)[..., None] * approach_axis)

    gripper_strike_start = grip_trajectory(0.0) - raised_rot.apply(grip_offset)
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

    # 5：抓取中心沿固定抛物线从静止开始匀加速。
    impact_force = PeakImpactForce(model, simulation)

    def observe():
        impact_force()
        if on_physics_step is not None:
            on_physics_step()
        return impact_force.peak

    if on_swing_step is not None:
        on_swing_step(grip_trajectory(0.0), head_strike_start)

    path_s = np.linspace(0.0, 1.0, 1001)
    path_grip = grip_trajectory(path_s)
    path_length = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(path_grip, axis=0), axis=1)),
    ))

    # 所有抓取使用同一个加速度，根据各自路径长度计算挥击时长。
    swing_duration = np.sqrt(2.0 * path_length[-1] / gripper_accel)
    steps = round(swing_duration / dt)
    swing_dt = swing_duration / steps

    for i in range(steps):
        t = (i + 1) * swing_dt

        # 夹爪沿路径从静止开始，以固定切向加速度运动。
        distance = 0.5 * gripper_accel * t * t
        s = np.interp(distance, path_length, path_s)

        # 保留原来的60°挥击旋转关系。
        rot = swing_rotation(-lift_angle * (1.0 - s))

        gripper_pos = grip_trajectory(s) - rot.apply(grip_offset)
        head_pos = gripper_pos + rot.apply(head_offset)

        if on_swing_step is not None:
            on_swing_step(grip_trajectory(s), head_pos)

        step(gripper_pos, rot, observe, step_dt=swing_dt)

    # 完整挥击后释放位姿驱动，保留速度并补偿夹爪和锤子的重力。
    impact_force.swing_complete = True
    weld_id = model.equality("gripper_target_weld").id
    moving_bodies = np.flatnonzero(np.isin(model.body_rootid, [
        model.body("gripper_dynamic_base").id, model.body("hammer_Object").id,
    ]))
    simulation.data.eq_active[0, weld_id] = False
    simulation.data.xfrc_applied[0, moving_bodies, :3] = torch.as_tensor(
        -model.body_mass[moving_bodies, None] * model.opt.gravity,
        dtype=simulation.data.xfrc_applied.dtype, device=simulation.device,
    )
    while not impact_force.impact_finished:
        step(data.mocap_pos[mocap_id].copy(), to_rotation(data.mocap_quat[mocap_id]),
             observe, step_dt=model.opt.timestep)

    # 6：从实际位姿恢复驱动，保持朝向沿端盖外法向撤离 3 cm。
    qpos_address = int(model.joint("gripper_base_free").qposadr[0])
    return_pos = data.qpos[qpos_address:qpos_address + 3].copy()
    return_rot = to_rotation(data.qpos[qpos_address + 3:qpos_address + 7])
    data.mocap_pos[mocap_id] = return_pos
    data.mocap_quat[mocap_id] = return_rot.as_quat()[[3, 0, 1, 2]]
    for name in ("mocap_pos", "mocap_quat"):
        target = getattr(simulation.data, name)
        target[0, mocap_id] = torch.as_tensor(
            getattr(data, name)[mocap_id], dtype=target.dtype, device=simulation.device,
        )
    normal = data.site("cylinder_endface_force_frame").xmat.reshape(3, 3)[:, 2].copy()
    simulation.data.xfrc_applied[0, moving_bodies, :3] = 0.0
    simulation.data.eq_active[0, weld_id] = True
    impact_force.phase = "return"
    steps = round(return_time / dt)
    for i in range(steps):
        u = (i + 1) / steps
        s = u * u * (3.0 - 2.0 * u)
        step(return_pos + 0.03 * s * normal, return_rot, observe)
    return impact_force
