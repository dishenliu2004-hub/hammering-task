"""在工具坐标系中对照人手抓取和 OmniPicker 位姿。"""

import numpy as np


def unit(vector):
    return vector / np.linalg.norm(vector)


def grasp_frame(closing, approach):
    x = unit(closing)
    z = unit(approach - np.dot(approach, x) * x)
    return np.column_stack((x, np.cross(z, x), z))


def mapped_gripper_frame(hand):
    closing = np.asarray(hand["closing_axis_O"])
    approach = unit(np.asarray(hand["approach_axis_O"]))
    # 固定开口朝向，在其法平面内对齐两侧指尖。
    closing = closing - np.dot(closing, approach) * approach
    return grasp_frame(closing, approach)


def encode(hand, gripper, handle_size):
    gripper_frame = mapped_gripper_frame(hand)
    reference = np.asarray(gripper["reference_rotation"])
    u_rl = 2 * hand["u_mean"] - 1
    center = np.array([u_rl * handle_size[1], 0.0, 0.0])
    candidates = []
    for sign in (1, -1):
        rotation = gripper_frame @ np.diag([sign, sign, 1]) @ reference.T
        base = center - rotation @ gripper["home_tcp_offset_G_m"]
        candidates.append(dict(rotation_OG=rotation.tolist(), base_pos_O_m=base.tolist()))
    return dict(
        version="2.0",
        coordinates=dict(frame="tool_canonical", length_unit="m",
                         x_axis="working_end_to_free_end", z_axis="toward_striking_face",
                         u_RL="x / handle_half_length"),
        hand_grasp=dict(hand),
        gripper=gripper,
        handle_size_m=handle_size,
        mapped_grasp=dict(grip_center_O_m=center.tolist(),
                          preferred_u_RL=u_rl, u_sigma_RL=2 * hand["u_std"],
                          preferred_closing_axis_O=gripper_frame[:, 0].tolist(),
                          preferred_approach_axis_O=gripper_frame[:, 2].tolist(),
                          pose_candidates=candidates),
    )
