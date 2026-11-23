"""在工具坐标系中对照人手抓取和 OmniPicker 位姿。"""

import numpy as np

HAND_TIPS = [8, 4]  # 食指尖、拇指尖，对应候选一的窄爪、宽爪。
PALM_KNUCKLES = [5, 9, 13, 17]  # 食指到小指的掌指关节。


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


def encode(hand, gripper, handle_size, *, record, source_coordinates):
    origin = np.asarray(source_coordinates["canonical_origin_in_source"])
    axes = np.asarray(source_coordinates["canonical_axes_in_source"])
    hand_points = (record["hands"][0] - origin) @ axes
    # ContactPose 以工作端为原点，对照图以原锤柄的中点为原点。
    hand_points[:, 0] -= source_coordinates["handle_length_m"] / 2
    palm_index = len(hand_points)
    palm_center = (hand_points[0] + hand_points[PALM_KNUCKLES].mean(axis=0)) / 2
    hand_points = np.vstack((hand_points, palm_center))

    gripper_frame = mapped_gripper_frame(hand)
    reference = np.asarray(gripper["reference_rotation"])
    u_rl = 2 * hand["u_mean"] - 1
    center = np.array([u_rl * handle_size[1], 0.0, 0.0])
    contact_sites = gripper["contact_sites"]
    candidates = []
    for sign, sites in ((1, contact_sites), (-1, contact_sites[::-1])):
        rotation = gripper_frame @ np.diag([sign, sign, 1]) @ reference.T
        base = center - rotation @ gripper["home_tcp_offset_G_m"]
        candidates.append(dict(
            rotation_OG=rotation.tolist(),
            base_pos_O_m=base.tolist(),
            keypoint_correspondences=[
                dict(hand_keypoint_index=index, gripper_site=site)
                for index, site in zip(HAND_TIPS, sites)
            ],
        ))
    return dict(
        version="2.0",
        coordinates=dict(frame="tool_canonical", origin="handle_midpoint", length_unit="m",
                         x_axis="working_end_to_free_end", z_axis="toward_striking_face",
                         u_RL="x / handle_half_length"),
        hand_grasp=dict(hand, keypoints_O_m=hand_points.tolist(),
                        links=record["hand_links"].tolist()),
        gripper=gripper,
        handle_size_m=handle_size,
        mapped_grasp=dict(grip_center_O_m=center.tolist(),
                          preferred_u_RL=u_rl, u_sigma_RL=2 * hand["u_std"],
                          preferred_closing_axis_O=gripper_frame[:, 0].tolist(),
                          preferred_approach_axis_O=gripper_frame[:, 2].tolist(),
                          base_keypoint_correspondence=dict(hand_keypoint_index=palm_index,
                                                            gripper_keypoint_index=0),
                          pose_candidates=candidates),
    )
