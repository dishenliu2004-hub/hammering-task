"""从固定挥击起点生成悬空 B 目标，直接渲染并缓存 AnyGrasp 候选。"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from generation.hammer_generate import SCENE, set_hammer_pose
from generation.hammer_geometry import HEAD_CENTER_LOCAL, STRIKE_FACE_CENTER_LOCAL, B_SPHERE_RADIUS
from generation.knock_path import compute_nominal_b_pose
from visualization.camera_rgb_pointcloud import CameraRGBPointCloud
from .prepare import WORKSPACE, bank_arrays, capture_candidates, stratified_split


def sample_b_poses(data, count, seed):
    nominal = compute_nominal_b_pose(
        data.site("cylinder_endface_force_frame").xpos,
        data.body("cylinder_sensing_endcap").xquat,
        data.site("cylinder_endface_force_frame").xmat.reshape(3, 3))
    rotation = Rotation.from_quat(nominal["hammer_quat"][[1, 2, 3, 0]])
    distance = np.linalg.norm(STRIKE_FACE_CENTER_LOCAL - HEAD_CENTER_LOCAL)
    theta_max = 2 * np.arcsin(B_SPHERE_RADIUS / (2 * distance))
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < count:
        axis = rng.normal(size=3)
        delta = axis / np.linalg.norm(axis) * rng.uniform(0., theta_max)
        sample_rotation = Rotation.from_rotvec(delta) * rotation
        position = nominal["head_center"] - sample_rotation.apply(HEAD_CENTER_LOCAL)
        face = position + sample_rotation.apply(STRIKE_FACE_CENTER_LOCAL)
        face_distance = np.linalg.norm(face - nominal["face_center"])
        if face_distance > B_SPHERE_RADIUS or any(
            np.linalg.norm(face - sample["face_center"]) < 1e-8 for sample in samples
        ):
            continue
        samples.append(dict(b_pose_id=np.int64(len(samples)), hammer_pos=position,
                            hammer_quat=sample_rotation.as_quat()[[3, 0, 1, 2]],
                            head_center=nominal["head_center"].copy(), face_center=face,
                            nominal_face_center=nominal["face_center"].copy(),
                            rotation_delta=delta, face_center_distance=face_distance))
    return samples


def hammer_workspace(model, data):
    """变换模型局部包围盒，覆盖 B 姿态下全部锤体几何。"""
    corners = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T
    points = []
    for name in ("hammer_handle", "hammer_neck", "hammer_head"):
        geom = model.geom(name)
        center, extent = model.geom_aabb[geom.id].reshape(2, 3)
        local = center + corners * extent
        points.extend(local @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id])
    points = np.asarray(points)
    return points.min(0) - 0.02, points.max(0) + 0.02


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="weight/hammer_rl/b_pose_bank_40.npz")
    parser.add_argument("--num-poses", type=int, default=40)
    parser.add_argument("--candidates-per-pose", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    samples = sample_b_poses(data, args.num_poses, args.seed)
    set_hammer_pose(model, data, samples[0]["hammer_pos"], samples[0]["hammer_quat"])
    lower, upper = hammer_workspace(model, data)
    camera_id = model.camera("fixed").id
    camera_distance = np.linalg.norm(data.cam_xpos[camera_id] - np.mean(WORKSPACE, axis=0))
    camera_position = ((lower + upper) / 2
                       + camera_distance * data.cam_xmat[camera_id].reshape(3, 3)[:, 2])
    # 原相机只覆盖支架；B 仅平移相机安装体，保留方向、内参与采集流程。
    model.body("camera_mount").pos[:] += camera_position - data.cam_xpos[camera_id]
    mujoco.mj_forward(model, data)
    candidates, offsets = [], [0]
    for sample in samples:
        set_hammer_pose(model, data, sample["hammer_pos"], sample["hammer_quat"])
        camera = CameraRGBPointCloud(model)
        captured, seeds = capture_candidates(model, data, camera, sample,
                                             args.candidates_per_pose, hammer_workspace(model, data))
        sample.update(captured)
        candidates.extend(seeds)
        offsets.append(len(candidates))
        print(f"B {sample['b_pose_id']}: candidates={len(seeds)}", flush=True)
    arrays = bank_arrays(samples, candidates, offsets)
    arrays.update(split=stratified_split(np.zeros(args.num_poses, dtype=int), args.seed),
                  sphere_radius=np.array(B_SPHERE_RADIUS))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"saved {output.resolve()}: {len(samples)} B poses, {len(candidates)} candidates")


if __name__ == "__main__":
    main()
