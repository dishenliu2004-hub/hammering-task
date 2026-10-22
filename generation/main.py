"""
总入口：相机显示 → AnyGrasp 抓姿 → 抓取并抬升 → 敲击。

cd /home/ubuntu20/CodeMujoco/hammering-task

PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw \
/home/ubuntu20/anaconda3/envs/hammer/bin/python generation/main.py \
--pointcloud full --noise --hammer-head-extra-mass 0.1
将 camera 替换为 full 使用完整场景点云；省略参数时默认 camera。
追加 --noise 为所选点云来源加噪，默认无噪，两种来源均支持。
"""

import argparse
from functools import partial
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.grasp_hammer import generate_grasps
from generation.hammer_generate import load_random_scene
from generation.knock_path import hammer_strike, move_to
from visualization.anygrasp_grasp_visualization import show_grasps
from visualization.camera_rgb_pointcloud import CameraRGBPointCloud
from visualization.full_scene_pointcloud import FullScenePointCloud
from visualization.swing_trajectory import SwingTrajectory

DT = 0.02


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", choices=("camera", "full"), default="camera",
                        help="AnyGrasp 点云来源：camera 相机视角，full 完整场景（默认 camera）")
    parser.add_argument("--noise", action="store_true", help="为所选点云来源加噪（默认关闭）")
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0,
                        help="锤头额外增加的质量，单位 kg（默认 0）")
    args = parser.parse_args()
    full_scene = args.pointcloud == "full"
    log_config = mujoco.MjLogConfig.get()
    log_config.logto_file = False
    log_config.set()
    model, data, simulation = load_random_scene(args.hammer_head_extra_mass)
    if not full_scene:
        camera = CameraRGBPointCloud(model, noise=args.noise)
    grasp_figure = None
    try:
        if full_scene:
            plt.ion()
            frame = FullScenePointCloud(model, noise=args.noise).capture(data)
        else:
            frame = camera.capture(data)
            camera.update(frame)
        grasps, pose = generate_grasps(model, data, frame, full_scene=full_scene)
        grasp_figure = show_grasps(frame, grasps)
        # 窗口保留抓取前的数据，运动期间仅处理鼠标和窗口事件。
        process_events = grasp_figure.canvas.flush_events

        with mujoco.viewer.launch_passive(model, data) as viewer:
            move = partial(move_to, model, data, viewer, dt=DT, on_step=process_events,
                           simulation=simulation)
            mount_pos, mount_quat = pose["mount_pos"], pose["mount_quat"]
            move(pose["pregrasp_pos"], mount_quat, 1.0, pose["gripper_open_ctrl"])
            move(mount_pos, mount_quat, 0.6, pose["gripper_open_ctrl"])
            move(mount_pos, mount_quat, 0.8, 1.0)

            # 沿用原抓稳判断和抬升轨迹，不重新创建场景。
            mocap_id = model.body_mocapid[model.body("gripper_mount").id]
            mount_rot = Rotation.from_quat(mount_quat[[1, 2, 3, 0]])
            hammer_pos = data.body("hammer_Object").xpos.copy()
            grasp_relative = mount_rot.inv().apply(hammer_pos - data.mocap_pos[mocap_id])
            move(mount_pos + [0, 0, 0.25], mount_quat, 0.8, 1.0)
            lifted_relative = mount_rot.inv().apply(
                data.body("hammer_Object").xpos - data.mocap_pos[mocap_id]
            )
            if np.linalg.norm(lifted_relative - grasp_relative) > 0.04:
                raise RuntimeError("锤子未抓稳，未执行敲击")

            mount_body, hammer_body = data.body("gripper_base_link"), data.body("hammer_Object")
            actual_mount_rot = Rotation.from_matrix(mount_body.xmat.reshape(3, 3))
            actual_hammer_rot = Rotation.from_matrix(hammer_body.xmat.reshape(3, 3))
            face_pos = hammer_body.xpos + actual_hammer_rot.apply([-0.24, 0, 0.04])
            tip_offset = actual_mount_rot.inv().apply(face_pos - mount_body.xpos)
            relative_rot = actual_mount_rot.inv() * actual_hammer_rot
            target_site = data.site("cylinder_endface_force_frame")
            impact_hammer_rot = (Rotation.from_matrix(target_site.xmat.reshape(3, 3))
                                 * Rotation.from_euler("y", np.pi))
            impact_mount_rot = impact_hammer_rot * relative_rot.inv()
            swing_trajectory = SwingTrajectory(viewer)
            impact_force = hammer_strike(
                model, data, mount_body.xpos.copy(), mount_body.xquat.copy(),
                target_site.xpos.copy(), tip_offset, impact_mount_rot.as_quat()[[3, 0, 1, 2]],
                gripper_action=1.0, dt=DT, viewer=viewer, on_step=process_events,
                simulation=simulation, on_swing_step=swing_trajectory,
            )
            print(f"挥击法向接触力峰值（含撞击及卸载）：{impact_force.swing_peak:.6f} N")
            print(f"撤离法向接触力峰值：{impact_force.return_peak:.6f} N")
            print(f"圆柱上表面撞击峰值：{impact_force.peak:.6f} N")
            print(f"挥击开始时夹爪抓取中心到锤头中心距离：{impact_force.distance:.6f} m")
            # 动作完成后继续保留初始画面，直到关闭 MuJoCo 窗口。
            while viewer.is_running():
                process_events()
                time.sleep(0.05)
    finally:
        if not full_scene:
            camera.close()
        plt.close(grasp_figure)


if __name__ == "__main__":
    main()
