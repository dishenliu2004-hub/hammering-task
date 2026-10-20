"""运行原抓取到敲击流程；--episodes 指定顺序执行次数。

conda activate hammer
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw \
python -m hammer_rl.evaluate --pointcloud full --noise --episodes 5
"""

import argparse
from functools import partial
import time

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from mjlab.envs import ManagerBasedRlEnv

from generation.grasp_hammer import generate_grasps
from visualization.anygrasp_grasp_visualization import show_grasps
from visualization.camera_rgb_pointcloud import CameraRGBPointCloud
from visualization.full_scene_pointcloud import FullScenePointCloud

from .env_cfg import DT, make_env_cfg
from .mdp.actions import hammer_strike, move_to
from .mdp.observations import sync_data


def run_episode(env, *, pointcloud="camera", noise=False, keep_open=True):
    env.reset()
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    sync_data(env, data)
    full_scene = pointcloud == "full"
    if not full_scene:
        camera = CameraRGBPointCloud(model, noise=noise)
    grasp_figure = None
    try:
        if full_scene:
            plt.ion()
            frame = FullScenePointCloud(model, noise=noise).capture(data)
        else:
            frame = camera.capture(data)
            camera.update(frame)
        grasps, pose = generate_grasps(model, data, frame, full_scene=full_scene)
        grasp_figure = show_grasps(frame, grasps)
        # 窗口保留抓取前的数据，运动期间仅处理鼠标和窗口事件。
        process_events = grasp_figure.canvas.flush_events

        with mujoco.viewer.launch_passive(model, data) as viewer:
            move = partial(move_to, env, data, viewer, dt=DT, on_step=process_events)
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
            """if np.linalg.norm(lifted_relative - grasp_relative) > 0.04:
                raise RuntimeError("锤子未抓稳，未执行敲击")"""

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
            hammer_strike(
                env, data,
                target_site.xpos.copy(), tip_offset, impact_mount_rot.as_quat()[[3, 0, 1, 2]],
                gripper_action=1.0, dt=DT, viewer=viewer, on_step=process_events,
            )
            # 动作完成后继续保留初始画面，直到关闭 MuJoCo 窗口。
            while keep_open and viewer.is_running():
                process_events()
                time.sleep(0.05)
    finally:
        if not full_scene:
            camera.close()
        plt.close(grasp_figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", choices=("camera", "full"), default="camera",
                        help="AnyGrasp 点云来源：camera 相机视角，full 完整场景（默认 camera）")
    parser.add_argument("--noise", action="store_true", help="为所选点云来源加噪（默认关闭）")
    parser.add_argument("--episodes", type=int, default=1, help="顺序执行次数（默认 1）")
    args = parser.parse_args()
    log_config = mujoco.MjLogConfig.get()
    log_config.logto_file = False
    log_config.set()
    env = ManagerBasedRlEnv(make_env_cfg(), device="cuda:0")
    try:
        for episode in range(args.episodes):
            run_episode(env, pointcloud=args.pointcloud, noise=args.noise,
                        keep_open=episode == args.episodes - 1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
