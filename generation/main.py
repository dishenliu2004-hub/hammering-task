"""
总入口：相机显示 → AnyGrasp 抓姿 → 共用执行器抓取、抬升并敲击三次。

cd /home/ubuntu20/CodeMujoco/hammering-task

PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw \
/home/ubuntu20/anaconda3/envs/hammer/bin/python generation/main.py \
--pointcloud full --noise --hammer-head-extra-mass 0.1
将 camera 替换为 full 使用完整场景点云；省略参数时默认 camera。
追加 --noise 为所选点云来源加噪，默认无噪，两种来源均支持。
"""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.grasp_hammer import generate_grasps
from generation.hammer_generate import load_random_scene
from hammer_rl.env_cfg import make_env_cfg
from generation.actions import DT
from generation.observations import sync_data
from hammer_rl.prepare import (bank_arrays, capture_observation, capture_scene_state,
                               prepare_candidates)
from visualization.anygrasp_grasp_visualization import show_grasps
from visualization.camera_rgb_pointcloud import CameraRGBPointCloud
from visualization.full_scene_pointcloud import FullScenePointCloud
from visualization.swing_trajectory import SwingTrajectory


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
            captured, _ = capture_observation(data, CameraRGBPointCloud(model))
        else:
            captured, frame = capture_observation(data, camera, close_camera=False)
            camera.update(frame)
        sample = capture_scene_state(model, data)
        sample.update(captured)
        grasps, _ = generate_grasps(model, data, frame, full_scene=full_scene,
                                   convert_first=False)
        grasp_figure = show_grasps(frame, grasps)
        # 窗口保留抓取前的数据，运动期间仅处理鼠标和窗口事件。
        process_events = grasp_figure.canvas.flush_events

        candidates = prepare_candidates(grasps, sample, 1)
        arrays = bank_arrays([sample], candidates, [0, 1])
        arrays.update(split=np.zeros(1, dtype=np.int64),
                      hammer_head_extra_mass=np.asarray(args.hammer_head_extra_mass))
        device = simulation.device
        del simulation
        with TemporaryDirectory(prefix="hammer_live_") as directory:
            bank = Path(directory) / "scene.npz"
            np.savez(bank, **arrays)
            cfg = make_env_cfg(args.hammer_head_extra_mass, data_file=bank,
                               num_envs=1, seed=None, training=False)
            cfg.randomize_physics = False
            cfg.events.pop("physics")
            cfg.auto_reset = False
            env = ManagerBasedRlEnv(cfg, device=device)
            try:
                env.reset()
                model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
                sync_data(env, data)
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    swing_trajectory = SwingTrajectory(viewer)

                    def display(term):
                        if term._substep % term.substeps == 0:
                            start = time.monotonic()
                            sync_data(env, data)
                            if term.swing_start <= term._substep <= term.swing_end:
                                rot = Rotation.from_quat(data.mocap_quat[term.mocap][[1, 2, 3, 0]])
                                offsets = torch.stack((term.grip_offset[0], term.head_offset[0])).cpu().numpy()
                                grip, head = data.mocap_pos[term.mocap] + rot.apply(offsets)
                                swing_trajectory(grip, head)
                            viewer.sync()
                            process_events()
                            time.sleep(max(0.0, DT - (time.monotonic() - start)))

                    term = env.action_manager.get_term("grasp")
                    term.on_physics_step = display
                    with torch.inference_mode():
                        env.step(torch.zeros((1, env.action_manager.total_action_dim), device=device))
                    for name in ("strike_force_1", "strike_force_2", "strike_force_3", "mean_force"):
                        print(f"{name}: {float(term.values[name][0]):.6f} N")
                    print(f"hit_count: {int(term.values['hit_count'][0])}/3")
                    print(f"task_success: {int(term.values['task_success'][0])}")
                    # 动作完成后保留终局和抓取前画面，直到关闭 MuJoCo 窗口。
                    while viewer.is_running():
                        process_events()
                        time.sleep(0.05)
            finally:
                env.close()
    finally:
        if not full_scene:
            camera.close()
        plt.close(grasp_figure)


if __name__ == "__main__":
    main()
