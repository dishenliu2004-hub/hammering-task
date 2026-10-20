"""独立演示入口：AnyGrasp 抓取 → 自动固定锤子 → 抬升与敲击。

cd /home/ubuntu20/CodeMujoco/hammering-task
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw \
/home/ubuntu20/anaconda3/envs/hammer/bin/python generation/demo.py --pointcloud full --noise
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
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) 
from generation.grasp_hammer import generate_grasps
from generation.hammer_generate import SCENE, create_simulation, load_random_scene, sync_data
from generation.knock_path import hammer_strike, move_to
from visualization.anygrasp_grasp_visualization import show_grasps
from visualization.camera_rgb_pointcloud import CameraRGBPointCloud
from visualization.full_scene_pointcloud import FullScenePointCloud

DT = 0.02


def load_demo_scene():
    model, data, simulation = load_random_scene()
    # 新约束需要在编译模型时加入；保留原随机场景和已稳定的仿真状态。
    state_type = (int(mujoco.mjtState.mjSTATE_INTEGRATION)
                  & ~int(mujoco.mjtState.mjSTATE_EQ_ACTIVE))
    state = np.empty(mujoco.mj_stateSize(model, state_type))
    mujoco.mj_getState(model, data, state, state_type)
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.body("hammer_Object").pos[:] = model.body("hammer_Object").pos
    spec.body("hammer_Object").quat[:] = model.body("hammer_Object").quat
    spec.option.cone = model.opt.cone
    spec.option.noslip_iterations = model.opt.noslip_iterations
    spec.geom("hammer_head").condim = int(model.geom("hammer_head").condim[0])
    spec.add_equality(
        name="demo_hammer_grip", type=mujoco.mjtEq.mjEQ_WELD,
        name1="gripper_mount", name2="hammer_Object", objtype=mujoco.mjtObj.mjOBJ_BODY,
        active=False, data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        solref=[0.004, 1], solimp=[0.999, 0.999, 0.001, 0.5, 2],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, state_type)
    mujoco.mj_forward(model, data)
    simulation = create_simulation(model, data)
    return model, data, simulation


def attach_hammer(model, data, simulation):
    # 闭爪后锁定当前相对位姿，不移动锤子，也不依赖手指摩擦抓持。
    sync_data(simulation, data)
    mount, hammer = data.body("gripper_mount"), data.body("hammer_Object")
    mount_rot = Rotation.from_matrix(mount.xmat.reshape(3, 3))
    hammer_rot = Rotation.from_matrix(hammer.xmat.reshape(3, 3))
    weld = model.equality("demo_hammer_grip")
    weld.data[3:6] = mount_rot.inv().apply(hammer.xpos - mount.xpos)
    weld.data[6:10] = (mount_rot.inv() * hammer_rot).as_quat()[[3, 0, 1, 2]]
    data.eq_active[weld.id] = True
    simulation.model.eq_data[0, weld.id] = torch.as_tensor(
        weld.data, dtype=simulation.model.eq_data.dtype, device=simulation.device,
    )
    simulation.data.eq_active[0, weld.id] = True
    sync_data(simulation, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", choices=("camera", "full"), default="camera",
                        help="AnyGrasp 点云来源：camera 相机视角，full 完整场景（默认 camera）")
    parser.add_argument("--noise", action="store_true", help="为所选点云来源加噪（默认关闭）")
    args = parser.parse_args()
    full_scene = args.pointcloud == "full"
    log_config = mujoco.MjLogConfig.get()
    log_config.logto_file = False
    log_config.set()
    model, data, simulation = load_demo_scene()
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
            move(pose["pregrasp_pos"], mount_quat, 1.0, 0.0)
            move(mount_pos, mount_quat, 0.6, 0.0)
            move(mount_pos, mount_quat, 0.8, 1.0)
            attach_hammer(model, data, simulation)

            # 固定后仍执行原抬升轨迹和 4 厘米抓稳判断。
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
            hammer_strike(
                model, data, mount_body.xpos.copy(), mount_body.xquat.copy(),
                target_site.xpos.copy(), tip_offset, impact_mount_rot.as_quat()[[3, 0, 1, 2]],
                gripper_action=1.0, dt=DT, viewer=viewer, on_step=process_events,
                simulation=simulation,
            )
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
