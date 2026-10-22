"""生成 AnyGrasp 候选和 Omnipicker 抓取目标；由 main.py 调用。"""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ANYGRASP_PY = "/home/ubuntu20/anaconda3/envs/hammer/bin/python"


def grasp_width_to_joint_angle(width):
    """宽度（米）对应的驱动角度（弧度）、开合指令及指尖中心偏移。"""
    import mujoco
    from scipy.optimize import brentq
    from generation.hammer_generate import create_simulation, sync_data, write_data

    # 独立夹爪的无接触响应保留闭环连杆、被动弹簧和执行器特性。
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/grippers/omnipicker/ominipicker.xml"))
    model.opt.gravity[:] = 0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    simulation = create_simulation(model, data)

    def opening_error(control):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.ctrl[model.actuator("gripper_actuator").id] = control
        write_data(simulation, data)
        for _ in range(round(2.0 / model.opt.timestep)):
            simulation.step()
        sync_data(simulation, data)
        return (data.site("hand_wide_contact_tip").xpos[1]
                - data.site("hand_narrow_contact_tip").xpos[1] - width)

    control = brentq(opening_error, 0.0, 1.0, xtol=1e-5)
    opening_error(control)
    angle = data.qpos[model.joint("hand_narrow1_joint").qposadr[0]]
    tcp_offset = (data.site("hand_narrow_contact_tip").xpos
                  + data.site("hand_wide_contact_tip").xpos) / 2
    return float(angle), control, tcp_offset


@contextmanager
def anygrasp_runtime():
    runtime = ROOT / "weight" / "local_runtime"
    if not (runtime / "license/licenseCfg.json").is_file():
        raise FileNotFoundError("AnyGrasp 授权配置不存在：{}".format(runtime / "license/licenseCfg.json"))
    with TemporaryDirectory(prefix="hammer_anygrasp_") as directory:
        sdk_runtime = Path(directory)
        # SDK 按扩展文件所在目录查找 license；用链接保留原文件布局和内容。
        for name, source in (
            ("gsnet.so", runtime / "anygrasp/grasp_detection/gsnet_versions/gsnet.cpython-311-x86_64-linux-gnu.so"),
            ("lib_cxx.so", runtime / "anygrasp/license_registration/lib_cxx_versions/lib_cxx.cpython-311-x86_64-linux-gnu.so"),
            ("license", runtime / "license"),
        ):
            if not source.exists():
                raise FileNotFoundError("AnyGrasp 运行文件不存在：{}".format(source))
            (sdk_runtime / name).symlink_to(source, target_is_directory=source.is_dir())
        previous_cwd, previous_path = Path.cwd(), sys.path[:]
        sys.path[:0] = [str(p) for p in (
            sdk_runtime,
            runtime / "anygrasp/pointnet2/build/lib.linux-x86_64-cpython-311",
            runtime / "graspnetAPI",
        )]
        try:
            os.chdir(sdk_runtime)
            yield
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path


def infer(input_path, output_path):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    with anygrasp_runtime():
        # 此 SDK 仍使用 NumPy 1.24 移除的别名；只在推理进程补充兼容。
        if not hasattr(np, "float"):
            np.float = float
        from gsnet import AnyGrasp

        cfg = SimpleNamespace(
            checkpoint_path=str(ROOT / "weight/pretrained_weights/anygrasp/checkpoint_detection.tar"),
            max_gripper_width=0.1, gripper_height=0.03,
            top_down_grasp=False, debug=False,
        )
        net = AnyGrasp(cfg)
        net.load_net()
        with np.load(input_path) as cloud:
            gg, _ = net.get_grasp(
                cloud["points"], cloud["colors"], lims=cloud["lims"],
                apply_object_mask=True, dense_grasp=True, collision_detection=True,
            )
        if gg is None or len(gg) == 0:
            raise RuntimeError("AnyGrasp 没有生成抓取候选")
        gg = gg.nms().sort_by_score()
        np.savez(
            output_path, translations=gg.translations, rotations=gg.rotation_matrices,
            depths=gg.depths, widths=gg.widths, scores=gg.scores,
        )


def generate_grasps(model, data, frame, *, full_scene=False):
    # 完整点云按锤柄和锤头颜色定位裁剪；相机点云沿用锤柄附近 0.30 米的工作空间。
    points, colors = frame["points"], frame["colors"]
    handle = data.geom("hammer_handle")
    if full_scene:
        hammer = (np.all(np.isclose(colors, [0.60, 0.39, 0.23]), axis=1)
                  | np.all(np.isclose(colors, [0.40, 0.40, 0.40]), axis=1))
        hammer_points = frame["world_points"][hammer]
        lower, upper = hammer_points.min(axis=0) - 0.07, hammer_points.max(axis=0) + 0.07
        workspace = np.all((frame["world_points"] >= lower)
                           & (frame["world_points"] <= upper), axis=1)
    else:
        workspace = np.linalg.norm(frame["world_points"] - handle.xpos, axis=1) < 0.30
    points, colors = points[workspace], colors[workspace]
    # 保留上下文点云，用相机坐标下锤柄外扩 0.02 米的包围盒限定抓取工作区。
    center = (handle.xpos - frame["camera_pos"]) @ frame["camera_rot"]
    axis = frame["camera_rot"].T @ handle.xmat.reshape(3, 3)[:, 2]
    radius, half_length = model.geom("hammer_handle").size[:2]
    extent = half_length * np.abs(axis) + radius + 0.02
    lims = np.column_stack((center - extent, center + extent)).ravel()

    with TemporaryDirectory() as directory:
        input_path = Path(directory) / "cloud.npz"
        output_path = Path(directory) / "grasps.npz"
        np.savez(input_path, points=points, colors=colors, lims=lims)
        subprocess.run(
            [ANYGRASP_PY, str(Path(__file__).resolve()), "--infer", str(input_path), str(output_path)],
            cwd=ROOT / "weight/local_runtime", check=True,
        )
        if not output_path.is_file():
            raise RuntimeError(
                "AnyGrasp 子进程未生成抓取结果，可能在授权或 SDK 加载阶段提前退出；"
                "请检查上方 AnyGrasp 日志。"
            )
        with np.load(output_path) as output:
            grasps = {name: output[name] for name in output.files}

    # 返回的全部候选使用世界坐标；原始平移参考点不在这里叠加抓取深度。
    grasps["translations"] = grasps["translations"] @ frame["camera_rot"].T + frame["camera_pos"]
    grasps["rotations"] = frame["camera_rot"] @ grasps["rotations"]
    i = 0

    _, gripper_open_ctrl, tcp_offset = grasp_width_to_joint_angle(grasps["widths"][i])

    grasp_rot = grasps["rotations"][i]
    mount_rot = Rotation.from_matrix(grasp_rot) * Rotation.from_euler("y", np.pi / 2)
    mount_quat = mount_rot.as_quat()[[3, 0, 1, 2]]
    mount_pos = (grasps["translations"][i] + grasps["depths"][i] * grasp_rot[:, 0]
                 - mount_rot.apply(tcp_offset))
    mount_pos += 0.03 * grasp_rot[:, 0] #新增，沿接近方向额外深入30mm
    pregrasp_pos = mount_pos - 0.10 * grasp_rot[:, 0]

    return grasps, {
        "pregrasp_pos": pregrasp_pos,
        "mount_pos": mount_pos,
        "mount_quat": mount_quat,
        "gripper_open_ctrl": gripper_open_ctrl,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--infer":
        infer(Path(sys.argv[2]), Path(sys.argv[3]))
