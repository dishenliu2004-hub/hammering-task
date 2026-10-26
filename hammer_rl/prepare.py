"""离线生成落稳初态、固定相机抓前深度与真实 AnyGrasp 候选。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from generation.grasp_hammer import generate_grasps, candidate_to_mount
from generation.hammer_generate import create_simulation, sync_data, SCENE
from generation.hammer_inertia import add_head_mass
from visualization.camera_rgb_pointcloud import CameraRGBPointCloud

STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active",
                "qfrc_applied", "xfrc_applied", "qacc_warmstart")
# 固定标定工作区：支架及其上方；不依赖本次锤子真值位姿。
WORKSPACE = ((0.14, -0.16, 0.11), (0.46, 0.41, 0.72))
PLACEMENTS = ("flat", "lean_head_down", "lean_head_up", "inverted")


def create_scene(rng, scene, hammer_head_extra_mass=0.0):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(spec, hammer_head_extra_mass)
    model = spec.compile()
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.geom("hammer_head").condim[0] = 6
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # position 和 local_quat 均相对于 L 形支架坐标系；四元数采用 wxyz。
    # 支架局部 Z 向上、X 沿底板长边、Y 沿宽度，顶部高度为 0.37 米。
    local_quat = np.zeros(4)
    # 等概率选择 0、1、2、3 四类初始摆放。

    if scene == 0:
        # 横躺：圆形敲击面朝下，沿底板长边随机偏转、平移。
        # yaw 在正负 12 度内采样；绕锤柄轴翻转 180 度后绕支架局部 Z 轴偏转。
        yaw = rng.uniform(-np.pi / 15, np.pi / 15)
        mujoco.mju_euler2Quat(local_quat, np.array([yaw, np.pi, 0.]), "zxy")
        # 在底板上小范围平移；固定高度用于设置接近接触面的初始状态。
        position = np.array([rng.uniform(0.29, 0.33), rng.uniform(-0.02, 0.02), 0.1605])

    elif scene == 1:
        # 斜靠：锤头落在底板上，锤柄接触板面或顶部边缘。
        # theta 为倾斜角，范围 30～90 度；s、c 简写其正弦和余弦。
        theta = rng.uniform(np.pi / 6, np.pi / 2)
        s, c = np.sin(theta), np.cos(theta)
        # r 为锤柄半径 0.025 米加 0.0005 米接触间隙。
        r = 0.0255
        # 0.1405 是由底板高度和锤头半径确定的基准高度；
        # 0.24*sin(theta) 补偿锤头相对锤体原点的高度差。
        z = 0.1405 + 0.24 * s
        # h 表示锤柄端部圆弧中心相对支架顶部的高度差。
        # t 是辅助轴向距离，与锤柄圆柱段半长 0.05 米比较以区分接触形态。
        h = z + 0.05 * s - 0.37
        t = (r * c + 0.37 - z) / s
        if h <= 0:
            # 端部圆弧中心未高于顶部，按与竖直板内侧接触计算横向位置。
            x = 0.02 + 0.05 * c + r
        elif t > 0.05:
            # 端部圆弧靠在顶部边缘，用圆的几何关系求水平间距。
            x = 0.02 + 0.05 * c + np.sqrt(r * r - h * h)
        else:
            # 接触位置涉及倾斜柄身，按柄身与顶部边缘相切求位置。
            x = 0.02 + (r + c * (0.37 - z)) / s
        # Y 方向随机移动；yzx 指定欧拉角序列，结果存入 wxyz 四元数。
        position = np.array([x, rng.uniform(-0.05, 0.05), z])
        mujoco.mju_euler2Quat(local_quat, np.array([theta, np.pi, np.pi / 2]), "yzx")

    elif scene == 2:
        # 斜靠：锤柄落在底板上，锤头接触板面或顶部边缘。
        # 上限限制锤头相对顶部的高度差不超过接触半径，保证圆弧公式有效。
        theta = rng.uniform(np.pi / 6, np.arcsin(0.245 / 0.29))
        s, c = np.sin(theta), np.cos(theta)
        # r 为锤头半径 0.020 米加 0.0005 米接触间隙。
        r = 0.0205
        # 按锤柄落在底板上的几何关系设置锤体原点高度。
        z = 0.1455 + 0.05 * s
        # 锤头中心相对顶部的高度差；未超过顶部时靠板面，否则靠顶部边缘。
        h = z + 0.24 * s - 0.37
        x = 0.02 + 0.24 * c + (r if h <= 0 else np.sqrt(r * r - h * h))
        position = np.array([x, rng.uniform(-0.05, 0.05), z])
        mujoco.mju_euler2Quat(local_quat, np.array([theta, 0., np.pi / 2]), "yzx")

    else:
        # 倒立：锤柄朝上，锤头削出的 2cm×6cm 平面朝下并接触底板。
        mujoco.mju_axisAngle2Quat(local_quat, np.array([0., 1., 0.]), -np.pi / 2)
        # 底板顶面为0.12m；锤头削平面距锤体原点0.25722345m，留0.5mm间隙。
        position = np.array([rng.uniform(0.29, 0.33), rng.uniform(-0.02, 0.02),
                             0.12 + 0.25722345 + 0.0005])

    # 获取支架当前世界位姿；名称带有总场景 attach 设置的 support_ 前缀。
    support = data.body("support_L_support")
    # 此处修改 model 中锤子的初始参考位姿，用于场景初始化。
    # 仿真运行后读取实际锤子位姿应使用 data.body("hammer_Object")。
    hammer = model.body("hammer_Object")
    # 世界位置 = 支架世界位置 + 支架旋转矩阵 × 支架局部位置。
    hammer.pos[:] = support.xpos + support.xmat.reshape(3, 3) @ position
    # 世界姿态 = 支架世界姿态 × 锤子相对支架的局部姿态。
    mujoco.mju_mulQuat(hammer.quat, support.xquat, local_quat)
    # 初始参考位姿变更后，重新计算模型常量和当前状态的派生量。
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)

    difficulty = (abs(yaw) / (np.pi / 15) if scene == 0 else
                  (theta - np.pi / 6) / ((np.pi / 2 if scene == 1 else
                    np.arcsin(0.245 / 0.29)) - np.pi / 6) if scene in (1, 2) else 1.0)
    simulation = create_simulation(model, data)
    for _ in range(round(2.0 / model.opt.timestep)):
        simulation.step()
    simulation.data.time[:] = 0
    sync_data(simulation, data)
    return model, data, simulation, difficulty


def prepare_scene(rng, placement, candidates_per_scene, hammer_head_extra_mass=0.0):
    model, data, simulation, difficulty = create_scene(rng, placement, hammer_head_extra_mass)
    sample = {name: getattr(data, name).copy() for name in STATE_FIELDS}
    sample.update(reference_pos=model.body("hammer_Object").pos.copy(),
                  reference_quat=model.body("hammer_Object").quat.copy(),
                  hammer_pos=data.body("hammer_Object").xpos.copy(),
                  hammer_quat=data.body("hammer_Object").xquat.copy(),
                  target_pos=data.site("cylinder_endface_force_frame").xpos.copy(),
                  target_quat=data.body("cylinder_sensing_endcap").xquat.copy(),
                  support_pos=data.body("support_L_support").xpos.copy(),
                  support_quat=data.body("support_L_support").xquat.copy(),
                  placement=placement, difficulty=difficulty)
    camera = CameraRGBPointCloud(model)
    try:
        frame = camera.capture(data)
        depth = camera.renderer.render()[::8, ::8].copy()
        depth[(depth <= 0) | (depth >= camera.far)] = 0
        sample.update(depth=depth[None].astype(np.float32),
                      camera_pos=frame["camera_pos"],
                      camera_quat=Rotation.from_matrix(frame["camera_rot"]).as_quat()[[3, 0, 1, 2]],
                      intrinsics=np.array([camera.focal / 8, camera.focal / 8,
                                           (camera.width - 1) / 16, (camera.height - 1) / 16]))
        grasps, _ = generate_grasps(model, data, frame, workspace_bounds=WORKSPACE,
                                   convert_first=False)
    finally:
        camera.close()
    count = min(candidates_per_scene, len(grasps["scores"]))
    object_rot = Rotation.from_quat(sample["hammer_quat"][[1, 2, 3, 0]])
    candidates = []
    for i in range(count):
        pose = candidate_to_mount(grasps, i)
        rotation = Rotation.from_quat(pose["mount_quat"][[1, 2, 3, 0]])
        candidates.append(dict(
            local_pos=object_rot.inv().apply(pose["mount_pos"] - sample["hammer_pos"]),
            local_quat=(object_rot.inv() * rotation).as_quat()[[3, 0, 1, 2]],
            open_ctrl=pose["gripper_open_ctrl"],
            **{name: grasps[name][i] for name in grasps}))
    del simulation
    return sample, candidates


def _prepare_scene_task(index, seed, candidates_per_scene, hammer_head_extra_mass):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    return prepare_scene(rng, index % 4, min(candidates_per_scene, 32),
                         hammer_head_extra_mass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="weight/hammer_rl/reset_bank.npz")
    parser.add_argument("--num-scenes", type=int, default=256)
    parser.add_argument("--candidates-per-scene", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="场景生成进程数（默认串行）")
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    args = parser.parse_args()
    task = partial(_prepare_scene_task, seed=args.seed,
                   candidates_per_scene=args.candidates_per_scene,
                   hammer_head_extra_mass=args.hammer_head_extra_mass)
    scenes, candidates, offsets = [], [], [0]
    with (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
          if args.workers > 1 else nullcontext()) as executor:
        results = (executor.map(task, range(args.num_scenes)) if executor is not None
                   else map(task, range(args.num_scenes)))
        for index, (sample, seeds) in enumerate(results):
            scenes.append(sample)
            candidates.extend(seeds)
            offsets.append(len(candidates))
            print(f"scene {index}: {PLACEMENTS[index % 4]}, candidates={len(seeds)}", flush=True)
    arrays = {key: np.stack([sample[key] for sample in scenes]) for key in scenes[0]}
    arrays.update({key: np.stack([sample[key] for sample in candidates]) for key in candidates[0]})
    # 各摆放按场景分割；候选共享场景标签，绝不跨 train/eval。
    split = np.zeros(args.num_scenes, dtype=np.int64)
    for placement in range(4):
        indices = np.flatnonzero(arrays["placement"] == placement)
        split[indices[-max(1, round(len(indices) * 0.2)):]] = 1
    arrays.update(offsets=np.array(offsets), split=split,
                  hammer_head_extra_mass=np.array(args.hammer_head_extra_mass))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"saved {output.resolve()}: {len(scenes)} scenes, {len(candidates)} candidates")


if __name__ == "__main__":
    main()
