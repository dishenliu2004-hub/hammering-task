"""离线混合四类历史姿态与连续随机姿态，缓存稳定 A 初态及 AnyGrasp 候选。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from generation.enclosure import build_enclosure_geometry
from generation.grasp_hammer import generate_grasps, candidate_to_mount
from generation.hammer_generate import create_simulation, sync_data, sample_legacy_placement, SCENE
from generation.hammer_inertia import add_head_mass

STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active",
                "qfrc_applied", "xfrc_applied", "qacc_warmstart")
# 固定标定工作区：支架及其上方；不依赖本次锤子真值位姿。
WORKSPACE = ((0.14, -0.16, 0.11), (0.46, 0.41, 0.72))


def create_scene(rng, hammer_head_extra_mass=0.0, placement=4):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(spec, hammer_head_extra_mass)
    handle = spec.geom("hammer_handle")
    handle.size[0] *= rng.uniform(0.95, 1.05)
    handle.size[1] *= rng.uniform(0.95, 1.05)
    model = spec.compile()
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.geom("hammer_head").condim[0] = 6
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hammer = model.body("hammer_Object")
    hammer_geoms = tuple(model.geom(name) for name in ("hammer_handle", "hammer_neck", "hammer_head"))
    support_geoms = tuple(model.geom(name) for name in
                          ("support_L_support_short_leg", "support_L_support_long_leg"))
    floor, wall = support_geoms
    wall_min, wall_max = wall.pos - wall.size, wall.pos + wall.size
    while True:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        if placement < 4:
            position, local_quat = sample_legacy_placement(rng, placement)
        else:
            # 连续采样支架局部位姿；Rz(yaw) @ Ry(-tilt) @ Rx(roll)，局部 +X 为柄轴。
            position = np.array([rng.uniform(0.12, 0.42), rng.uniform(-0.10, 0.10), 0.0])
            yaw, tilt, roll = (rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi / 2, np.pi / 2),
                               rng.uniform(-np.pi, np.pi))
            rotation = Rotation.from_euler("ZYX", [yaw, -tilt, roll])
            local_quat = rotation.as_quat()[[3, 0, 1, 2]]
            # 用编译模型的 capsule 端点/半径与 mesh 顶点求投影，预留 1 mm 接触间隙。
            # 挡板投影重叠时提高初始高度；不裁剪 x/y，也不要求锤体完全落在底板内。
            for geom in hammer_geoms:
                geom_rotation = Rotation.from_quat(geom.quat[[1, 2, 3, 0]])
                if geom.type[0] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    endpoints = geom.pos + geom_rotation.apply(
                        np.array([[0.0, 0.0, -geom.size[1]], [0.0, 0.0, geom.size[1]]]))
                    points = rotation.apply(endpoints)
                    lower, upper = points.min(axis=0) - geom.size[0], points.max(axis=0) + geom.size[0]
                else:
                    mesh = geom.dataid[0]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    points = rotation.apply(geom.pos + geom_rotation.apply(model.mesh_vert[start:start + count]))
                    lower, upper = points.min(axis=0), points.max(axis=0)
                position[2] = max(position[2], floor.pos[2] + floor.size[2] - lower[2] + 0.001)
                if np.all(position[:2] + upper[:2] > wall_min[:2]) and np.all(
                    position[:2] + lower[:2] < wall_max[:2]
                ):
                    position[2] = max(position[2], wall_max[2] - lower[2] + 0.001)
        support = data.body("support_L_support")
        # 参考位姿仍由支架局部转换到世界；沉降后的实际位姿保存在 data。
        hammer.pos[:] = support.xpos + support.xmat.reshape(3, 3) @ position
        mujoco.mju_mulQuat(hammer.quat, support.xquat, local_quat)
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
        simulation = create_simulation(model, data)
        for _ in range(round(2.0 / model.opt.timestep)):
            simulation.step()
        sync_data(simulation, data)
        settled_pos = data.body("hammer_Object").xpos.copy()
        settled_rotation = Rotation.from_quat(data.body("hammer_Object").xquat[[1, 2, 3, 0]])
        for _ in range(round(0.5 / model.opt.timestep)):
            simulation.step()
        sync_data(simulation, data)
        # 根据观察窗口末态的真实接触接纳支架上的稳定结果。
        mujoco.mj_forward(model, data)
        displacement = np.linalg.norm(data.body("hammer_Object").xpos - settled_pos)
        final_rotation = Rotation.from_quat(data.body("hammer_Object").xquat[[1, 2, 3, 0]])
        angle = (settled_rotation.inv() * final_rotation).magnitude()
        support_contact = any(
            contact.efc_address >= 0 and (
                (model.geom_bodyid[contact.geom1] == hammer.id
                 and contact.geom2 in (floor.id, wall.id))
                or (model.geom_bodyid[contact.geom2] == hammer.id
                    and contact.geom1 in (floor.id, wall.id)))
            for contact in data.contact[:data.ncon]
        )
        if displacement < 0.005 and angle < np.deg2rad(3) and support_contact:
            simulation.data.time[:] = 0
            sync_data(simulation, data)
            return model, data, simulation
        del simulation


def prepare_scene(rng, candidates_per_scene, hammer_head_extra_mass=0.0, placement=4):
    from visualization.camera_rgb_pointcloud import CameraRGBPointCloud

    while True:
        model, data, simulation = create_scene(rng, hammer_head_extra_mass, placement)
        camera = CameraRGBPointCloud(model)
        camera.renderer.update_scene(data, camera=camera.camera)
        camera.renderer.enable_segmentation_rendering()
        segmentation = camera.renderer.render()[::4, ::4]
        camera.renderer.disable_segmentation_rendering()
        # 与相机点云同分辨率：仅接纳锤柄存在可见像素的场景。
        if np.any((segmentation[..., 0] == model.geom("hammer_handle").id)
                  & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)):
            break
        camera.close()
        del simulation
    geometry = build_enclosure_geometry(model, "cpu")
    sample = {name: getattr(data, name).copy() for name in STATE_FIELDS}
    sample.update(handle_size=model.geom("hammer_handle").size[:2].copy(),
                  handle_convex_planes=geometry.convex_parts[geometry.handle_part_index].numpy(),
                  reference_pos=model.body("hammer_Object").pos.copy(),
                  reference_quat=model.body("hammer_Object").quat.copy(),
                  hammer_pos=data.body("hammer_Object").xpos.copy(),
                  hammer_quat=data.body("hammer_Object").xquat.copy(),
                  target_pos=data.site("cylinder_endface_force_frame").xpos.copy(),
                  target_quat=data.body("cylinder_sensing_endcap").xquat.copy(),
                  support_pos=data.body("support_L_support").xpos.copy(),
                  support_quat=data.body("support_L_support").xquat.copy())
    sample["source_type"] = np.int64(placement)
    captured, candidates = capture_candidates(model, data, camera, sample, candidates_per_scene, WORKSPACE)
    sample.update(captured)
    del simulation
    return sample, candidates


def capture_candidates(model, data, camera, sample, count, workspace):
    """A/B 共用原相机、SDK、NMS、覆盖采样和一次 30 mm 转换。"""
    try:
        frame = camera.capture(data)
        depth = camera.renderer.render()[::8, ::8].copy()
        depth[(depth <= 0) | (depth >= camera.far)] = 0
        captured = dict(depth=depth[None].astype(np.float32),
                        camera_pos=frame["camera_pos"],
                        camera_quat=Rotation.from_matrix(frame["camera_rot"]).as_quat()[[3, 0, 1, 2]],
                        intrinsics=np.array([camera.focal / 8, camera.focal / 8,
                                             (camera.width - 1) / 16, (camera.height - 1) / 16]))
        grasps, _ = generate_grasps(model, data, frame, full_scene=False,
                                   workspace_bounds=workspace, convert_first=False)
    finally:
        camera.close()
    return captured, _prepare_candidates(grasps, sample, min(count, 32))


def _select_grasps(grasps, count):
    """从最高分候选开始，在已有 NMS 结果中按平移和旋转距离覆盖采样。"""
    order = np.argsort(-grasps["scores"], kind="stable")
    positions = grasps["translations"][order]
    rotations = Rotation.from_matrix(grasps["rotations"][order])
    distances = np.full(len(order), np.inf)
    selected = []
    for _ in range(min(count, len(order))):
        i = distances.argmax()
        selected.append(order[i])
        distance = (np.sum((positions - positions[i]) ** 2, axis=1) / 0.03 ** 2
                    + (rotations[i].inv() * rotations).magnitude() ** 2 / (np.pi / 6) ** 2)
        distances = np.minimum(distances, distance)
        distances[i] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def _prepare_candidates(grasps, sample, candidates_per_scene):
    object_rot = Rotation.from_quat(sample["hammer_quat"][[1, 2, 3, 0]])
    candidates = []
    for i in _select_grasps(grasps, candidates_per_scene):
        pose = candidate_to_mount(grasps, i, extra_depth=0.03)
        rotation = Rotation.from_quat(pose["mount_quat"][[1, 2, 3, 0]])
        candidates.append(dict(
            local_pos=object_rot.inv().apply(pose["mount_pos"] - sample["hammer_pos"]),
            local_quat=(object_rot.inv() * rotation).as_quat()[[3, 0, 1, 2]],
            open_ctrl=pose["gripper_open_ctrl"],
            mount_pos=pose["mount_pos"], mount_quat=pose["mount_quat"],
            **{name: grasps[name][i] for name in grasps}))
    return candidates


def _prepare_scene_task(index, seed, candidates_per_scene, hammer_head_extra_mass, placements):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    try:
        return prepare_scene(rng, min(candidates_per_scene, 32),
                             hammer_head_extra_mass, int(placements[index]))
    except Exception as error:
        raise RuntimeError(f"准备场景失败：scene={index}, seed={seed}\n{error}") from error


def stratified_split(sources, seed):
    rng = np.random.default_rng(seed)
    split = np.zeros(len(sources), dtype=np.int64)
    for source in np.unique(sources):
        indices = rng.permutation(np.flatnonzero(sources == source))
        split[indices[:round(len(indices) * 0.2)]] = 1
    return split


def bank_arrays(scenes, candidates, offsets):
    arrays = {key: np.stack([sample[key] for sample in scenes]) for key in scenes[0]}
    arrays.update({key: np.stack([sample[key] for sample in candidates]) for key in candidates[0]})
    arrays["offsets"] = np.asarray(offsets, dtype=np.int64)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="输出路径，默认按普通/测试模式选择")
    parser.add_argument("--test-set", action="store_true",
                        help="独立生成 128 个测试场景，忽略三个场景数量参数，不再划分")
    parser.add_argument("--num-scenes", type=int, default=256)
    parser.add_argument("--legacy-per-placement", type=int, default=45)
    parser.add_argument("--random-scenes", type=int, default=76)
    parser.add_argument("--candidates-per-scene", type=int, default=32)
    parser.add_argument("--seed", type=int, help="随机种子，普通模式默认 42，测试模式默认 43")
    parser.add_argument("--workers", type=int, default=1, help="场景生成进程数（默认串行）")
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 43 if args.test_set else 42
    if args.output is None:
        args.output = ("weight/hammer_rl/test_bank_ab_128.npz" if args.test_set
                       else "weight/hammer_rl/reset_bank_ab_256.npz")
    if args.test_set:
        args.num_scenes = 128
        placements = np.repeat(np.arange(5), [23, 23, 22, 22, 38])
    else:
        placements = np.repeat(np.arange(5), [args.legacy_per_placement] * 4 + [args.random_scenes])
    task = partial(_prepare_scene_task, seed=args.seed,
                   candidates_per_scene=args.candidates_per_scene,
                   hammer_head_extra_mass=args.hammer_head_extra_mass, placements=placements)
    scenes, candidates, offsets = [], [], [0]
    with (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
          if args.workers > 1 else nullcontext()) as executor:
        results = (executor.map(task, range(args.num_scenes)) if executor is not None
                   else map(task, range(args.num_scenes)))
        for index, (sample, seeds) in enumerate(results):
            scenes.append(sample)
            candidates.extend(seeds)
            offsets.append(len(candidates))
            print(f"scene {index}: candidates={len(seeds)}", flush=True)
    arrays = bank_arrays(scenes, candidates, offsets)
    arrays.update(split=(np.ones(args.num_scenes, dtype=np.int64) if args.test_set
                         else stratified_split(placements, args.seed)),
                  hammer_head_extra_mass=np.array(args.hammer_head_extra_mass))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"saved {output.resolve()}: {len(scenes)} scenes, {len(candidates)} candidates")


if __name__ == "__main__":
    main()
