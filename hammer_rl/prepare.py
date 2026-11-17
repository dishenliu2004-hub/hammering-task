"""离线按教师 v1 接触几何采样稳定位姿，缓存场景初态、视觉观测及 AnyGrasp 候选。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
import json
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from scipy.spatial.transform import Rotation

from generation.enclosure import build_enclosure_geometry
from generation.grasp_hammer import generate_grasps, candidate_to_mount
from generation.hammer_generate import (create_simulation, sync_data, sample_legacy_placement,
    support_contacts, SCENE, STABILITY_DURATION, STABILITY_INTERVAL, STABILITY_POSITION,
    STABILITY_ROTATION_DEG)
from generation.hammer_inertia import add_head_mass

STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active",
                "qfrc_applied", "xfrc_applied", "qacc_warmstart")
MODEL_FIELDS = ("body_pos", "body_quat", "geom_size", "geom_rbound", "geom_aabb",
                "dof_invweight0", "body_invweight0", "tendon_length0",
                "tendon_invweight0", "actuator_acc0")
# 固定标定工作区：支架及其上方；不依赖本次锤子真值位姿。
WORKSPACE = ((0.14, -0.16, 0.11), (0.46, 0.41, 0.72))


def restore_bank(sim, bank, indices):
    """与 prepare/reset_scene 相同的模型参数与完整缓存状态，不绑定抓取候选。"""
    model = sim.mj_model
    h, body = model.geom("hammer_handle").id, model.body("hammer_Object").id

    def tensor(values, target):
        return torch.as_tensor(values, dtype=target.dtype, device=sim.device)

    sizes = tensor(bank["handle_size"][indices], sim.model.geom_size)
    sim.model.geom_size[:, h, :2] = sizes
    sim.model.geom_rbound[:, h] = sizes.sum(dim=1)
    sim.model.geom_aabb[:, h, 1, :2] = sizes[:, :1]
    sim.model.geom_aabb[:, h, 1, 2] = sizes.sum(dim=1)
    sim.model.body_pos[:, body] = tensor(bank["reference_pos"][indices], sim.model.body_pos)
    sim.model.body_quat[:, body] = tensor(bank["reference_quat"][indices], sim.model.body_quat)
    sim.recompute_constants(RecomputeLevel.set_const_0)
    # set_const 可能改写 data，必须在其后恢复包括 warmstart 在内的全部状态。
    sim.reset()
    for name in STATE_FIELDS:
        target = getattr(sim.data, name)
        target[:] = tensor(bank[name][indices], target)
    sim.data.time[:] = 0
    sim.forward()


def bank_simulation(bank, indices, device="cuda:0"):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(spec, float(bank["hammer_head_extra_mass"]))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 0
    spec.geom("hammer_head").condim = 6
    model = spec.compile()
    opt = model.opt
    # 与 generation.hammer_generate.create_simulation 使用相同的物理设置。
    cfg = SimulationCfg(nconmax=128, njmax=1024, mujoco=MujocoCfg(
        timestep=opt.timestep,
        integrator=mujoco.mjtIntegrator(opt.integrator).name.removeprefix("mjINT_").lower(),
        cone="elliptic",
        jacobian=mujoco.mjtJacobian(opt.jacobian).name.removeprefix("mjJAC_").lower(),
        solver=mujoco.mjtSolver(opt.solver).name.removeprefix("mjSOL_").lower(),
        impratio=opt.impratio, iterations=opt.iterations, tolerance=opt.tolerance,
        ls_iterations=opt.ls_iterations, ls_tolerance=opt.ls_tolerance,
        ccd_iterations=opt.ccd_iterations, gravity=tuple(opt.gravity),
        disableflags=("multiccd",)))
    sim = Simulation(len(indices), cfg, model=model, device=device)
    sim.expand_model_fields(MODEL_FIELDS)
    restore_bank(sim, bank, indices)
    return sim


def _stable_window(simulation, position, quaternion):
    """整个观察窗口相对同一初态检查位姿和 GPU 有效支架接触。"""
    hammer = simulation.mj_model.body("hammer_Object").id
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    stride = round(STABILITY_INTERVAL / simulation.mj_model.opt.timestep)
    for sample_index in range(round(STABILITY_DURATION / STABILITY_INTERVAL) + 1):
        if sample_index:
            for _ in range(stride):
                simulation.step()
        simulation.forward()
        current_pos = simulation.data.xpos[0, hammer].cpu().numpy()
        current_quat = simulation.data.xquat[0, hammer].cpu().numpy()
        displacement = np.linalg.norm(current_pos - position)
        current_rotation = Rotation.from_quat(current_quat[[1, 2, 3, 0]])
        angle = (rotation.inv() * current_rotation).magnitude()
        if (displacement > STABILITY_POSITION or angle > np.deg2rad(STABILITY_ROTATION_DEG)
                or not support_contacts(simulation)[0]):
            return False
    return True


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
    scene = placement if placement < 4 else int(rng.choice((0, 2, 3)))
    while True:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        position, local_quat = sample_legacy_placement(
            rng, scene, model=model, expanded=placement == 4)
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
        if _stable_window(simulation, data.body("hammer_Object").xpos.copy(),
                          data.body("hammer_Object").xquat.copy()):
            simulation.data.time[:] = 0
            sync_data(simulation, data)
            return model, data, simulation
        del simulation


def prepare_scene(rng, candidates_per_scene, hammer_head_extra_mass=0.0, placement=4):
    from visualization.camera_rgb_pointcloud import CameraRGBPointCloud

    while True:
        model, data, simulation = create_scene(rng, hammer_head_extra_mass, placement)
        # S 是最终准备保存的完整状态；复核只推进重建后的单环境仿真。
        bank = {name: getattr(data, name).copy()[None] for name in STATE_FIELDS}
        bank.update(handle_size=model.geom("hammer_handle").size[:2].copy()[None],
                    reference_pos=model.body("hammer_Object").pos.copy()[None],
                    reference_quat=model.body("hammer_Object").quat.copy()[None],
                    hammer_pos=data.body("hammer_Object").xpos.copy()[None],
                    hammer_quat=data.body("hammer_Object").xquat.copy()[None],
                    hammer_head_extra_mass=np.asarray(hammer_head_extra_mass))
        device = simulation.device
        del simulation
        indices = np.array([0])
        simulation = bank_simulation(bank, indices, device=device)
        if not _stable_window(simulation, bank["hammer_pos"][0], bank["hammer_quat"][0]):
            del simulation
            continue
        # 回到同一个 S 后才采图；不保存复核推进后的末态。
        restore_bank(simulation, bank, indices)
        sync_data(simulation, data)
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
    captured, frame = capture_observation(data, camera)
    sample.update(captured)
    del simulation
    grasps, _ = generate_grasps(model, data, frame, full_scene=False,
                               workspace_bounds=WORKSPACE, convert_first=False)
    return sample, _prepare_candidates(grasps, sample, min(candidates_per_scene, 32))


def capture_observation(data, camera, *, rgb_path=None):
    """沿用原相机的采集顺序、深度分辨率及相机参数。"""
    frame = camera.capture(data)
    if rgb_path is not None:
        from PIL import Image
        Image.fromarray(frame["rgb"]).save(rgb_path)
    depth = camera.renderer.render()[::8, ::8].copy()
    depth[(depth <= 0) | (depth >= camera.far)] = 0
    captured = dict(depth=depth[None].astype(np.float32),
                    camera_pos=frame["camera_pos"],
                    camera_quat=Rotation.from_matrix(frame["camera_rot"]).as_quat()[[3, 0, 1, 2]],
                    intrinsics=np.array([camera.focal / 8, camera.focal / 8,
                                         (camera.width - 1) / 16, (camera.height - 1) / 16]))
    camera.close()
    return captured, frame


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


def export_scene_context(data_file, scene_index, output):
    """从既有 bank 恢复一个初态，导出原分辨率 RGB、现有深度和物理摘要。"""
    from visualization.camera_rgb_pointcloud import CameraRGBPointCloud

    with np.load(data_file) as bank:
        spec = mujoco.MjSpec.from_file(str(SCENE))
        add_head_mass(spec, float(bank["hammer_head_extra_mass"]))
        spec.geom("hammer_handle").size[:2] = bank["handle_size"][scene_index]
        spec.body("hammer_Object").pos = bank["reference_pos"][scene_index]
        spec.body("hammer_Object").quat = bank["reference_quat"][scene_index]
        model = spec.compile()
        data = mujoco.MjData(model)
        for name in STATE_FIELDS:
            getattr(data, name)[:] = bank[name][scene_index]
    mujoco.mj_forward(model, data)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = output.with_suffix(".rgb.png")
    depth_path = output.with_suffix(".depth.npy")
    camera = CameraRGBPointCloud(model)
    observation, _ = capture_observation(data, camera, rgb_path=rgb_path)
    depth = observation.pop("depth")[0]
    np.save(depth_path, depth)
    context = dict(task_id="hammer", profile_id="default", scene_id=scene_index,
        data_file=str(Path(data_file).resolve()), rgb=rgb_path.name, depth=depth_path.name,
        depth_unit="m", depth_invalid=0, width=depth.shape[1], height=depth.shape[0],
        rgb_width=camera.width, rgb_height=camera.height,
        rgb_intrinsics=(observation["intrinsics"] * 8).tolist(),
        camera_frame="optical: x right, y down, z forward",
        **{key: value.tolist() for key, value in observation.items()},
        physics=dict(handle_size=model.geom("hammer_handle").size[:2].tolist(),
            tool_mass_kg=float(model.body("hammer_Object").mass[0]),
            tool_inertia_kg_m2=model.body("hammer_Object").inertia.tolist(),
            handle_friction=model.geom("hammer_handle").friction.tolist(),
            tool_pos=data.body("hammer_Object").xpos.tolist(),
            tool_quat_wxyz=data.body("hammer_Object").xquat.tolist(),
            target_pos=data.site("cylinder_endface_force_frame").xpos.tolist(),
            trajectory="single 6D grasp, lift, fixed three-strike trajectory, unload"))
    output.write_text(json.dumps(context, indent=2) + "\n")
    print(f"scene context: {output.resolve()}")


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
    parser.add_argument("--scene-context", type=Path, help="仅从既有 bank 导出离线场景上下文")
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--scene-index", type=int, default=0)
    args = parser.parse_args()
    if args.scene_context is not None:
        export_scene_context(args.data_file, args.scene_index, args.scene_context)
        return
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
