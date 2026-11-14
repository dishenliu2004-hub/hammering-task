"""逐个恢复已有 A reset bank 并额外等待 30 秒，观察慢滑或倾倒。

N 下一场景，P 上一场景，空格暂停/继续，R 从缓存重新播放；--wait-time 0 查看静态缓存。

cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
MUJOCO_GL=glfw python -m visualization.hammer_reset_bank
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from hammer_rl.prepare import STATE_FIELDS
from generation.hammer_generate import sync_data
from hammer_rl.check_reset_bank import bank_simulation, restore_bank, rotation_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "weight/hammer_rl/reset_bank_ab_256.npz")
    parser.add_argument("--wait-time", type=float, default=30.0,
                        help="从缓存状态额外等待的仿真秒数，0 为静态浏览（默认 30）")
    parser.add_argument("--scene-index", type=int, default=0, help="从指定场景开始（零起始编号）")
    args = parser.parse_args()
    if not np.isfinite(args.wait_time) or args.wait_time < 0:
        parser.error("wait-time 必须是非负有限数")

    with np.load(args.data_file) as source:
        bank = {name: source[name] for name in (*STATE_FIELDS, "handle_size",
                "reference_pos", "reference_quat", "split", "hammer_head_extra_mass")}

    count = len(bank["reference_pos"])
    if not 0 <= args.scene_index < count:
        parser.error(f"scene-index 必须在 0 到 {count - 1} 之间")
    simulation = bank_simulation(bank, np.array([args.scene_index]))
    model = simulation.mj_model
    data = mujoco.MjData(model)
    hammer = model.body("hammer_Object")
    handle = model.geom("hammer_handle").id
    initial_pos = np.zeros(3)
    initial_quat = np.zeros(4)

    def restore_scene(index):
        radius, half_length = bank["handle_size"][index]
        model.geom_size[handle, :2] = radius, half_length
        model.geom_rbound[handle] = radius + half_length
        model.geom_aabb[handle].reshape(2, 3)[1] = radius, radius, radius + half_length
        hammer.pos[:] = bank["reference_pos"][index]
        hammer.quat[:] = bank["reference_quat"][index]
        mujoco.mj_setConst(model, data)
        for name in STATE_FIELDS:
            getattr(data, name)[:] = bank[name][index]
        data.time = 0
        mujoco.mj_forward(model, data)
        restore_bank(simulation, bank, np.array([index]))
        sync_data(simulation, data)
        initial_pos[:] = data.body("hammer_Object").xpos
        initial_quat[:] = data.body("hammer_Object").xquat

    index = args.scene_index
    pending_step = 0
    pending_reload = False
    paused = False

    def key_callback(key):
        nonlocal pending_step, pending_reload, paused
        if key == ord("N"):
            pending_step += 1
        elif key == ord("P"):
            pending_step -= 1
        elif key == ord("R"):
            pending_reload = True
        elif key == 32:
            paused = not paused

    restore_scene(index)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            support = data.body("support_L_support")
            a_lookat = support.xpos + support.xmat.reshape(3, 3) @ np.array(
                [0.22, 0.0, 0.20])
            viewer.cam.lookat[:] = a_lookat
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -35.0

        while viewer.is_running():
            started = time.monotonic()
            if pending_step or pending_reload:
                with viewer.lock():
                    index = (index + pending_step) % count
                    pending_step = 0
                    pending_reload = False
                    restore_scene(index)

            if not paused and data.time < args.wait_time - model.opt.timestep / 2:
                with viewer.lock():
                    remaining = round((args.wait_time - data.time) / model.opt.timestep)
                    for _ in range(min(round(0.02 / model.opt.timestep), remaining)):
                        simulation.step()
                    sync_data(simulation, data)

            split = ("train", "eval")[int(bank["split"][index])]
            displacement = np.linalg.norm(data.body("hammer_Object").xpos - initial_pos) * 1000
            angle = float(rotation_distance(data.body("hammer_Object").xquat, initial_quat))
            finished = data.time >= args.wait_time - model.opt.timestep / 2
            status = "paused" if paused else ("finished" if finished else "waiting")
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                              "Reset bank\nScene ID\nSplit\nExtra wait\nDrift\nStatus\nNext / Previous\nPause / Replay",
                              f"A: {count} scenes\n{index}\n{split}\n{data.time:.2f} / {args.wait_time:g} s"
                              f"\n{displacement:.2f} mm / {angle:.2f} deg\n{status}\nN / P\nSpace / R"))
            viewer.sync()
            time.sleep(max(0, 0.02 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
