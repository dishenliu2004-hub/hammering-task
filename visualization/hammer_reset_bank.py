"""逐个浏览 A reset bank 和 B 起点位姿，N 下一场景，P 上一场景。

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

from generation.hammer_inertia import add_head_mass
from generation.hammer_generate import set_hammer_pose
from hammer_rl.prepare import SCENE, STATE_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "weight/hammer_rl/reset_bank_ab_256.npz")
    parser.add_argument("--b-pose-file", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "weight/hammer_rl/b_pose_bank_40.npz")
    args = parser.parse_args()

    with np.load(args.data_file) as source:
        bank = {name: source[name] for name in (*STATE_FIELDS, "handle_size",
                "reference_pos", "reference_quat", "split", "hammer_head_extra_mass")}
    with np.load(args.b_pose_file) as source:
        b_bank = {name: source[name] for name in ("hammer_pos", "hammer_quat", "split")}

    spec = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(spec, float(bank["hammer_head_extra_mass"]))
    model = spec.compile()
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.geom("hammer_head").condim[0] = 6
    data = mujoco.MjData(model)
    hammer = model.body("hammer_Object")
    handle = model.geom("hammer_handle").id
    handle_size = model.geom_size[handle, :2].copy()
    a_count = len(bank["reference_pos"])
    b_count = len(b_bank["hammer_pos"])
    count = a_count + b_count

    def restore_scene(index):
        is_b = index >= a_count
        radius, half_length = handle_size if is_b else bank["handle_size"][index]
        model.geom_size[handle, :2] = radius, half_length
        model.geom_rbound[handle] = radius + half_length
        model.geom_aabb[handle].reshape(2, 3)[1] = radius, radius, radius + half_length
        if is_b:
            b_index = index - a_count
            mujoco.mj_resetData(model, data)
            set_hammer_pose(model, data, b_bank["hammer_pos"][b_index],
                            b_bank["hammer_quat"][b_index])
            return
        hammer.pos[:] = bank["reference_pos"][index]
        hammer.quat[:] = bank["reference_quat"][index]
        mujoco.mj_setConst(model, data)
        for name in STATE_FIELDS:
            getattr(data, name)[:] = bank[name][index]
        data.time = 0
        mujoco.mj_forward(model, data)

    index = 0
    pending_step = 0

    def key_callback(key):
        nonlocal pending_step
        if key == ord("N"):
            pending_step += 1
        elif key == ord("P"):
            pending_step -= 1

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
            if pending_step:
                with viewer.lock():
                    index = (index + pending_step) % count
                    pending_step = 0
                    restore_scene(index)
                    viewer.cam.lookat[:] = (data.body("hammer_Object").xpos
                                            if index >= a_count else a_lookat)

            is_b = index >= a_count
            scene_index = index - a_count if is_b else index
            split = ("train", "eval")[int((b_bank if is_b else bank)["split"][scene_index])]
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                              "Reset bank\nScene ID\nSplit\nNext / Previous",
                              f"{'B' if is_b else 'A'}: {b_count if is_b else a_count} scenes"
                              f"\n{scene_index}\n{split}\nN / P"))
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
