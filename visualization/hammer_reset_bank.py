"""逐个浏览 reset bank 中缓存的锤子摆放，N 下一场景，P 上一场景。

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
from hammer_rl.prepare import SCENE, STATE_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "weight/hammer_rl/reset_bank_dr_256.npz")
    args = parser.parse_args()

    with np.load(args.data_file) as source:
        bank = {name: source[name] for name in (*STATE_FIELDS, "handle_size",
                "reference_pos", "reference_quat", "split", "hammer_head_extra_mass")}

    spec = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(spec, float(bank["hammer_head_extra_mass"]))
    model = spec.compile()
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.geom("hammer_head").condim[0] = 6
    data = mujoco.MjData(model)
    hammer = model.body("hammer_Object")
    handle = model.geom("hammer_handle").id
    count = len(bank["reference_pos"])

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
            viewer.cam.lookat[:] = support.xpos + support.xmat.reshape(3, 3) @ np.array(
                [0.22, 0.0, 0.20])
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -35.0

        while viewer.is_running():
            if pending_step:
                with viewer.lock():
                    index = (index + pending_step) % count
                    pending_step = 0
                    restore_scene(index)

            split = ("train", "eval")[int(bank["split"][index])]
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                              "Reset bank\nScene ID\nSplit\nNext / Previous",
                              f"{count} scenes\n{index}\n{split}\nN / P"))
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
