"""逐个浏览连续随机生成的稳定锤子位姿，按 N 生成下一个。

cd /home/ubuntu20/CodeMujoco/hammering-task
conda activate hammer
MUJOCO_GL=glfw python -m visualization.hammer_poses --seed 42
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from hammer_rl.prepare import STATE_FIELDS, create_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    args = parser.parse_args()

    def generate_scene(index):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        model, data, simulation = create_scene(rng, args.hammer_head_extra_mass)
        del simulation
        return model, data

    index = 0
    next_requested = False

    def key_callback(key):
        nonlocal next_requested
        if key == ord("N"):
            next_requested = True

    model, data = generate_scene(index)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            support = data.body("support_L_support")
            viewer.cam.lookat[:] = support.xpos + support.xmat.reshape(3, 3) @ np.array(
                [0.22, 0.0, 0.20])
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -35.0

        while viewer.is_running():
            if next_requested:
                next_requested = False
                index += 1
                next_model, next_data = generate_scene(index)
                with viewer.lock():
                    hammer = model.body("hammer_Object")
                    hammer.pos[:] = next_model.body("hammer_Object").pos
                    hammer.quat[:] = next_model.body("hammer_Object").quat
                    mujoco.mj_setConst(model, data)
                    for name in STATE_FIELDS:
                        getattr(data, name)[:] = getattr(next_data, name)
                    data.time = next_data.time
                    mujoco.mj_forward(model, data)
                del next_model, next_data

            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                              "Continuous stable hammer poses\nSample\nNext pose",
                              f"Seed {args.seed}\n{index}\nN"))
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
