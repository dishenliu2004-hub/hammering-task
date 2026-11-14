"""
加载训练权重，在原生 MuJoCo 窗口中按网格并行演示独立锤击环境。

python -m visualization.parallel_hammering \
  --checkpoint weight/hammer_rl/runs/teacher/formal_01/model_2350.pt \
  --num-envs 64
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.viewer.native import NativeMujocoViewer

from hammer_rl.env_cfg import make_env_cfg
from hammer_rl.rl_cfg import ppo_cfg


class ParallelHammeringViewer(NativeMujocoViewer):

    def __init__(self, env, policy, spacing):
        super().__init__(env, policy, frame_rate=50.0, enable_perturbations=False)
        self.increase_speed()
        self.increase_speed()
        self.increase_speed()
        model = env.unwrapped.sim.mj_model
        model.tex("gripper_skybox").data[:] = 255
        model.vis.headlight.ambient[:] = 0.5
        model.vis.headlight.diffuse[:] = 0.8
        columns = math.ceil(math.sqrt(env.num_envs))
        self.offsets = np.array([(i % columns * spacing, i // columns * spacing, 0.0)
                                 for i in range(env.num_envs)])
        extent = self.offsets.max(axis=0)
        self.cfg.origin_type = self.cfg.OriginType.WORLD
        self.cfg.lookat = tuple(extent / 2 + (0.0, 0.0, -0.1))
        self.cfg.distance = 1.8 * (max(extent) + 2.0)
        self.cfg.elevation = -35.0
        self.cfg.azimuth = 135.0
        env.unwrapped.action_manager.get_term("grasp").on_physics_step = self.display

    def _step_physics(self, dt):
        # 一次环境步包含完整尝试，显示节奏由物理子步回调控制。
        self._execute_step()

    def display(self, term):
        if term._substep % term.substeps == 0 and self.is_running():
            start = time.monotonic()
            self.sync_env_to_viewer()
            time.sleep(max(0.0, self.frame_time / self.target_realtime - (time.monotonic() - start)))

    def _render_other_env_geoms(self, viewer, sim, sim_data):
        for i in range(self.env.num_envs):
            if i == self.env_idx:
                continue
            self._sync_env_state_to_mjdata(self.vd, sim_data, i)
            self._sync_model_fields(sim, i)
            mujoco.mj_forward(self.mjm, self.vd)
            first = viewer.user_scn.ngeom
            mujoco.mjv_addGeoms(self.mjm, self.vd, self.vopt, viewer.perturb,
                               mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
            # 平移完整场景（含静态桌子）的渲染几何，不回写物理状态。
            offset = self.offsets[i] - self.offsets[self.env_idx]
            for geom in viewer.user_scn.geoms[first:viewer.user_scn.ngeom]:
                geom.pos[:] += offset
        self._sync_model_fields(sim, self.env_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--spacing", type=float, default=1.4)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    saved = json.loads((args.checkpoint.parent / "run_config.json").read_text())
    stage = saved["stage"]
    cfg = make_env_cfg(saved["hammer_head_extra_mass"],
        data_file=args.data_file or saved["data_file"], stage=stage,
        num_envs=args.num_envs, split=args.split, seed=args.seed, training=False)
    cfg.randomize_physics = False
    cfg.events.pop("physics", None)
    cfg.visual_noise = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        wrapper = RslRlVecEnvWrapper(env)
        runner = MjlabOnPolicyRunner(wrapper,
            asdict(ppo_cfg("teacher" if stage == "teacher" else "student")),
            log_dir=None, device=args.device)
        loaded = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        actor = loaded["student_state_dict"] if stage == "distill" else loaded["actor_state_dict"]
        runner.alg.load({"actor_state_dict": actor}, {"actor": True}, strict=True)
        viewer = ParallelHammeringViewer(wrapper, runner.get_inference_policy(), args.spacing)
        with torch.inference_mode():
            viewer.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
