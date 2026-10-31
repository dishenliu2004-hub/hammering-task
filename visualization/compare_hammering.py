"""同一放置类型的三组 AnyGrasp / 训练策略同步锤击对照；回车重播。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from hammer_rl.env_cfg import make_env_cfg
from hammer_rl.rl_cfg import ppo_cfg
from visualization.parallel_hammering import ParallelHammeringViewer

PLACEMENTS = ("flat", "lean_head_down", "lean_head_up", "inverted")


class ComparisonHammeringViewer(ParallelHammeringViewer):
    def __init__(self, env, policy, spacing, placement, checkpoint):
        super().__init__(env, policy, spacing)
        self.placement = placement
        self.checkpoint = checkpoint
        self.scene_ids = env.unwrapped.hammer.scene_id.cpu().tolist()
        self.offsets[3:, 1] = -spacing
        self.cfg.azimuth = 90.0
        self.cfg.elevation = -50.0
        self.cfg.lookat = (0.65 * spacing, -spacing / 2, -0.1)
        self.cfg.distance *= 0.8
        self._show_debug_vis = False

    def _execute_step(self):
        # 每次播放从同一组初态开始；结束后保留终态和指标，不自动 reset。
        self.env.reset()
        result = super()._execute_step()
        self.pause()
        return result

    def reset_environment(self):
        super().reset_environment()
        self.resume()

    def _set_status_overlay(self, viewer):
        term = self.env.unwrapped.action_manager.get_term("grasp")
        reward = sum(term.values[name] * cfg.weight for name, cfg in self.env.cfg.rewards.items())
        values = torch.stack((term.values["peak_force"], reward), dim=-1).cpu().tolist()
        names = [f"{'Trained' if i < 3 else 'AnyGrasp'} {i % 3 + 1} / S{scene}  "
                 for i, scene in enumerate(self.scene_ids)]
        numbers = [f"{force:.3f} N | {total:.4f}" for force, total in values]
        phase = "Final" if term._substep == self.env.cfg.decimation else "Running"
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "\n".join([self.placement, "Environment", *names, "Enter: replay"]),
            "\n".join([self.checkpoint, "Peak force | Total reward", *numbers, phase])))

    def _add_env_selection_marker(self, viewer):
        # 标签与网格使用相同平移，切换所选环境时仍对应各自场景。
        for i, scene in enumerate(self.scene_ids):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            pos = self.offsets[i] - self.offsets[self.env_idx] + (0.0, 0.0, 0.9)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3), pos, np.eye(3).ravel(), np.ones(4))
            geom.label = f"{'T' if i < 3 else 'A'}{i % 3 + 1} / S{scene}"
            viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--placement", choices=PLACEMENTS, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacing", type=float, default=1.4)
    args = parser.parse_args()

    saved = json.loads((args.checkpoint.parent / "run_config.json").read_text())
    stage = saved["stage"]
    data_file = args.data_file or saved["data_file"]
    with np.load(data_file) as bank:
        scenes = np.flatnonzero((bank["split"] == (args.split == "eval"))
                               & (bank["placement"] == PLACEMENTS.index(args.placement)))
        scenes = np.random.default_rng(args.seed).choice(scenes, size=3, replace=False)
        candidates = bank["offsets"][scenes]
    cfg = make_env_cfg(saved["hammer_head_extra_mass"], data_file=data_file, stage=stage,
        num_envs=6, split=args.split, seed=args.seed, training=False)
    cfg.randomize_physics = False
    cfg.events.pop("physics", None)
    cfg.visual_noise = False
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env.hammer.selection = tuple(torch.as_tensor(np.tile(values, 2), device=env.device)
                                     for values in (scenes, candidates))
        wrapper = RslRlVecEnvWrapper(env)
        runner = MjlabOnPolicyRunner(wrapper,
            asdict(ppo_cfg("teacher" if stage == "teacher" else "student")),
            log_dir=None, device=args.device)
        loaded = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        actor = loaded["student_state_dict"] if stage == "distill" else loaded["actor_state_dict"]
        runner.alg.load({"actor_state_dict": actor}, {"actor": True}, strict=True)
        policy = runner.get_inference_policy()

        def paired_policy(observations):
            action = policy(observations)
            action[3:] = 0
            return action

        viewer = ComparisonHammeringViewer(wrapper, paired_policy, args.spacing,
                                           args.placement, args.checkpoint.name)
        with torch.inference_mode():
            viewer.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
