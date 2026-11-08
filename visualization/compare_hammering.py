"""每轮三个缓存场景成对展示：学习选择＋残差，与共享候选集的 AnyGrasp 零残差。"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from hammer_rl.env_cfg import make_env_cfg
from hammer_rl.inference import load_models, select_candidates
from visualization.parallel_hammering import ParallelHammeringViewer

class ComparisonHammeringViewer(ParallelHammeringViewer):
    def __init__(self, env, actor, critic, spacing, checkpoint):
        self.actor, self.critic = actor, critic
        self.action = torch.zeros((env.num_envs, 6), device=env.device)
        super().__init__(env, lambda observations: self.action, spacing)
        self.checkpoint = checkpoint
        self.offsets[3:, 1] = -spacing
        self.cfg.azimuth = 90.0
        self.cfg.elevation = -50.0
        self.cfg.lookat = (0.65 * spacing, -spacing / 2, -0.1)
        self.cfg.distance *= 0.8
        self._show_debug_vis = False

    def _prepare_round(self):
        bank = self.env.unwrapped.hammer.bank
        scenes = (bank["split"] == (self.env.cfg.split == "eval")).nonzero().flatten()
        scene = scenes[torch.randint(len(scenes), (3,), device=self.env.device)]
        sdk = torch.stack([start + bank["scores"][start:end].argmax()
                           for start, end in zip(bank["offsets"][scene], bank["offsets"][scene + 1])])
        self.env.unwrapped.hammer.selection = (scene.repeat(2), sdk.repeat(2))
        self.env.reset()
        result = select_candidates(self.env.unwrapped, self.actor, self.critic,
                                   torch.arange(3, device=self.env.device))
        self.action.zero_()
        self.action[:3] = result["action"]

    def _execute_step(self):
        self._prepare_round()
        result = super()._execute_step()
        self.pause()
        return result

    def reset_environment(self):
        super().reset_environment()
        self.resume()

    def _set_status_overlay(self, viewer):
        term = self.env.unwrapped.action_manager.get_term("grasp")
        reward = sum(term.values[name] * cfg.weight for name, cfg in self.env.cfg.rewards.items())
        values = torch.stack((term.values["mean_force"], reward), dim=-1).cpu().tolist()
        names = [f"{'Trained' if i < 3 else 'AnyGrasp'} {i % 3 + 1}  "
                 for i in range(self.env.num_envs)]
        scene = self.env.unwrapped.hammer.scene_id.cpu().tolist()
        candidate = self.env.unwrapped.hammer.candidate_id.cpu().tolist()
        numbers = [f"S{s} C{c} | {force:.3f} N | {total:.4f}"
                   for s, c, (force, total) in zip(scene, candidate, values)]
        phase = "Final" if term._substep == self.env.cfg.decimation else "Running"
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "\n".join(["Continuous scenes", "Environment", *names, "Enter: new grasps"]),
            "\n".join([self.checkpoint, "Scene Candidate | Peak force | Total reward", *numbers, phase])))

    def _add_env_selection_marker(self, viewer):
        # 标签与网格使用相同平移，切换所选环境时仍对应各自场景。
        for i in range(self.env.num_envs):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            pos = self.offsets[i] - self.offsets[self.env_idx] + (0.0, 0.0, 0.9)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3), pos, np.eye(3).ravel(), np.ones(4))
            geom.label = f"{'T' if i < 3 else 'A'}{i % 3 + 1}"
            viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spacing", type=float, default=1.4)
    args = parser.parse_args()

    saved = json.loads((args.checkpoint.parent / "run_config.json").read_text())
    stage = saved["stage"]
    data_file = args.data_file or saved["data_file"]
    cfg = make_env_cfg(saved["hammer_head_extra_mass"], data_file=data_file, stage=stage,
        num_envs=6, split=args.split, seed=None, training=False)
    cfg.randomize_physics = False
    cfg.events.pop("physics", None)
    cfg.visual_noise = False
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        wrapper = RslRlVecEnvWrapper(env)
        actor, critic = load_models(args.checkpoint, args.device)
        viewer = ComparisonHammeringViewer(wrapper, actor, critic, args.spacing,
            args.checkpoint.name)
        with torch.inference_mode():
            viewer.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
