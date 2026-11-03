"""每轮重新生成六个场景：三个使用训练策略，三个使用深入 30 mm 的 AnyGrasp。"""

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
from hammer_rl.prepare import prepare_scene
from hammer_rl.rl_cfg import ppo_cfg
from visualization.parallel_hammering import ParallelHammeringViewer

PLACEMENTS = ("flat", "lean_head_down", "lean_head_up", "inverted")


class ComparisonHammeringViewer(ParallelHammeringViewer):
    def __init__(self, env, policy, spacing, placement, checkpoint, hammer_head_extra_mass):
        super().__init__(env, policy, spacing)
        self.placement = placement
        self.checkpoint = checkpoint
        self.hammer_head_extra_mass = hammer_head_extra_mass
        self.offsets[3:, 1] = -spacing
        self.cfg.azimuth = 90.0
        self.cfg.elevation = -50.0
        self.cfg.lookat = (0.65 * spacing, -spacing / 2, -0.1)
        self.cfg.distance *= 0.8
        self._show_debug_vis = False

    def _execute_step(self):
        # prepare_scene 每次调用 AnyGrasp，并沿接近方向额外深入 0.03 m。
        rng = np.random.default_rng()
        samples, candidates = zip(*(prepare_scene(rng, PLACEMENTS.index(self.placement),
            1, self.hammer_head_extra_mass) for _ in range(self.env.num_envs)))
        arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        arrays.update({key: np.stack([candidate[0][key] for candidate in candidates])
                       for key in candidates[0][0]})
        self.env.unwrapped.hammer.bank = {
            key: torch.as_tensor(value, device=self.env.device,
                dtype=torch.float32 if value.dtype.kind == "f" else None)
            for key, value in arrays.items()}
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
        names = [f"{'Trained' if i < 3 else 'AnyGrasp'} {i % 3 + 1}  "
                 for i in range(self.env.num_envs)]
        numbers = [f"{force:.3f} N | {total:.4f}" for force, total in values]
        phase = "Final" if term._substep == self.env.cfg.decimation else "Running"
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "\n".join([self.placement, "Environment", *names, "Enter: new grasps"]),
            "\n".join([self.checkpoint, "Peak force | Total reward", *numbers, phase])))

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
    parser.add_argument("--placement", choices=PLACEMENTS, required=True)
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
        indices = torch.arange(env.num_envs, device=env.device)
        env.hammer.selection = (indices, indices)
        runner = MjlabOnPolicyRunner(wrapper,
            asdict(ppo_cfg("teacher" if stage == "teacher" else "student")),
            log_dir=None, device=args.device)
        loaded = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        actor = loaded["student_state_dict"] if stage == "distill" else loaded["actor_state_dict"]
        runner.alg.load({"actor_state_dict": actor}, {"actor": True}, strict=True)
        policy = runner.get_inference_policy()

        def comparison_policy(observations):
            action = torch.zeros((env.num_envs, wrapper.num_actions), device=env.device)
            action[:3] = policy(observations[:3])
            return action

        viewer = ComparisonHammeringViewer(wrapper, comparison_policy, args.spacing,
            args.placement, args.checkpoint.name, saved["hammer_head_extra_mass"])
        with torch.inference_mode():
            viewer.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
