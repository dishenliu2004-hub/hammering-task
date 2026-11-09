"""复用 A 完整 GPU 尝试，在挥锤开始前收集真实有效抓持快照。"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

from visualization.camera_rgb_pointcloud import CameraRGBPointCloud
from .env_cfg import make_env_cfg
from .evaluate import candidate_batches
from .inference import load_models
from .mdp.observations import sync_data
from .mdp.resets import STATE_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="weight/hammer_rl/reset_bank_dr_256.npz")
    parser.add_argument("--output", type=Path, default=Path("weight/hammer_rl/strike_reset_bank_256.npz"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--residual-samples", type=int, default=0,
                        help="每个候选在零残差或 actor 输出之外额外执行的随机扰动次数")
    parser.add_argument("--residual-std", type=float, default=0.2, help="原始 6D 动作扰动的标准差")
    args = parser.parse_args()
    with np.load(args.data_file) as source:
        split = source["split"].copy()
        offsets = source["offsets"].copy()
        extra_mass = float(source["hammer_head_extra_mass"])
    torch.manual_seed(args.seed)
    actor = None
    if args.checkpoint:
        actor, _ = load_models(args.checkpoint, args.device, with_critic=False)
    samples = []
    for name, split_id in (("train", 0), ("eval", 1)):
        pairs = [(int(scene), candidate) for scene in np.flatnonzero(split == split_id)
                 for candidate in range(int(offsets[scene]), int(offsets[scene + 1]))]
        cfg = make_env_cfg(extra_mass, data_file=args.data_file, stage="teacher",
            num_envs=args.num_envs, split=name, seed=args.seed, training=False)
        cfg.events.pop("physics")
        cfg.randomize_physics = False
        env = ManagerBasedRlEnv(cfg, device=args.device)
        model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
        camera = CameraRGBPointCloud(model)
        try:
            wrapper = RslRlVecEnvWrapper(env)
            term = env.action_manager.get_term("grasp")
            for residual_sample in range(args.residual_samples + 1):
                for first, _ in candidate_batches(env, pairs, len(pairs)):
                    count = min(env.num_envs, len(pairs) - first)

                    def capture(term):
                        if term._substep != term.swing_start:
                            return
                        ids = term.grasp_valid[:count].nonzero().flatten()
                        gpu = env.sim.data
                        for env_id in ids.tolist():
                            sample = {key: getattr(gpu, key)[env_id].cpu().numpy().copy()
                                      for key in STATE_FIELDS}
                            hammer_pos, hammer_quat = gpu.xpos[env_id, term.hammer], gpu.xquat[env_id, term.hammer]
                            grip_pos, grip_quat = gpu.xpos[env_id, term.base], gpu.xquat[env_id, term.base]
                            sample.update(scene_id=int(env.hammer.scene_id[env_id]),
                                candidate_id=int(env.hammer.candidate_id[env_id]),
                                hammer_pos=hammer_pos.cpu().numpy().copy(),
                                hammer_quat=hammer_quat.cpu().numpy().copy(),
                                rel_grasp_pos=quat_apply_inverse(hammer_quat, grip_pos - hammer_pos).cpu().numpy(),
                                rel_grasp_quat=quat_mul(quat_conjugate(hammer_quat), grip_quat).cpu().numpy(),
                                reference_closure_pos=quat_apply_inverse(grip_quat, hammer_pos - grip_pos).cpu().numpy(),
                                reference_closure_quat=quat_mul(quat_conjugate(grip_quat), hammer_quat).cpu().numpy())
                            sync_data(env, data, env_id=env_id)
                            camera.capture(data)
                            depth = camera.renderer.render()[::8, ::8].copy()
                            depth[(depth <= 0) | (depth >= camera.far)] = 0
                            sample["depth"] = depth[None].astype(np.float32)
                            samples.append(sample)

                    term.on_physics_step = capture
                    with torch.no_grad():
                        action = (torch.zeros((env.num_envs, 6), device=env.device) if actor is None
                                  else actor(wrapper.get_observations()))
                        if residual_sample:
                            action += args.residual_std * torch.randn_like(action)
                        wrapper.step(action)
                    print(f"{name} residual={residual_sample}: {first + count}/{len(pairs)}, "
                          f"valid snapshots={len(samples)}", flush=True)
        finally:
            camera.close()
            env.close()
    samples.sort(key=lambda sample: sample["scene_id"])
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    counts = np.bincount(arrays["scene_id"], minlength=len(split))
    arrays["strike_offsets"] = np.r_[0, counts.cumsum()].astype(np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"saved {args.output.resolve()}: {len(samples)} snapshots, "
          f"{np.count_nonzero(counts)} scenes")


if __name__ == "__main__":
    main()
