"""逐候选或每场景选择一次，使用相同固定执行器评估与 native 回放。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply

from .env_cfg import make_env_cfg
from .mdp.observations import sync_data
from .inference import load_models, select_candidates


def candidate_batches(env, pairs, episodes):
    """显式绑定 A 场景/候选，供评估和离线抓持快照采集共用。"""
    for first in range(0, episodes, env.num_envs):
        selected = torch.tensor([pairs[i % len(pairs)]
            for i in range(first, first + env.num_envs)], device=env.device)
        env.hammer.selection = selected.unbind(-1)
        env.reset()
        yield first, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="weight/hammer_rl/reset_bank.npz")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--policy", choices=("zero", "trained"), default="zero")
    parser.add_argument("--selection", choices=("all", "sdk", "learned"), default="all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=32, help="0 为全部；all 按候选，sdk/learned 按场景")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viewer", choices=("native",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    parser.add_argument("--grasp-depth-offset", type=float, default=0.0,
                        help="相对缓存抓姿沿接近方向的深度偏移（米）")
    parser.add_argument("--output", type=Path, help="逐样本原始指标 npz，建议放在 weight/ 下")
    args = parser.parse_args()
    if args.episodes < 0 or args.num_envs < 1:
        parser.error("episodes 必须非负，num-envs 必须为正")
    if args.policy == "trained" and args.checkpoint is None:
        parser.error("trained 策略需要 --checkpoint")
    with np.load(args.data_file) as source:
        mask = source["split"] == (args.split == "eval")
        pairs = [(int(s), c) for s in np.flatnonzero(mask)
                 for c in range(int(source["offsets"][s]), int(source["offsets"][s + 1]))]
        candidate_count = len(pairs)
        if args.selection != "all":
            pairs = [(int(s), int(source["offsets"][s]) + int(source["scores"][
                source["offsets"][s]:source["offsets"][s + 1]].argmax()))
                for s in np.flatnonzero(mask)]
    if not pairs:
        parser.error("指定划分下没有抓取候选")
    if args.episodes == 0:
        args.episodes = len(pairs)
    stage = "teacher"
    if args.policy == "trained":
        saved = json.loads((args.checkpoint.parent / "run_config.json").read_text())
        stage = saved["stage"]
        args.hammer_head_extra_mass = saved["hammer_head_extra_mass"]
    count = 1 if args.viewer else min(args.num_envs, args.episodes)
    cfg = make_env_cfg(args.hammer_head_extra_mass, data_file=args.data_file, stage=stage,
        num_envs=count, split=args.split, seed=args.seed, training=False)
    # 对照默认使用相同名义物理参数；--noise 同时用于 zero/teacher/student。
    cfg.randomize_physics = False
    cfg.events.pop("physics", None)
    cfg.visual_noise = args.noise
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapper = RslRlVecEnvWrapper(env)
    policy = None
    try:
        bank = env.hammer.bank
        depth_offset = torch.zeros_like(bank["local_pos"])
        depth_offset[:, 2] = args.grasp_depth_offset
        bank["local_pos"] += quat_apply(bank["local_quat"], depth_offset)
        if args.policy == "trained":
            policy, critic = load_models(args.checkpoint, args.device,
                                         with_critic=args.selection == "learned")
        env.seed(args.seed)
        print(f"Evaluation: policy={args.policy}, selection={args.selection}, "
              f"scenes={len(set(s for s, _ in pairs))}, candidates={candidate_count}, "
              f"episodes={args.episodes}", flush=True)
        rows = []
        model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
        if args.viewer:
            from mujoco import viewer as mj_viewer
            sync_data(env, data)
            context = mj_viewer.launch_passive(model, data)
        else:
            context = nullcontext(None)
        with context as viewer:
            if viewer is not None:
                def display(term):
                    if term._substep % term.substeps == 0:
                        start = time.monotonic()
                        sync_data(env, data)
                        viewer.sync()
                        time.sleep(max(0.0, 0.02 - (time.monotonic() - start)))
                env.action_manager.get_term("grasp").on_physics_step = display
            for first, selected_tensor in candidate_batches(env, pairs, args.episodes):
                scene, candidate = selected_tensor.unbind(-1)
                selected = selected_tensor.tolist()
                # 环境会保留本批生成的状态张量，供下一批在上下文外 reset。
                with torch.no_grad():
                    if args.selection == "learned":
                        result = select_candidates(env, policy, critic, torch.arange(count, device=env.device))
                        action = result["action"]
                        selected = list(zip(scene.tolist(), result["candidate_id"].tolist()))
                    else:
                        action = (torch.zeros((count, 6), device=env.device) if policy is None else
                                  policy(wrapper.get_observations()))
                    if viewer is not None:
                        sync_data(env, data)
                    # 自动 reset 会清空过程量；仅评估在最终子步克隆结果。
                    term = env.action_manager.get_term("grasp")
                    callback = term.on_physics_step
                    captured = {}
                    def capture(term):
                        if callback is not None:
                            callback(term)
                        if term._substep == env.cfg.decimation:
                            captured.update({key: value.clone() for key, value in term.values.items()})
                    term.on_physics_step = capture
                    _, reward, done, _ = wrapper.step(action)
                    term.on_physics_step = callback
                for i in range(min(count, args.episodes - first)):
                    rows.append(dict(scene_id=selected[i][0], candidate_id=selected[i][1],
                        reward=float(reward[i]), **{key: float(value[i]) for key, value in captured.items()}))
                    if args.selection == "learned":
                        rows[-1]["predicted_score"] = float(result["score"][i])
                print(f"Evaluated {len(rows)}/{args.episodes} episodes", flush=True)
        for name in rows[0]:
            if name in ("scene_id", "candidate_id"):
                continue
            values = np.array([row[name] for row in rows])
            print(f"{name}: mean={values.mean():.6f}, p50={np.median(values):.6f}, p90={np.quantile(values, .9):.6f}, max={values.max():.6f}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.output, **{name: np.array([row[name] for row in rows]) for name in rows[0]})
    finally:
        env.close()


if __name__ == "__main__":
    main()
