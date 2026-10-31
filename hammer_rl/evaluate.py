"""同一 scene/candidate、同一固定执行器的 zero / checkpoint 评估与 native 回放。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from .env_cfg import make_env_cfg
from .mdp.observations import sync_data
from .rl_cfg import ppo_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="weight/hammer_rl/reset_bank.npz")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--placement", choices=("all", "flat", "lean_head_down", "lean_head_up", "inverted"),
                        default="all", help="只评估指定初始摆放")
    parser.add_argument("--policy", choices=("zero", "trained"), default="zero")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=32, help="评估次数；0 表示所有选定候选各一次")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viewer", choices=("native",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="逐样本原始指标 npz，建议放在 weight/ 下")
    args = parser.parse_args()
    if args.episodes < 0 or args.num_envs < 1:
        parser.error("episodes 必须非负，num-envs 必须为正")
    if args.policy == "trained" and args.checkpoint is None:
        parser.error("trained 策略需要 --checkpoint")
    with np.load(args.data_file) as source:
        mask = source["split"] == (args.split == "eval")
        if args.placement != "all":
            placement = ("flat", "lean_head_down", "lean_head_up", "inverted").index(args.placement)
            mask &= source["placement"] == placement
        pairs = [(int(s), c) for s in np.flatnonzero(mask)
                 for c in range(int(source["offsets"][s]), int(source["offsets"][s + 1]))]
    if not pairs:
        parser.error("指定划分和摆放下没有抓取候选")
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
        if args.policy == "trained":
            from dataclasses import asdict
            runner = MjlabOnPolicyRunner(wrapper, asdict(ppo_cfg("teacher" if stage == "teacher" else "student")),
                                         log_dir=None, device=args.device)
            loaded = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            actor = loaded["student_state_dict"] if stage == "distill" else loaded["actor_state_dict"]
            runner.alg.load({"actor_state_dict": actor}, {"actor": True}, strict=True)
            policy = runner.get_inference_policy()
        print(f"Evaluation: policy={args.policy}, placement={args.placement}, "
              f"scenes={len(set(s for s, _ in pairs))}, candidates={len(pairs)}, "
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
            for first in range(0, args.episodes, count):
                selected = [pairs[i % len(pairs)] for i in range(first, first + count)]
                scene, candidate = torch.tensor(selected, device=env.device).unbind(-1)
                env.hammer.selection = (scene, candidate)
                # 每批使用确定种子，使冻结观测噪声也可复现。
                env.reset(seed=args.seed + first)
                if viewer is not None:
                    sync_data(env, data)
                # 环境会保留本批生成的状态张量，供下一批在上下文外 reset。
                with torch.no_grad():
                    action = (torch.zeros((count, 6), device=env.device) if policy is None else
                              policy(wrapper.get_observations()))
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
