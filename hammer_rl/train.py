"""薄训练入口：原生 manager env、现成 PPO/DistillationRunner 与 checkpoint 衔接。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner

from .env_cfg import make_env_cfg
from .rl_cfg import ppo_cfg, distill_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "distill", "student"), required=True)
    parser.add_argument("--data-file", default="weight/hammer_rl/reset_bank.npz")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=3000, help="本次追加的 update 数")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="仅 student：蒸馏 checkpoint 初始化 actor")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--curriculum-milestones", type=int, nargs=2, default=(500, 1200))
    args = parser.parse_args()
    if args.resume:
        run_dir = args.resume.resolve().parent
        config = json.loads((run_dir / "run_config.json").read_text())
        args.stage = config["stage"]
        args.data_file = config["data_file"]
        args.hammer_head_extra_mass = config["hammer_head_extra_mass"]
        args.curriculum_milestones = config["curriculum_milestones"]
    else:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_dir or Path("weight/hammer_rl/runs") / args.stage / stamp
        config = dict(stage=args.stage, data_file=str(Path(args.data_file).resolve()),
            hammer_head_extra_mass=args.hammer_head_extra_mass,
            curriculum_milestones=list(args.curriculum_milestones), seed=args.seed)
        config["rl"] = distill_cfg() if args.stage == "distill" else asdict(ppo_cfg(args.stage))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    env_cfg = make_env_cfg(args.hammer_head_extra_mass, data_file=args.data_file, stage=args.stage,
        num_envs=args.num_envs, seed=args.seed, curriculum_milestones=args.curriculum_milestones)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))
    train_cfg = config["rl"]
    train_cfg["check_for_nan"] = False
    runner_type = DistillationRunner if args.stage == "distill" else MjlabOnPolicyRunner
    try:
        runner = runner_type(env, train_cfg, str(run_dir), device=args.device)
        if args.resume:
            infos = runner.load(str(args.resume), map_location=args.device)
            runner.current_learning_iteration += 1
            if args.stage == "distill" and infos:
                env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
            # wrapper 构造时已经 reset，恢复课程计数后重新采样初始回合。
            env.reset()
        elif args.stage == "distill":
            runner.load(str(args.teacher_checkpoint), map_location=args.device)
        elif args.stage == "student":
            checkpoint = torch.load(args.init_from, map_location=args.device, weights_only=False)
            runner.alg.load({"actor_state_dict": checkpoint["student_state_dict"]}, {"actor": True}, strict=True)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
        runner.save(str(checkpoint), infos={"env_state": {"common_step_counter": env.unwrapped.common_step_counter}})
        print(f"checkpoint: {checkpoint.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
