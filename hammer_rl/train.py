"""薄训练入口：原生 manager env、现成 PPO/DistillationRunner 与 checkpoint 衔接。"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner

from .env_cfg import make_env_cfg
from .mdp.curriculum import SGS_PARAMS
from .rl_cfg import ppo_cfg, distill_cfg


class _TeacherSgsRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        term = (
            self.env.unwrapped.curriculum_manager
            .get_term_cfg("placement").func
        )
        infos = {**(infos or {}), "placement_sgs": term.state_dict()}
        super().save(path, infos=infos)


class _RewardLogVecEnvWrapper(RslRlVecEnvWrapper):
    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        # 直接记录加权评分，不按回合时长归一化。
        for name, term in self.cfg.rewards.items():
            extras["log"][f"Episode_Reward/{name}"] = extras["log"][f"Episode_Metrics/{name}"] * term.weight
        extras["log"]["Train/total_reward"] = reward.mean()
        return obs, reward, done, extras


def _enable_best_checkpoint(runner, stage, run_dir):
    best_path = run_dir / "model_best.pt"
    best_score = (
        torch.load(best_path, map_location="cpu", weights_only=False)["infos"]["best_score"]
        if best_path.exists() else float("-inf")
    )
    log = runner.logger.log

    def log_with_best(**kwargs):
        nonlocal best_score
        # 原日志方法会清空 ep_extras，先读取本轮指标。
        score = (
            -kwargs["loss_dict"]["behavior"] if stage == "distill"
            else torch.stack([
                info["Train/total_reward"] for info in runner.logger.ep_extras
            ]).mean().item()
        )
        with redirect_stdout(StringIO()) as output:
            log(**kwargs)
        for line in output.getvalue().splitlines():
            label = line.split(":", 1)[0].strip()
            if (
                "Learning iteration " in line
                or label.startswith("Episode_Reward/")
                or label in ("Train/total_reward", "Total steps", "Iteration time")
                or (label.startswith("Mean ") and label.endswith(" loss"))
            ):
                print(line)
        if score > best_score:
            best_score = score
            runner.save(str(best_path), infos={
                "best_score": best_score,
                "env_state": {"common_step_counter": runner.env.unwrapped.common_step_counter},
            })

    runner.logger.log = log_with_best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "distill", "student"), required=True)
    parser.add_argument("--data-file", default="weight/hammer_rl/reset_bank.npz")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=3000, help="本次追加的 update 数")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-placement-sgs",
        action="store_true",
        help="teacher续训时使用当前课程配置并清空课程状态，保留模型和优化器",
    )
    parser.add_argument("--init-from", type=Path, help="仅 student：蒸馏 checkpoint 初始化 actor")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.resume:
        run_dir = args.resume.resolve().parent
        config = json.loads((run_dir / "run_config.json").read_text())
        args.stage = config["stage"]
        args.data_file = config["data_file"]
        args.hammer_head_extra_mass = config["hammer_head_extra_mass"]
        if args.stage == "teacher" and args.reset_placement_sgs:
            config["placement_sgs"] = SGS_PARAMS.copy()
    else:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_dir or Path("weight/hammer_rl/runs") / args.stage / stamp
        config = dict(stage=args.stage, data_file=str(Path(args.data_file).resolve()),
            hammer_head_extra_mass=args.hammer_head_extra_mass,
            seed=args.seed)
        if args.stage == "teacher":
            config["placement_sgs"] = SGS_PARAMS.copy()
        config["rl"] = distill_cfg() if args.stage == "distill" else asdict(ppo_cfg(args.stage))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    env_cfg = make_env_cfg(args.hammer_head_extra_mass, data_file=args.data_file, stage=args.stage,
        num_envs=args.num_envs, seed=args.seed)
    if args.stage == "teacher":
        env_cfg.curriculum["placement"].params = config["placement_sgs"]
    env = _RewardLogVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))
    train_cfg = config["rl"]
    train_cfg["check_for_nan"] = False
    runner_type = (
        _TeacherSgsRunner if args.stage == "teacher"
        else DistillationRunner if args.stage == "distill"
        else MjlabOnPolicyRunner
    )
    try:
        runner = runner_type(env, train_cfg, str(run_dir), device=args.device)
        if args.resume:
            infos = runner.load(str(args.resume), map_location=args.device)
            runner.current_learning_iteration += 1
            if args.stage == "teacher" and not args.reset_placement_sgs:
                term = env.unwrapped.curriculum_manager.get_term_cfg("placement").func
                term.load_state_dict(infos["placement_sgs"])
            if args.stage == "distill" and infos:
                env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
            # wrapper 构造时已经 reset，恢复状态后重新采样初始回合。
            env.reset()
        elif args.stage == "distill":
            runner.load(str(args.teacher_checkpoint), map_location=args.device)
        elif args.stage == "student":
            checkpoint = torch.load(args.init_from, map_location=args.device, weights_only=False)
            runner.alg.load({"actor_state_dict": checkpoint["student_state_dict"]}, {"actor": True}, strict=True)
        _enable_best_checkpoint(runner, args.stage, run_dir)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
        runner.save(str(checkpoint), infos={"env_state": {"common_step_counter": env.unwrapped.common_step_counter}})
        print(f"checkpoint: {checkpoint.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
