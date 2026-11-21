"""薄训练入口：原生 manager env、现成 PPO/DistillationRunner 与 checkpoint 衔接。"""

import argparse
from contextlib import redirect_stdout
from copy import copy
from dataclasses import asdict
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner
from tensordict import TensorDict

from .collection import HammerTrainingEnv
from .env_cfg import make_env_cfg
from .mdp.curriculum import SGS_PARAMS
from .mdp.resets import restore_start
from .rl_cfg import ROLLOUT_STEPS, ppo_cfg, distill_cfg


class _TeacherSgsRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        term = (
            self.env.unwrapped.curriculum_manager
            .get_term_cfg("scene_sgs").func
        )
        infos = {**(infos or {}), "scene_sgs": term.state_dict()}
        super().save(path, infos=infos)


class _TrainingVecEnv(RslRlVecEnvWrapper):
    """将异步完成的单动作尝试整理成原生 runner 的 N 行采样批次。"""

    def bind_algorithm(self, alg):
        self.alg = alg
        self.sample = alg.act
        self.second_batch = None
        alg.act = self._act

    def _act(self, obs):
        if self.second_batch is not None:
            self.alg.transition = self.second_batch[0]
            return self.alg.transition.actions
        # 新一轮按物理世界顺序派发，上一批结果按样本顺序交给 runner。
        return self.sample(self.get_observations().clone())

    def step(self, actions):
        if self.second_batch is not None:
            _, result = self.second_batch
            self.second_batch = None
            return result

        env = self.unwrapped
        caller = torch.cuda.current_stream(self.device)
        env._stream.wait_stream(caller)
        with torch.cuda.stream(env._stream):
            first, self.second_batch = self._collect(actions)
        caller.wait_stream(env._stream)
        self.alg.transition, result = first
        return result

    def _collect(self, actions):
        env, n = self.unwrapped, self.num_envs
        size = ROLLOUT_STEPS * n
        worlds = env._all_envs
        slots = worlds.clone()
        post_reset = self.alg.transition.observations.new_empty(size)
        rewards = torch.empty(size, device=self.device)
        metrics = {name: torch.empty_like(rewards) for name in self.cfg.metrics}
        curriculum_logs = []
        sample_groups = [[] for _ in range(ROLLOUT_STEPS)]
        sample_groups[0].append(copy(self.alg.transition))
        waiting = []
        env.start_attempts(worlds, actions)
        started, completed = n, 0
        while completed < size:
            ids, reward, scores = env.advance()
            rows = slots[ids]
            rewards[rows] = reward
            for name, value in scores.items():
                metrics[name][rows] = value
            completed += len(ids)
            obs, extras = self.env.reset(env_ids=ids)
            obs = TensorDict(obs, batch_size=[n])[ids]
            post_reset[rows] = obs
            if completed // n > len(curriculum_logs):
                env.common_step_counter += 1
                curriculum_logs.append({key: value for key, value in extras["log"].items()
                                        if key.startswith("Curriculum/")})

            count = min(len(ids), size - started)
            if count:
                slots[ids[:count]] = torch.arange(started, started + count, device=self.device)
                action = self.sample(obs[:count])
                sample_groups[1].append(copy(self.alg.transition))
                env.start_attempts(ids[:count], action)
                started += count
            if count < len(ids) and completed < size:
                waiting.append(ids[count:])

        # MJWarp 始终整批积分。恢复收尾期间空闲世界已经抽好的初态，下一轮沿用同一观测。
        if waiting:
            idle = torch.cat(waiting)
            env.sim.reset(idle)
            restore_start(env, idle, env.hammer.candidate_id[idle])
            env.sim.forward()

        ready = []
        done = torch.ones(n, dtype=torch.long, device=self.device)
        for batch, group in enumerate(sample_groups):
            rows = slice(batch * n, (batch + 1) * n)
            transition = group[0]
            if len(group) > 1:
                for name in ("observations", "actions", "values", "actions_log_prob", "privileged_actions"):
                    if getattr(transition, name) is not None:
                        setattr(transition, name, torch.cat([getattr(t, name) for t in group]))
                if transition.distribution_params is not None:
                    transition.distribution_params = tuple(torch.cat(parts) for parts in
                        zip(*(t.distribution_params for t in group)))
            log = curriculum_logs[batch]
            log.update({f"Episode_Metrics/{name}": value[rows].mean() for name, value in metrics.items()})
            for name, term in self.cfg.rewards.items():
                log[f"Episode_Reward/{name}"] = log[f"Episode_Metrics/{name}"] * term.weight
            log["Episode_Termination/attempt_complete"] = n
            log["Train/total_reward"] = rewards[rows].mean()
            ready.append((transition, (post_reset[rows], rewards[rows], done, {"log": log})))
        return ready[0], ready[1] if len(ready) == 2 else None


def _enable_best_checkpoint(runner, stage, run_dir, *, recipe_changed=False):
    best_path = run_dir / "model_best.pt"
    best_score = (
        torch.load(best_path, map_location="cpu", weights_only=False)["infos"]["best_score"]
        if not recipe_changed and best_path.exists() else float("-inf")
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
    parser.add_argument("--data-file", default="weight/hammer_rl/reset_bank_ab_256.npz")
    parser.add_argument("--split", choices=("train", "all"), default="train")
    parser.add_argument("--reward-recipe", type=Path, help="离线 RewardRecipe JSON；省略时沿用原奖励")
    parser.add_argument("--reward-recipe-override", type=Path, help="本次续训采用的配方")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=5000, help="本次追加的 update 数")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-scene-sgs",
        action="store_true",
        help="teacher续训时使用本次数据路径和当前课程配置，清空课程状态并保留模型和优化器",
    )
    parser.add_argument("--init-from", type=Path, help="PPO 同结构权重初始化，不恢复优化器")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--hammer-head-extra-mass", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    recipe_changed = False
    if args.resume:
        run_dir = args.log_dir or args.resume.resolve().parent
        config = json.loads((args.resume.resolve().parent / "run_config.json").read_text())
        if args.log_dir:
            config["seed"] = args.seed
        args.stage = config["stage"]
        args.hammer_head_extra_mass = config["hammer_head_extra_mass"]
        if args.stage == "teacher" and args.reset_scene_sgs:
            # 显式重置课程时采用本次缓存路径，并清除旧课程配置键。
            config.pop("placement_sgs", None)
            config["scene_sgs"] = SGS_PARAMS.copy()
            config["data_file"] = str(Path(args.data_file).resolve())
        args.data_file = config["data_file"]
        config.setdefault("split", "train")
        args.reward_recipe = args.reward_recipe_override or config.get("reward_recipe")
    else:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_dir or Path("weight/hammer_rl/runs") / args.stage / stamp
        config = dict(stage=args.stage, data_file=str(Path(args.data_file).resolve()), split=args.split,
            hammer_head_extra_mass=args.hammer_head_extra_mass,
            seed=args.seed)
        if args.stage == "teacher":
            config["scene_sgs"] = SGS_PARAMS.copy()
        config["rl"] = distill_cfg() if args.stage == "distill" else asdict(ppo_cfg(args.stage))
    if args.reward_recipe is not None and (not args.resume or args.reward_recipe_override):
        recipe = json.loads(args.reward_recipe.read_text())
        recipe_changed = args.resume is not None and config.get("reward_recipe") != str(args.reward_recipe.resolve())
        config.update(reward_recipe=str(args.reward_recipe.resolve()),
                      recipe_id=recipe["recipe_id"], recipe_version=recipe["version"])
    # 续训时也按当前采集批数分配 RSL-RL 存储。
    if ROLLOUT_STEPS not in (1, 2):
        raise ValueError("ROLLOUT_STEPS must be 1 or 2.")
    config["rl"]["num_steps_per_env"] = ROLLOUT_STEPS
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    env_cfg = make_env_cfg(args.hammer_head_extra_mass, data_file=args.data_file, stage=args.stage,
        num_envs=args.num_envs, split=config["split"], seed=args.seed, reward_recipe=args.reward_recipe)
    if args.stage == "teacher":
        env_cfg.curriculum["scene_sgs"].params = config["scene_sgs"]
    env = _TrainingVecEnv(HammerTrainingEnv(env_cfg, device=args.device))
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
            if args.stage == "teacher" and not args.reset_scene_sgs:
                term = env.unwrapped.curriculum_manager.get_term_cfg("scene_sgs").func
                term.load_state_dict(infos["scene_sgs"])
            if args.stage == "distill" and infos:
                env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
            # wrapper 构造时已经 reset，恢复状态后重新采样初始回合。
            env.reset()
        elif args.stage == "distill":
            runner.load(str(args.teacher_checkpoint), map_location=args.device)
        elif args.init_from:
            checkpoint = torch.load(args.init_from, map_location=args.device, weights_only=False)
            actor_key = "student_state_dict" if "student_state_dict" in checkpoint else "actor_state_dict"
            runner.alg.actor.load_state_dict(checkpoint[actor_key], strict=True)
            if "critic_state_dict" in checkpoint:
                runner.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        env.bind_algorithm(runner.alg)
        _enable_best_checkpoint(runner, args.stage, run_dir, recipe_changed=recipe_changed)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
        runner.save(str(checkpoint), infos={"env_state": {"common_step_counter": env.unwrapped.common_step_counter}})
        print(f"checkpoint: {checkpoint.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
