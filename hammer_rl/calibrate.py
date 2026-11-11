"""冻结最终 actor，采集训练候选的确定性回报并仅校准 critic。"""

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from .env_cfg import make_env_cfg
from .inference import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=0, help="0 为全部训练候选乘重复次数；正值限制采集数")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads((args.checkpoint.parent / "run_config.json").read_text())
    data_file = args.data_file or Path(config["data_file"])
    output_dir = args.output_dir or args.checkpoint.parent / f"calibration_{args.checkpoint.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    actor, critic = load_models(args.checkpoint, args.device)
    actor.requires_grad_(False)
    cfg = make_env_cfg(config["hammer_head_extra_mass"], data_file=data_file,
        b_pose_file=config["b_pose_file"], stage=config["stage"], num_envs=args.num_envs, split="train", seed=args.seed, training=False)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        wrapper = RslRlVecEnvWrapper(env)
        bank = env.hammer.bank
        pairs = [(int(scene), candidate) for scene in (bank["split"] == 0).nonzero().flatten()
                 for candidate in range(int(bank["offsets"][scene]), int(bank["offsets"][scene + 1]))]
        pairs *= args.repeats
        episodes = args.episodes or len(pairs)
        observations, rewards, selections = [], [], []
        for first in range(0, episodes, args.num_envs):
            selected = torch.tensor([pairs[i % len(pairs)]
                for i in range(first, first + args.num_envs)], device=env.device)
            env.hammer.selection = selected.unbind(-1)
            env.reset()
            valid = min(args.num_envs, episodes - first)
            obs = wrapper.get_observations()
            # 初始 buffer 会在 step 的自动 reset 中原地重写，必须执行前保存。
            observations.append(obs[:valid].select(*config["rl"]["obs_groups"]["critic"]).clone().cpu())
            with torch.no_grad():
                _, reward, _, _ = wrapper.step(actor(obs))
            rewards.append(reward[:valid].clone().cpu())
            selections.append(selected[:valid].cpu())
            print(f"Collected {first + valid}/{episodes}", flush=True)
    finally:
        env.close()
    obs, reward, selected = torch.cat(observations), torch.cat(rewards), torch.cat(selections)
    torch.save(dict(observations=obs, reward=reward, scene_id=selected[:, 0],
                    candidate_id=selected[:, 1]), output_dir / "samples.pt")
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    critic.train()
    # 不调用 update_normalization：actor 和 critic 均沿用源模型的运行统计。
    for epoch in range(args.epochs):
        order = torch.randperm(episodes)
        total_loss = 0.0
        for indices in order.split(args.batch_size):
            prediction = critic(obs[indices].to(args.device)).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, reward[indices].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"Epoch {epoch + 1}/{args.epochs}: mse={total_loss / episodes:.6f}", flush=True)
    critic.eval()
    torch.save(dict(actor_state_dict=actor.state_dict(), critic_state_dict=critic.state_dict()),
               output_dir / "deployment.pt")
    config["data_file"] = str(data_file.resolve())
    config["calibration"] = dict(source_checkpoint=str(args.checkpoint.resolve()),
        target="total_reward", split="train", episodes=episodes, repeats=args.repeats,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Deployment checkpoint: {(output_dir / 'deployment.pt').resolve()}")


if __name__ == "__main__":
    main()
