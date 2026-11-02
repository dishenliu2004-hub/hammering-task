"""指定训练权重，评估 SDK 基线和训练策略并导出四类放置的两份 CSV。"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hammer_rl.env_cfg import make_env_cfg

PLACEMENTS = ("flat", "lean_head_down", "lean_head_up", "inverted")
METRICS = ("valid_strike", "peak_force")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, help="独立评估缓存，每类 32 个场景，每场景一个 SDK 最高分候选")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint.parent / "run_config.json").read_text())
    data_file = args.data_file or Path(config["data_file"]).with_name("eval_bank_128.npz")
    output_dir = args.output_dir or checkpoint.parent / f"comparison_{checkpoint.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = [
        sys.executable, "-m", "hammer_rl.evaluate",
        "--split", "eval", "--placement", "all",
        "--episodes", "0", "--num-envs", str(args.num_envs),
        "--device", args.device, "--seed", str(config["seed"]), "--headless",
    ]
    with TemporaryDirectory(prefix="hammer_sdk_eval_") as directory:
        with np.load(data_file) as source:
            sdk_bank = dict(source)
        # 离线候选已额外深入 30 mm；SDK 基线沿局部接近轴回退 20 mm。
        sdk_bank["local_pos"] -= Rotation.from_quat(sdk_bank["local_quat"][:, [1, 2, 3, 0]]).apply([0, 0, 0.02])
        sdk_data_file = Path(directory) / "sdk_bank.npz"
        np.savez_compressed(sdk_data_file, **sdk_bank)
        for policy in ("zero", "trained"):
            command = [*evaluation, "--data-file", str(sdk_data_file if policy == "zero" else data_file),
                "--policy", policy, "--output", str(output_dir / f"{policy}.npz")]
            if policy == "zero":
                command.extend(("--hammer-head-extra-mass", str(config["hammer_head_extra_mass"])))
            else:
                command.extend(("--checkpoint", str(checkpoint)))
            subprocess.run(command, check=True)
    with np.load(output_dir / "zero.npz") as source:
        zero = dict(source)
    with np.load(output_dir / "trained.npz") as source:
        trained = dict(source)
    if zero.keys() != trained.keys():
        raise ValueError("Metric names differ")
    for key in ("scene_id", "candidate_id"):
        np.testing.assert_array_equal(zero[key], trained[key])
    pairs = np.column_stack((zero["scene_id"], zero["candidate_id"]))
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate scene/candidate pairs")
    with np.load(data_file) as bank:
        top1 = np.array([c == int(bank["offsets"][s]) for s, c in pairs])
        placements = bank["placement"][zero["scene_id"]]
        for s in np.unique(pairs[:, 0]):
            scores = bank["scores"][bank["offsets"][s]:bank["offsets"][s + 1]]
            if not np.all(scores[:-1] >= scores[1:]):
                raise ValueError("SDK candidates must be sorted by score")
    selections = {name: top1 & (placements == i) for i, name in enumerate(PLACEMENTS)}
    reward_terms = make_env_cfg(data_file=str(data_file), training=False).rewards
    rewards_path = output_dir / "trained_rewards.csv"
    comparison_path = output_dir / "metrics_comparison.csv"
    with (
        rewards_path.open("w", newline="", encoding="utf-8") as rewards_output,
        comparison_path.open("w", newline="", encoding="utf-8") as comparison_output,
    ):
        rewards_writer = csv.writer(rewards_output)
        comparison_writer = csv.writer(comparison_output)
        rewards_writer.writerow((
            "placement", "scenes", "episodes",
            *(f"reward_{name}" for name in reward_terms), "total_reward",
        ))
        comparison_writer.writerow((
            "placement", "scenes", "episodes",
            "anygrasp_valid_strike_percent", "trained_valid_strike_percent", "delta_valid_strike_pp",
            "anygrasp_peak_force_N", "trained_peak_force_N", "delta_peak_force_N",
        ))
        for placement, mask in selections.items():
            counts = (placement, len(np.unique(zero["scene_id"][mask])), int(mask.sum()))
            # 与训练日志一致：原始评分乘环境配置的奖励权重。
            rewards_writer.writerow((
                *counts,
                *(trained[name][mask].mean() * term.weight for name, term in reward_terms.items()),
                trained["reward"][mask].mean(),
            ))
            comparison_row = list(counts)
            for name, scale in zip(METRICS, (100.0, 1.0)):
                a, b = zero[name][mask].mean() * scale, trained[name][mask].mean() * scale
                comparison_row.extend((a, b, b - a))
            comparison_writer.writerow(comparison_row)
    print(rewards_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
