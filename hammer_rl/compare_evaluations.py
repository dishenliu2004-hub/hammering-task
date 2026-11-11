"""总体对比 AnyGrasp 与训练权重的敲击成功率和最大敲击力。"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

METRICS = (("valid_strike", 100.0, np.mean),
           ("mean_force", 1.0, np.mean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, help="已有测试场景及 SDK 抓取位姿缓存")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=0, help="0 为全部评估场景；正值用于短运行")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anygrasp-extra-depth", type=float, default=0.03,
                        help="AnyGrasp 原始预测深度上的额外深入量（米），替换缓存中的 0.03 m")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint.parent / "run_config.json").read_text())
    data_file = args.data_file or Path(config["data_file"])
    output_dir = args.output_dir or checkpoint.parent / f"comparison_{checkpoint.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = [
        sys.executable, "-m", "hammer_rl.evaluate",
        "--split", "eval", "--b-pose-file", config["b_pose_file"],
        "--episodes", str(args.episodes), "--num-envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--device", args.device, "--headless",
    ]
    for name, policy, selection in (("zero", "zero", "sdk"),
                                    ("trained", "trained", "learned")):
        command = [*evaluation, "--data-file", str(data_file),
            "--policy", policy, "--selection", selection,
            "--output", str(output_dir / f"{name}.npz")]
        if policy == "zero":
            command.extend(("--hammer-head-extra-mass", str(config["hammer_head_extra_mass"])))
            command.extend(("--grasp-depth-offset", str(args.anygrasp_extra_depth - 0.03)))
        else:
            command.extend(("--checkpoint", str(checkpoint)))
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    results = {}
    for name in ("zero", "trained"):
        with np.load(output_dir / f"{name}.npz") as source:
            results[name] = dict(source)
        (output_dir / f"{name}.npz").unlink()
    comparison_path = output_dir / "metrics_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as comparison_output:
        comparison_writer = csv.writer(comparison_output)
        comparison_writer.writerow((
            "scenes",
            "anygrasp_valid_strike_percent", "trained_valid_strike_percent", "delta_valid_strike_pp",
            "anygrasp_mean_force_N", "trained_mean_force_N", "delta_mean_force_N",
        ))
        comparison_row = ["all"]
        for name, scale, aggregate in METRICS:
            a, b = (aggregate(results[method][name]) * scale
                    for method in ("zero", "trained"))
            comparison_row.extend((a, b, b - a))
        comparison_writer.writerow(comparison_row)
    print(comparison_path)


if __name__ == "__main__":
    main()
