"""读取训练 TensorBoard 日志，分别保存每轮命中次数和有效命中的平均峰值力曲线。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help="单次训练的日志目录")
    parser.add_argument("--num-envs", type=int, required=True, help="训练实际使用的并行环境数")
    args = parser.parse_args()

    events = EventAccumulator(str(args.log_dir), size_guidance={"scalars": 0},
                              purge_orphaned_data=False).Reload()
    strikes = {event.step: event.value for event in events.Scalars("Episode_Metrics/valid_strike")}
    forces = {event.step: event.value for event in events.Scalars("Episode_Metrics/peak_force")}
    steps = sorted(strikes)
    hits = [round(strikes[step] * args.num_envs) for step in steps]
    # peak_force 已包含未命中时的零值，除以命中比例得到仅有效命中的均值。
    mean_forces = [forces[step] / strikes[step] if strikes[step] > 0 else 0 for step in steps]

    for values, title, ylabel, filename in (
        (hits, "Successful hammer hits per iteration", "Successful hits",
         "hits_per_iteration.png"),
        (mean_forces, "Mean peak force of successful hits", "Mean peak force (N)",
         "mean_hit_force_per_iteration.png"),
    ):
        figure, axes = plt.subplots()
        axes.plot(steps, values)
        axes.set(xlabel="Training iteration", ylabel=ylabel, title=title)
        figure.tight_layout()
        figure.savefig(args.log_dir / filename)
        plt.close(figure)


if __name__ == "__main__":
    main()
