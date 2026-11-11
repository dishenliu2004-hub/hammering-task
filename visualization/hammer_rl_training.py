"""读取训练 TensorBoard 日志，保存总奖励得分随训练轮次变化的折线图。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help="单次训练的日志目录")
    args = parser.parse_args()

    events = EventAccumulator(str(args.log_dir), size_guidance={"scalars": 0},
                              purge_orphaned_data=False).Reload()
    rewards = events.Scalars("Train/total_reward")
    steps = [event.step for event in rewards]
    values = [event.value for event in rewards]

    figure, axes = plt.subplots()
    axes.plot(steps, values)
    axes.set(xlabel="Training iteration", ylabel="Total reward", title="Total reward per iteration")
    figure.tight_layout()
    figure.savefig(args.log_dir / "total_reward_per_iteration.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
