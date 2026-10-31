"""按四类放置配对比较 SDK 最高分候选的敲击力、成功率和五项原始评分。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PLACEMENTS = ("flat", "lean_head_down", "lean_head_up", "inverted")
METRICS = ("peak_force", "lift_success", "valid_strike", "wrap", "orientation",
           "distance", "stability", "force")


def summarize(zero, trained, mask):
    scenes = np.unique(zero["scene_id"][mask])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(scenes), size=(10000, len(scenes)))
    result = {"episodes": int(mask.sum()), "scenes": len(scenes), "metrics": {}}
    for name in METRICS:
        a, b = zero[name][mask], trained[name][mask]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"Nonfinite metric: {name}")
        # 每场景一个 SDK 最高分候选，以场景为单位有放回配对抽样。
        sums = np.array([(b - a)[zero["scene_id"][mask] == s].sum() for s in scenes])
        counts = np.array([(zero["scene_id"][mask] == s).sum() for s in scenes])
        bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
        binary = name in ("lift_success", "valid_strike")
        metric = dict(zero_mean=float(a.mean()), trained_mean=float(b.mean()),
                      delta=float((b - a).mean()),
                      relative_change_percent=float((b.mean() / a.mean() - 1) * 100) if a.mean() != 0 else None,
                      delta_cluster_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist())
        if binary:
            metric.update(zero_successes=int(a.sum()), trained_successes=int(b.sum()),
                          failure_to_success=int(((a == 0) & (b == 1)).sum()),
                          success_to_failure=int(((a == 1) & (b == 0)).sum()))
        result["metrics"][name] = metric
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.zero) as source:
        zero = dict(source)
    with np.load(args.trained) as source:
        trained = dict(source)
    if zero.keys() != trained.keys():
        raise ValueError("Metric names differ")
    for key in ("scene_id", "candidate_id"):
        np.testing.assert_array_equal(zero[key], trained[key])
    pairs = np.column_stack((zero["scene_id"], zero["candidate_id"]))
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate scene/candidate pairs")
    with np.load(args.data_file) as bank:
        top1 = np.array([c == int(bank["offsets"][s]) for s, c in pairs])
        placements = bank["placement"][zero["scene_id"]]
        for s in np.unique(pairs[:, 0]):
            scores = bank["scores"][bank["offsets"][s]:bank["offsets"][s + 1]]
            if not np.all(scores[:-1] >= scores[1:]):
                raise ValueError("SDK candidates must be sorted by score")
    selections = {name: top1 & (placements == i) for i, name in enumerate(PLACEMENTS)}
    result = {name: summarize(zero, trained, mask) for name, mask in selections.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "per_scene.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("placement", "scene_id", "episodes", "metric", "zero_mean", "trained_mean", "delta"))
        for placement, selection in selections.items():
            for scene in np.unique(zero["scene_id"][selection]):
                mask = selection & (pairs[:, 0] == scene)
                for name in METRICS:
                    a, b = zero[name][mask].mean(), trained[name][mask].mean()
                    writer.writerow((placement, int(scene), int(mask.sum()), name, a, b, b - a))
    force_rows = []
    for placement, selection in selections.items():
        for scene in np.unique(zero["scene_id"][selection]):
            mask = selection & (pairs[:, 0] == scene)
            a, b = zero["peak_force"][mask], trained["peak_force"][mask]
            force_rows.append(dict(placement=placement, scene_id=int(scene), episodes=int(mask.sum()),
                zero_mean_N=float(a.mean()), trained_mean_N=float(b.mean()),
                mean_delta_N=float(b.mean() - a.mean())))
    with (args.output_dir / "force_per_scene.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(force_rows[0]))
        writer.writeheader()
        writer.writerows(force_rows)
    lines = ["四类放置的 SDK 最高分候选配对评估：AnyGrasp vs Trained",
             "基线：缓存的 SDK 候选，经项目既有 mount 转换/开度标定后以零残差执行。",
             "Trained：相同候选加指定训练模型的确定性六维残差。",
             "每个场景只统计 SDK 最高分候选；两种策略逐场景配对。",
             "抓取成功 lift_success：抬高至少 0.15 m 且锤柄双侧有效接触。",
             "敲击成功 valid_strike：挥击前有效抓持，且敲击计分窗口内出现锤头与端盖的正法向接触力。",
             "敲击峰值力 peak_force：挥击至卸载时窗内，锤头作用于目标端盖的压缩法向合力峰值，单位 N；无有效抓持或未命中计 0。",
             "平均峰值：先取每次尝试的敲击力峰值，再对全部尝试求平均，包含失败或未命中的 0 N 回合。",
             "趋势图按场景编号排列，各场景独立；不是训练迭代趋势，也不是单次敲击的力-时间曲线。",
             "五项平均分数 wrap/orientation/distance/stability/force 均为未加权原始评分。",
             "95% CI：按场景配对 bootstrap，10000 次，seed=42。", ""]
    labels = {"peak_force": "平均敲击峰值力(N)", "lift_success": "抓取成功率",
              "valid_strike": "敲击成功率", "wrap": "平均包络评分",
              "orientation": "平均朝向评分", "distance": "平均距离评分",
              "stability": "平均稳定评分", "force": "平均敲击力评分"}
    for placement, stats in result.items():
        lines.append(f"{placement}: {stats['scenes']} 场景，{stats['episodes']} 次")
        for name, label in labels.items():
            m = stats["metrics"][name]
            if "zero_successes" in m:
                lo, hi = np.array(m["delta_cluster_bootstrap_95_ci"]) * 100
                lines.append(f"{label}: {m['zero_successes']}/{stats['episodes']} ({m['zero_mean']:.2%}) -> "
                             f"{m['trained_successes']}/{stats['episodes']} ({m['trained_mean']:.2%}); "
                             f"差值 {m['delta'] * 100:+.2f} 个百分点, 95% CI [{lo:+.2f}, {hi:+.2f}]; "
                             f"失败转成功 {m['failure_to_success']}, 成功转失败 {m['success_to_failure']}")
            else:
                lines.append(f"{label}: {m['zero_mean']:.6f} -> {m['trained_mean']:.6f}; 差值 {m['delta']:+.6f}")
        m = stats["metrics"]["peak_force"]
        relative = f"{m['relative_change_percent']:+.2f}%" if m["relative_change_percent"] is not None else "N/A"
        lines.append(f"平均敲击峰值力相对变化：{relative}")
        rows = [row for row in force_rows if row["placement"] == placement]
        lines.append("逐场景敲击峰值力 (N)，AnyGrasp -> Trained：")
        lines.extend(f"  scene {row['scene_id']}: {row['zero_mean_N']:.2f} -> {row['trained_mean_N']:.2f} "
                     f"({row['mean_delta_N']:+.2f})" for row in rows)
        lines.append(f"平均敲击峰值力上升/下降/不变场景数："
                     f"{sum(row['mean_delta_N'] > 0 for row in rows)}/"
                     f"{sum(row['mean_delta_N'] < 0 for row in rows)}/"
                     f"{sum(row['mean_delta_N'] == 0 for row in rows)}")
        lines.append("")
    lines.append("解读：平均敲击峰值力同时受命中率和命中时的敲击力影响。")
    report = "\n".join(lines)
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    titles = ("Impact peak force", "Grasp success", "Strike success", "Wrap score",
              "Orientation score", "Distance score", "Stability score", "Force score")
    placement_labels = ("Flat", "Lean\nhead down", "Lean\nhead up", "Inverted")
    x = np.arange(4)
    for ax, name, title in zip(axes.flat, METRICS, titles):
        binary = name in ("lift_success", "valid_strike")
        scale = 100 if binary else 1
        a = np.array([stats["metrics"][name]["zero_mean"] for stats in result.values()]) * scale
        b = np.array([stats["metrics"][name]["trained_mean"] for stats in result.values()]) * scale
        for bars in (ax.bar(x - .2, a, .4, label="AnyGrasp"),
                     ax.bar(x + .2, b, .4, label="Trained")):
            ax.bar_label(bars, fmt="%.1f" if binary or name == "peak_force" else "%.3f", fontsize=8)
        ax.set_xticks(x, placement_labels)
        ax.set_title(title)
        ax.set_ylabel("Success rate (%)" if binary else "Mean peak force (N)" if name == "peak_force" else "Mean raw score")
        ax.margins(y=.2)
        if binary:
            ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
    fig.suptitle("Four placements: paired SDK top-1 evaluation (failures included)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)
    # 独立导出力的趋势图，便于报告中直接引用。
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, placement in zip(axes.flat, PLACEMENTS):
        rows = [row for row in force_rows if row["placement"] == placement]
        for policy, label, marker in (("zero", "AnyGrasp", "o"), ("trained", "Trained", "s")):
            ax.plot(np.arange(len(rows)), [row[f"{policy}_mean_N"] for row in rows], marker + "-", label=label)
        ax.set_title(f"{placement}: SDK top-1 peak forces")
        ax.set_ylabel("Impact peak force (N)")
        ax.set_xticks(np.arange(len(rows)), [str(row["scene_id"]) for row in rows], rotation=45)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.set_xlabel("Held-out scene ID (independent scenes; not time)")
    fig.suptitle("Four placements: per-scene impact peak force (failures included)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "force_trend.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
