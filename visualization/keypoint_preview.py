"""根据映射 latent 绘制人手与 OmniPicker 的三维关键点对照图。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

TIP_COLORS = ("#087f8c", "#bc521c")
LINK_COLOR = "#48586c"
LOOP_COLOR = "#8a96a5"


def draw_skeleton(axis, points, links, tips):
    axis.add_collection3d(Line3DCollection(points[links], colors=LINK_COLOR, linewidths=2))
    colors = [LINK_COLOR] * len(points)
    sizes = np.full(len(points), 30)
    for tip, color in zip(tips, TIP_COLORS):
        colors[tip] = color
        sizes[tip] = 75
    axis.scatter(*points.T, c=colors, s=sizes, depthshade=False, edgecolors="white", linewidths=0.7)


def set_3d_view(axes, points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = 0.56 * np.ptp(points, axis=0).max()
    xlim, ylim, zlim = np.column_stack((center - radius, center + radius))
    for axis in axes:
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=xlim, ylim=ylim, zlim=zlim)
        axis.tick_params(labelsize=9)
        for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate.set_major_locator(MultipleLocator(40))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=23, azim=-63)


def create_preview(mapping, candidate_index):
    hand = mapping["hand_grasp"]
    gripper = mapping["gripper"]
    candidate = mapping["mapped_grasp"]["pose_candidates"][candidate_index]
    pairs = candidate["keypoint_correspondences"]
    hand_points = np.asarray(hand["keypoints_O_m"]) * 1000
    rotation = np.asarray(candidate["rotation_OG"])
    gripper_points = (np.asarray(gripper["keypoints_G_m"]) @ rotation.T + candidate["base_pos_O_m"]) * 1000
    hand_tips = [pair["hand_keypoint_index"] for pair in pairs]
    gripper_tips = [gripper["contact_keypoint_indices"][pair["gripper_site"]] for pair in pairs]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 6), subplot_kw={"projection": "3d", "proj_type": "ortho"})
    figure.subplots_adjust(left=0.035, right=0.93, bottom=0.12, top=0.91, wspace=0.12)
    draw_skeleton(left, hand_points, hand["links"], hand_tips)
    draw_skeleton(right, gripper_points, gripper["links"], gripper_tips)
    right.add_collection3d(Line3DCollection(gripper_points[gripper["loop_links"]], colors=LOOP_COLOR,
                                            linestyles="dashed", linewidths=1.5))
    set_3d_view((left, right), np.vstack((hand_points, gripper_points)))
    left.set_title("ContactPose · Human hand")
    right.set_title(f"OmniPicker · Open pose {candidate_index}")

    handles = []
    for pair, color in zip(pairs, TIP_COLORS):
        side = pair["gripper_site"].removeprefix("hand_").removesuffix("_contact_tip")
        label = f"Hand keypoint {pair['hand_keypoint_index']} ↔ {side} tip"
        handles.append(Line2D([], [], marker="o", linestyle="none", color=color, label=label))
    handles.append(Line2D([], [], linestyle="dashed", color=LOOP_COLOR, label="Loop linkage"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncols=3, frameon=False)
    return figure


def save_preview(mapping, candidate_index, output):
    figure = create_preview(mapping, candidate_index)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="保存静态预览图")
    mode.add_argument("--show", action="store_true", help="打开可旋转的对照窗口")
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text())
    if args.show:
        plt.switch_backend("TkAgg")
        figure = create_preview(mapping, args.candidate)
        left, right = figure.axes
        right.shareview(left)
        plt.show()
    else:
        save_preview(mapping, args.candidate, args.output)


if __name__ == "__main__":
    main()
