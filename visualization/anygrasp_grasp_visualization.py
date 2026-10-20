"""以四条线组成的叉形显示 AnyGrasp 候选，分数由蓝、紫到红渐变。"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


def show_grasps(frame, grasps, top_k=50):
    indices = np.argsort(-grasps["scores"])[:top_k]
    scores = grasps["scores"][indices]
    cmap = LinearSegmentedColormap.from_list("grasp_score", ["blue", "purple", "red"], N=257)
    norm = Normalize(scores.min(), scores.max())
    segments, colors = [], []
    # 后画高分候选，减少重叠时低分线条对红色抓姿的遮挡。
    for i, score in zip(indices[::-1], scores[::-1]):
        half_width, depth = grasps["widths"][i] / 2, grasps["depths"][i]
        # 局部 X 为接近方向、Y 为开口方向。横梁在掌侧，柄线在相反一侧。
        fork = np.array([
            [[-0.02, -half_width, 0], [-0.02, half_width, 0]],
            [[-0.02, -half_width, 0], [depth, -half_width, 0]],
            [[-0.02, half_width, 0], [depth, half_width, 0]],
            [[-0.02, 0, 0], [-0.06, 0, 0]],
        ])
        segments.extend(fork @ grasps["rotations"][i].T + grasps["translations"][i])
        colors.extend([cmap(norm(score))] * 4)

    figure = plt.figure(figsize=(10, 8))
    figure.canvas.manager.set_window_title("AnyGrasp poses")
    axes = figure.add_subplot(111, projection="3d")
    points = frame["world_points"]
    axes.scatter(*points.T, c=frame["colors"], s=1, depthshade=False, edgecolors="none")
    axes.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=1.5))
    axes.set(title=f"AnyGrasp top {len(indices)}", xlabel="X (m)", ylabel="Y (m)",
             zlabel="Z (m)", xlim=(-0.5, 0.5), ylim=(-0.7, 0.5), zlim=(0, 0.8))
    axes.set_box_aspect((1, 1.2, 0.8))
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="Grasp score", shrink=0.7)
    figure.tight_layout()
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    return figure
