"""三维对照人手和 OmniPicker 开爪构型的关节及指尖。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .contactpose_reader import hand_line_ids, read_records
from .grasp_preview import project_pixels
from .mapping_encoder import mapped_gripper_frame
from .omnipicker_adapter import CONTACT_SITES, FINGER_SIDES, home_gripper_reference, load_gripper_home

ROOT = Path(__file__).resolve().parents[1]
HAND_DATA = ROOT / "data/human_grasp"
CONTACTPOSE_ROOT = HAND_DATA / "source/contactpose"
HAND_TIPS = [8, 4]  # 食指尖、拇指尖，对应候选一的窄爪、宽爪。
DEFAULT_VIEW = dict(elev=23, azim=-63)
TIP_COLORS = ("#087f8c", "#bc521c")
LINK_COLOR = "#48586c"
LOOP_COLOR = "#8a96a5"


def load_hand_pose(pose):
    if pose is not None:
        joints = np.asarray(pose["joints_m"], dtype=float)
        index_tip, thumb_tip = joints[HAND_TIPS]
        # 没有物体接触数据时，用腕部到两指尖中点的方向定义接近轴。
        grasp = dict(closing_axis_O=thumb_tip - index_tip,
                     approach_axis_O=(index_tip + thumb_tip) / 2 - joints[0])
        return joints, hand_line_ids(CONTACTPOSE_ROOT), grasp, "Input pose · Human hand", pose.get("view", DEFAULT_VIEW)

    source = json.loads((HAND_DATA / "processed/hammer_mapping.json").read_text())
    sample = source["source"]["records"][0]
    record = next(read_records(CONTACTPOSE_ROOT, sample["object_name"], [sample["participant_id"]]))
    joints = record["hands"][0] @ source["coordinates"]["canonical_axes_in_source"]
    return joints, record["hand_links"], source["hand_grasp"], "ContactPose · Human hand", DEFAULT_VIEW


def gripper_skeleton(data):
    points = [data.body("base_link").xpos]
    links, loops, tips = [], [], []
    for side, site in zip(FINGER_SIDES, CONTACT_SITES):
        first, second, third, tip, loop_joint, loop_end = range(len(points), len(points) + 6)
        points.extend(data.joint(f"hand_{side}{i}_joint").xanchor for i in (1, 2, 3))
        points.extend((data.site(site).xpos, data.joint(f"hand_{side}_loop_joint").xanchor,
                       data.site(f"hand_{side}_loop_tip").xpos))
        links.extend(((0, first), (first, second), (second, third), (third, tip)))
        loops.extend(((0, loop_joint), (loop_joint, loop_end), (loop_end, third)))
        tips.append(tip)
    return np.array(points), links, loops, tips


def joint_markers(count, tips):
    colors = [LINK_COLOR] * count
    sizes = np.full(count, 30)
    for tip, color in zip(tips, TIP_COLORS):
        colors[tip] = color
        sizes[tip] = 75
    return colors, sizes


def draw_skeleton(axis, points, links, tips):
    axis.add_collection3d(Line3DCollection(points[links], colors=LINK_COLOR, linewidths=2))
    colors, sizes = joint_markers(len(points), tips)
    axis.scatter(*points.T, c=colors, s=sizes, depthshade=False, edgecolors="white", linewidths=0.7)


def draw_overlay(axis, image, points, links, tips):
    axis.imshow(image)
    axis.add_collection(LineCollection(points[links], colors=LINK_COLOR, linewidths=2, zorder=2))
    colors, sizes = joint_markers(len(points), tips)
    axis.scatter(*points.T, c=colors, s=sizes, edgecolors="white", linewidths=0.8, zorder=4)
    axis.set_axis_off()


def render_gripper(model, data, points, rotation, view):
    # 把对照坐标系中的相机方向换算到 XML 的 home 坐标系。
    native_from_comparison = data.body("base_link").xmat.reshape(3, 3) @ rotation.T
    elev, azim, roll = np.radians([view["elev"], view["azim"], view.get("roll", 0)])
    eye = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0])
    up = np.cross(eye, right) * np.cos(roll) + right * np.sin(roll)
    forward = native_from_comparison @ -eye

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (points.min(axis=0) + points.max(axis=0)) / 2
    camera.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
    camera.elevation = np.degrees(np.arctan2(forward[2], np.linalg.norm(forward[:2])))
    camera.distance = 0.34
    options = mujoco.MjvOption()
    model.geom("floor").group[:] = 5
    options.geomgroup[5] = 0
    options.sitegroup[:] = 0

    width, height = 640, 480
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        for eye_camera in renderer.scene.camera:
            eye_camera.up[:] = native_from_comparison @ up
        image = renderer.render()
        view_camera = mujoco.mjv_averageCamera(*renderer.scene.camera)
    return image, project_pixels(points, view_camera, width, height)


def set_3d_view(axes, points, view):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = 0.56 * np.ptp(points, axis=0).max()
    xlim, ylim, zlim = np.column_stack((center - radius, center + radius))
    for axis in axes:
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=xlim, ylim=ylim, zlim=zlim)
        axis.tick_params(labelsize=9)
        for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate.set_major_locator(MultipleLocator(40))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(**view)
    axes[0].shareview(axes[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-pose", type=Path, help="三维姿态 JSON，joints_m 为映射坐标系中的 21 个 XYZ 点，单位 m。"
                        "顺序为手腕、拇指、食指、中指、无名指、小指，每指从近端到指尖。")
    parser.add_argument("--hand-image", type=Path, help="在原图上叠加关键点。与含 joints_uv 的 --hand-pose 一起传入。"
                        "joints_uv 与 joints_m 同序，坐标从左上到右下归一化为 0 到 1。")
    parser.add_argument("--output", type=Path, help="保存图片到指定路径。省略时打开预览窗口。")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "mapping/configs/mapping.yaml").read_text())
    pose = json.loads(args.hand_pose.read_text()) if args.hand_pose else None
    hand, hand_links, grasp, hand_title, view = load_hand_pose(pose)

    model, data = load_gripper_home(ROOT / config["gripper_xml"])
    gripper, links, loops, tips = gripper_skeleton(data)
    reference, tcp_offset = home_gripper_reference(data)
    rotation = mapped_gripper_frame(grasp) @ reference.T
    subplot_kw = {} if args.hand_image else {"projection": "3d", "proj_type": "ortho"}
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 6), subplot_kw=subplot_kw)
    figure.subplots_adjust(left=0.035, right=0.93, bottom=0.12, top=0.91, wspace=0.12)
    if args.hand_image:
        hand_image = plt.imread(args.hand_image)
        height, width = hand_image.shape[:2]
        hand_pixels = np.asarray(pose["joints_uv"]) * [width - 1, height - 1]
        draw_overlay(left, hand_image, hand_pixels, hand_links, HAND_TIPS)
        gripper_image, gripper_pixels = render_gripper(model, data, gripper, rotation, view)
        draw_overlay(right, gripper_image, gripper_pixels, links, tips)
        right.add_collection(LineCollection(gripper_pixels[loops], colors=LOOP_COLOR,
                                             linestyles="dashed", linewidths=1.5, zorder=3))
    else:
        base = data.body("base_link")
        hand = (hand - hand[HAND_TIPS].mean(axis=0)) * 1000
        gripper = ((gripper - base.xpos) @ base.xmat.reshape(3, 3) - tcp_offset) @ rotation.T * 1000
        draw_skeleton(left, hand, hand_links, HAND_TIPS)
        draw_skeleton(right, gripper, links, tips)
        right.add_collection3d(Line3DCollection(gripper[loops], colors=LOOP_COLOR,
                                                linestyles="dashed", linewidths=1.5))
        set_3d_view((left, right), np.vstack((hand, gripper)), view)

    left.set_title(hand_title)
    right.set_title("OmniPicker · Open position")
    figure.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=TIP_COLORS[0],
               label="Index tip ↔ narrow tip"),
        Line2D([], [], marker="o", linestyle="none", color=TIP_COLORS[1],
               label="Thumb tip ↔ wide tip"),
        Line2D([], [], linestyle="dashed", color=LOOP_COLOR, label="Loop linkage"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.035), ncols=3, frameon=False)

    if args.output is None:
        plt.show()
    else:
        figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
