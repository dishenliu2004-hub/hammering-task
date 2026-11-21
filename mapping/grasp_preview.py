"""对照 ContactPose 接触先验、工具坐标系映射和 OmniPicker 参考位姿。"""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch
from scipy.spatial.transform import Rotation

HAMMER_XML = Path(__file__).resolve().parents[1] / "assets/scenes/hammer.xml"
LINK_COLOR = "#bd242e"
JOINT_COLOR = "#3167bb"
REGION_COLOR = "#f85572"
SIDE_VIEW = np.array([[1, 0], [0, -0.28], [0, 0.96]])


def project_pixels(points, camera, width, height):
    relative = points - camera.pos
    right = np.cross(camera.forward, camera.up)
    near_scale = camera.frustum_near / (relative @ camera.forward)
    units_per_pixel = (camera.frustum_top - camera.frustum_bottom) / height
    x = (near_scale * (relative @ right) - camera.frustum_center) / units_per_pixel + width / 2
    y = (camera.frustum_top - near_scale * (relative @ camera.up)) / units_per_pixel
    return np.column_stack((x - 0.5, y - 0.5))


def draw_gripper(axis, mapping):
    mapped = mapping["mapped_grasp"]
    pose = mapped["pose_candidates"][0]
    spec = mujoco.MjSpec.from_file(mapping["gripper"]["source_xml"])
    spec.delete(spec.geom("floor"))
    spec.texture("skybox").rgb1 = [0.95, 0.96, 0.98]
    spec.texture("skybox").rgb2 = [0.70, 0.74, 0.81]
    rotation = np.asarray(pose["rotation_OG"])
    base = spec.body("base_link")
    base.pos = pose["base_pos_O_m"]
    base.quat = Rotation.from_matrix(rotation).as_quat(scalar_first=True)

    hammer_spec = mujoco.MjSpec.from_file(str(HAMMER_XML))
    hammer = hammer_spec.body("Object")
    # 在工具坐标系显示锤体和候选位姿对应的开爪构型。
    for joint in list(hammer.joints):
        hammer_spec.delete(joint)
    hammer.pos = [0, 0, 0]
    hammer.quat = [1, 0, 0, 0]
    hammer_spec.geom("handle").size[:2] = mapping["handle_size_m"]
    spec.worldbody.add_frame().attach_body(hammer, "hammer_", "")
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.asarray(mapped["grip_center_O_m"]) - [0.10, 0, 0]
    # 从手指开合平面的侧面斜看，让两侧手指和锤柄都可见。
    reference = np.asarray(mapping["gripper"]["reference_rotation"])
    view = rotation @ (reference[:, 1] - 0.65 * reference[:, 2])
    camera.distance = 0.55
    camera.azimuth = np.degrees(np.arctan2(view[1], view[0]))
    camera.elevation = np.degrees(np.arctan2(view[2], np.linalg.norm(view[:2])))
    height, width = 360, 480
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera)
        image = renderer.render()
        view_camera = mujoco.mjv_averageCamera(*renderer.scene.camera)
    axis.imshow(image)
    for side, site in zip(("narrow", "wide"), mapping["gripper"]["contact_sites"]):
        joints = [data.joint(f"hand_{side}{i}_joint").xanchor for i in (1, 2, 3)]
        chain = project_pixels(np.array([*joints, data.site(site).xpos]), view_camera, width, height)
        regions = np.array([data.geom(f"hand_{side}_fingertip_{part}").xpos
                            for part in ("proximal", "distal", "tip")])
        contacts = project_pixels(regions, view_camera, width, height)
        axis.plot(*chain.T, color=LINK_COLOR, linewidth=2)
        axis.scatter(*contacts.T, s=115, color=REGION_COLOR, alpha=0.35, edgecolors="none", zorder=3)
        axis.scatter(*chain[:-1].T, s=38, color=JOINT_COLOR, edgecolors="white", linewidths=0.6,
                     zorder=4)
    axis.text(0.5, -0.09, "Highlighted fingertip contact surfaces", transform=axis.transAxes,
              ha="center", fontsize=9)


def draw_contactpose(axis, source, record):
    coordinates = source["coordinates"]
    origin = np.asarray(coordinates["canonical_origin_in_source"])
    projection = np.asarray(coordinates["canonical_axes_in_source"]) @ SIDE_VIEW
    vertices = (record["vertices"] - origin) @ projection
    joints = (record["hands"][0] - origin) @ projection
    hotspots = np.array([spot["position_O"] for spot in source["contact_prior"]["contact_hotspots"]])
    axis.scatter(*vertices.T, s=2, color="#a7aab0", alpha=0.3, edgecolors="none")
    axis.scatter(*(hotspots @ SIDE_VIEW).T, s=155, color=REGION_COLOR, alpha=0.35,
                 edgecolors="none", label="Contact hotspots")
    axis.add_collection(LineCollection(joints[record["hand_links"]], colors=LINK_COLOR,
                                       linewidths=2, label="Hand links"))
    axis.scatter(*joints.T, s=38, color=JOINT_COLOR, edgecolors="white", linewidths=0.6,
                 label="Hand joints", zorder=3)
    axis.set_aspect("equal")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False,
                fontsize=8, handletextpad=0.4, columnspacing=0.8)


def draw_mapping(axis, mapping):
    mapped = mapping["mapped_grasp"]
    radius, half_length = mapping["handle_size_m"]
    center = np.asarray(mapped["grip_center_O_m"]) @ SIDE_VIEW
    sigma = mapped["u_sigma_RL"] * half_length
    axis.plot([-half_length, half_length], [0, 0], color="#a7aab0", lw=12, solid_capstyle="round")
    axis.plot(center[0] + np.array([-sigma, sigma]), [0, 0], color=REGION_COLOR, alpha=0.5, lw=12,
              solid_capstyle="butt", label="Mapped grasp region (±1σ)")
    for field, color, label in (
        ("preferred_closing_axis_O", LINK_COLOR, "Closing axis"),
        ("preferred_approach_axis_O", "#36876d", "Approach axis"),
    ):
        vector = np.asarray(mapped[field]) @ SIDE_VIEW * (1.8 * radius)
        axis.quiver(*center, *vector, angles="xy", scale_units="xy", scale=1, color=color,
                    width=0.013, label=label, zorder=3)
    axis.scatter(*center, s=52, color=JOINT_COLOR, edgecolors="white", linewidths=0.6, zorder=4)
    for x, label in ((-half_length, "Work end"), (half_length, "Free end")):
        axis.annotate(label, (x, 0), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=9)
    axis.set_xlim(-1.4 * half_length, 1.4 * half_length)
    axis.set_ylim(-2 * radius, 2 * radius)
    axis.set_aspect("equal")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)


def save_preview(mapping, source, record, output):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), width_ratios=(1, 1, 1.25))
    figure.subplots_adjust(left=0.025, right=0.98, bottom=0.20, top=0.86, wspace=0.20)
    draw_contactpose(axes[0], source, record)
    draw_mapping(axes[1], mapping)
    draw_gripper(axes[2], mapping)
    u_rl = mapping["mapped_grasp"]["preferred_u_RL"]
    titles = ("ContactPose · human hammer grasp", f"Tool-frame mapping · u_RL = {u_rl:.2f}",
              "OmniPicker · mapped open pose")
    for axis, title in zip(axes, titles):
        axis.axis("off")
        bounds = axis.get_position(original=True)
        figure.text(bounds.x0 + bounds.width / 2, 0.88, title, ha="center", fontsize=12)
    for left, right in pairwise(axes):
        figure.add_artist(ConnectionPatch((1, 0.5), (0, 0.5), "axes fraction", "axes fraction",
                                          axesA=left, axesB=right, arrowstyle="-|>",
                                          shrinkA=6, shrinkB=6, linewidth=1.5, mutation_scale=17,
                                          color=JOINT_COLOR))
    figure.savefig(output, dpi=150)
    plt.close(figure)
