"""从独立 OmniPicker XML 的 home 状态读取接触参考系和机构参数。"""

import mujoco
import numpy as np

FINGER_SIDES = ("narrow", "wide")
CONTACT_SITES = tuple(f"hand_{side}_contact_tip" for side in FINGER_SIDES)


def load_gripper_home(xml):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def home_gripper_reference(data):
    base = data.body("base_link")
    rotation = base.xmat.reshape(3, 3)
    tips = np.array([data.site(name).xpos for name in CONTACT_SITES])
    closing = rotation.T @ (tips[1] - tips[0])
    home_tcp_offset = rotation.T @ (tips.mean(axis=0) - base.xpos)
    # +Z 与现有抓姿解码接近轴一致；参考对向轴投影到其正交平面。
    approach = np.array([0., 0., 1.])
    closing -= np.dot(closing, approach) * approach
    closing /= np.linalg.norm(closing)
    reference = np.column_stack((closing, np.cross(approach, closing), approach))
    return reference, home_tcp_offset


def gripper_skeleton(data):
    base = data.body("base_link")
    points = [base.xpos]
    links, loops, tips = [], [], {}
    for side, site in zip(FINGER_SIDES, CONTACT_SITES):
        first, second, third, tip, loop_joint, loop_end = range(len(points), len(points) + 6)
        points.extend(data.joint(f"hand_{side}{i}_joint").xanchor for i in (1, 2, 3))
        points.extend((data.site(site).xpos, data.joint(f"hand_{side}_loop_joint").xanchor,
                       data.site(f"hand_{side}_loop_tip").xpos))
        links.extend(((0, first), (first, second), (second, third), (third, tip)))
        loops.extend(((0, loop_joint), (loop_joint, loop_end), (loop_end, third)))
        tips[site] = tip
    local_points = (np.asarray(points) - base.xpos) @ base.xmat.reshape(3, 3)
    return local_points, links, loops, tips


def gripper_description(xml):
    model, data = load_gripper_home(xml)
    reference, home_tcp_offset = home_gripper_reference(data)
    points, links, loops, tips = gripper_skeleton(data)
    geom_names = [model.geom(i).name for i in range(model.ngeom)
                  if model.geom_contype[i] or model.geom_conaffinity[i]]
    springs = [i for i in range(model.njnt) if model.jnt_stiffness[i] > 0]
    return dict(name="omnipicker", type="underactuated_two_finger", actuator_count=model.nu,
        actuators=[dict(name=model.actuator(i).name, ctrlrange=model.actuator_ctrlrange[i].tolist())
                   for i in range(model.nu)],
        contact_parts=["proximal", "distal", "tip", "base"],
        collision_geoms=geom_names, contact_sites=list(CONTACT_SITES),
        reference_rotation=reference.tolist(),
        home_tcp_offset_G_m=home_tcp_offset.tolist(),
        passive_springs=[dict(name=model.joint(i).name, stiffness=float(model.jnt_stiffness[i]),
                             springref=float(model.qpos_spring[model.jnt_qposadr[i]])) for i in springs],
        equalities=[dict(name=model.equality(i).name, type=int(model.eq_type[i]),
                         data=model.eq_data[i].tolist()) for i in range(model.neq)],
        keypoints_G_m=points.tolist(), links=links, loop_links=loops, contact_keypoint_indices=tips)
