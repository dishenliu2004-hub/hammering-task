"""ICRA-BUILD 的严格真包络与指节角度评分，适配批量 Torch 状态。"""

from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import torch
import trimesh


ENC_THETA_MAX_N_DEG = 66.129901697435
ENC_THETA_MAX_W_DEG = 65.615340743932
ENCLOSURE_BALANCE = 0.20
ENCLOSURE_SHAPE_EXPONENT = 0.70
ENCLOSURE_EPSILON = 1e-8


def build_enclosure_geometry(model, device):
    """缓存关节/site 索引、外切线局部方向及膨胀 1 mm 的目标碰撞凸包。"""
    # 来自 direct_rollout._body_convex_parts；本场景目标为单刚体锤子。
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hammer = model.body("hammer_Object").id
    root_rotation = data.xmat[hammer].reshape(3, 3)
    parts = []
    for geom_id in np.flatnonzero(model.geom_bodyid == hammer):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius, half_length = model.geom_size[geom_id, :2]
            vertices = trimesh.creation.capsule(radius=radius, height=2 * half_length).vertices
        else:
            mesh_id = model.geom_dataid[geom_id]
            start = model.mesh_vertadr[mesh_id]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        local = ((vertices @ rotation.T + data.geom_xpos[geom_id]) - data.xpos[hammer]) @ root_rotation
        equations = ConvexHull(local).equations.copy()
        equations[:, 3] -= 0.001 * np.linalg.norm(equations[:, :3], axis=1)
        parts.append(torch.tensor(equations, dtype=torch.float32, device=device))

    base = model.body("gripper_base_link").id
    palm = model.geom("gripper_base_collision_part1").id
    mesh = model.geom_dataid[palm]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    palm_local = ((vertices @ data.geom_xmat[palm].reshape(3, 3).T + data.geom_xpos[palm])
                  - data.xpos[base]) @ data.xmat[base].reshape(3, 3)

    sides = ("narrow", "wide")
    origins = np.array([model.site(f"gripper_hand_{side}_outer_tip").pos for side in sides])
    # 原 ominipicker_*_outer_tangent_direction 的局部坐标。
    points = np.array(((0.055776885, -0.026843479, 0.001318760),
                       (0.055588539, 0.026719227, -0.002056346)))
    directions = points - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return SimpleNamespace(
        hammer=hammer, base=base, convex_parts=tuple(parts), support_height=float(palm_local[:, 2].max()),
        second_joints=[model.joint(f"gripper_hand_{side}2_joint").id for side in sides],
        distal_joints=[model.joint(f"gripper_hand_{side}3_joint").id for side in sides],
        distal_bodies=[model.body(f"gripper_hand_{side}3_Link").id for side in sides],
        tips=[model.site(f"gripper_hand_{side}_contact_tip").id for side in sides],
        origins=[model.site(f"gripper_hand_{side}_outer_tip").id for side in sides],
        directions=torch.tensor(directions, dtype=torch.float32, device=device),
        theta_max=torch.tensor((ENC_THETA_MAX_N_DEG, ENC_THETA_MAX_W_DEG),
                               dtype=torch.float32, device=device))


def compute_handle_seating(contact, data, geometry):
    """锤柄在基座前端面承受至少 0.1 N，位置容差 2 mm，法向偏角不超过 30°。"""
    axis = data.xmat[:, geometry.base].reshape(-1, 3, 3)[:, :, 2]
    height = ((contact.pos - data.xpos[:, geometry.base, None]) * axis[:, None]).sum(-1)
    alignment = (contact.normal * axis[:, None]).sum(-1)
    return ((contact.found > 0) & (contact.force[:, :, 0].abs() >= 0.1)
            & ((height - geometry.support_height).abs() <= 0.002)
            & (alignment >= 3**0.5 / 2)).any(-1)


def _segment_union_inside_length(origin, direction, length_m, convex_parts):
    """direct_rollout 同名函数的批量半空间裁剪及区间并集。"""
    starts, stops = [], []
    for equations in convex_parts:
        values = origin @ equations[:, :3].T + equations[:, 3]
        slopes = direction @ equations[:, :3].T
        parallel = slopes.abs() <= 1e-12
        boundary = -values / torch.where(parallel, 1.0, slopes)
        lower = torch.where(~parallel & (slopes < 0), boundary, 0.0).amax(-1)
        upper = torch.where(~parallel & (slopes > 0), boundary, length_m[..., None]).amin(-1)
        valid = ~(parallel & (values > 0)).any(-1) & (upper > lower)
        starts.append(torch.where(valid, lower, 0.0))
        stops.append(torch.where(valid, upper, 0.0))
    starts, order = torch.stack(starts, -1).sort(-1)
    stops = torch.stack(stops, -1).gather(-1, order)
    previous = torch.cat((torch.zeros_like(stops[..., :1]), stops.cummax(-1).values[..., :-1]), -1)
    return (stops - torch.maximum(starts, previous)).clamp_min(0).sum(-1)


def _outer_tangent_segments(origins, directions, tool_rotation):
    """沿用 direct_rollout 的 TCP YZ 投影求交与前向线段判据。"""
    directions_tool = directions @ tool_rotation
    delta = ((origins[:, 1] - origins[:, 0])[:, None] @ tool_rotation)[:, 0]
    left, right = directions_tool[:, 0], directions_tool[:, 1]
    determinant = right[:, 1] * left[:, 2] - left[:, 1] * right[:, 2]
    nonparallel = determinant.abs() > 1e-6
    denominator = torch.where(nonparallel, determinant, 1.0)
    distances = torch.stack((right[:, 1] * delta[:, 2] - delta[:, 1] * right[:, 2],
                             left[:, 1] * delta[:, 2] - delta[:, 1] * left[:, 2]), -1) / denominator[:, None]
    valid = (nonparallel & torch.isfinite(distances).all(-1)
             & (distances > 0).all(-1) & (distances <= 0.15).all(-1))
    return torch.where(valid[:, None], distances, 0.0), valid


def _distal_enclosure_geometry(v23, v_tip, v_inward, plane_normal):
    """driver_calibration 同名函数的投影锐角与镜像外弯判据。"""
    normal = plane_normal / plane_normal.norm(dim=-1, keepdim=True)
    projected = []
    for vector in (v23, v_tip, v_inward):
        vector = vector - normal * (vector * normal).sum(-1, keepdim=True)
        projected.append(vector / vector.norm(dim=-1, keepdim=True))
    base, tip, inward = projected
    cross = torch.cross(base, tip, dim=-1)
    theta_deg = torch.rad2deg(torch.atan2(cross.norm(dim=-1), (base * tip).sum(-1).abs()))
    side_tip = (normal * cross).sum(-1)
    side_inward = (normal * torch.cross(base, inward, dim=-1)).sum(-1)
    return theta_deg, (side_tip.abs() > 1e-10) & (side_tip * side_inward < 0)


def enclosure_score(left, right, outer_any):
    """network.scoring.enclosure_score 的 Torch 公式及原有常数。"""
    upper = torch.maximum(left, right)
    total = left + right
    harmonic = 2 * left * right / total.clamp_min(ENCLOSURE_EPSILON)
    harmonic = torch.where(total <= ENCLOSURE_EPSILON, 0.0, harmonic)
    negative = -((1 - ENCLOSURE_BALANCE) * harmonic + ENCLOSURE_BALANCE * upper)
    positive = ((1 - ENCLOSURE_BALANCE) * harmonic.pow(ENCLOSURE_SHAPE_EXPONENT)
                + ENCLOSURE_BALANCE * upper.pow(ENCLOSURE_SHAPE_EXPONENT))
    return torch.where(upper <= ENCLOSURE_EPSILON, 0.0, torch.where(outer_any, negative, positive))


def compute_enclosure_score(data, geometry, palm_contact):
    """返回闭合末帧的 y_true_enc * S_enc(angle)，每个环境一个得分。"""
    base_rotation = data.xmat[:, geometry.base].reshape(-1, 3, 3)
    origins = data.site_xpos[:, geometry.origins]
    distal_rotation = data.xmat[:, geometry.distal_bodies].reshape(-1, 2, 3, 3)
    directions = (distal_rotation @ geometry.directions[None, :, :, None])[..., 0]
    lengths, valid = _outer_tangent_segments(origins, directions, base_rotation)
    target_rotation = data.xmat[:, geometry.hammer].reshape(-1, 3, 3)
    origins_local = (origins - data.xpos[:, geometry.hammer, None]) @ target_rotation
    directions_local = directions @ target_rotation
    penetration = _segment_union_inside_length(origins_local, directions_local, lengths, geometry.convex_parts)
    true_enclosure = palm_contact & valid & (penetration <= 0.003).all(-1)

    # 原 v4 TCP 与 base 同向，局部位置为 (0, 0, 0.14825)。
    tcp = data.xpos[:, geometry.base] + 0.14825 * base_rotation[:, :, 2]
    second = data.xanchor[:, geometry.second_joints]
    distal = data.xanchor[:, geometry.distal_joints]
    angles, outer = _distal_enclosure_geometry(distal - second,
        data.site_xpos[:, geometry.tips] - distal, tcp[:, None] - 0.5 * (second + distal),
        data.xaxis[:, geometry.distal_joints])
    # direct_rollout._enclosure_sides：外弯侧为 0.1，内弯侧从 5° 起归一化。
    sides = torch.where(outer, 0.1, ((angles - 5.0) / (geometry.theta_max - 5.0)).clamp(0, 1))
    score = enclosure_score(sides[:, 0], sides[:, 1], outer.any(-1))
    return torch.where(true_enclosure, score, 0.0)
