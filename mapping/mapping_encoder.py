"""接触统计与解析坐标映射，不含学习参数。"""

import numpy as np


def unit(vector):
    return vector / np.linalg.norm(vector)


def canonical_frame(frame):
    work, free, transverse = (np.asarray(frame[name], dtype=float)
                              for name in ("work_end", "free_end", "transverse_point"))
    x = unit(free - work)
    y = transverse - work
    y = unit(y - np.dot(y, x) * x)
    return work, np.column_stack((x, y, np.cross(x, y))), np.linalg.norm(free - work)


def encode(task, records, source_frame, gripper, config):
    origin, rotation, length = canonical_frame(source_frame)
    points, normals, weights, frames, sources, coverages = [], [], [], [], [], []
    bins = config["angular_bins"]
    for record in records:
        vertices = (record["vertices"] - origin) @ rotation
        u = vertices[:, 0] / length
        # 只编码已测量的把手区域；未测量点不参与接触或覆盖统计。
        selected = record["measured"] & (u >= 0) & (u <= 1)
        p = vertices[selected]
        w = record["intensity"][selected]
        center = np.average(p, axis=0, weights=w)
        hands = [(joints - origin) @ rotation for joints in record["hands"]]
        # 双手记录选接触中心最近的掌部，不将其拆为两个独立示范。
        joints = min(hands, key=lambda hand: np.linalg.norm(hand[[0, 5, 9, 13, 17]].mean(0) - center))
        x = unit(joints[4] - joints[8])
        z = center - joints[0]
        z = unit(z - np.dot(z, x) * x)
        frames.append(np.column_stack((x, np.cross(z, x), z)))
        theta = np.arctan2(p[:, 2], p[:, 1])
        angular = np.floor((theta + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        coverages.append(len(np.unique(angular[w >= config["contact_threshold"]])) / bins)
        points.append(p)
        normals.append(record["normals"][selected] @ rotation)
        # 每条 participant × intent × object 等权，不按帧数或网格密度增加示范权重。
        weights.append(w / w.sum())
        sources.append({key: record[key] for key in
                        ("participant_id", "object_name", "intent", "annotation", "contact_map", "frame_count")})
    p, n, w = np.concatenate(points), np.concatenate(normals), np.concatenate(weights)
    w /= len(sources)
    u = p[:, 0] / length
    mean = np.dot(w, u)
    std = np.sqrt(np.dot(w, (u - mean)**2))
    order = np.argsort(u)
    interval = np.interp(config["interval_quantiles"], np.cumsum(w[order]), u[order])
    # 对向轴符号无关；主特征向量与首条记录同向以固定输出符号。
    axes = np.array(frames)[:, :, 0]
    _, eigenvectors = np.linalg.eigh(axes.T @ axes)
    x = eigenvectors[:, -1]
    x *= 1 if np.dot(x, axes[0]) >= 0 else -1
    z = np.mean(np.array(frames)[:, :, 2], axis=0)
    z = unit(z - np.dot(z, x) * x)
    hand_frame = np.column_stack((x, np.cross(z, x), z))
    angular = np.floor((np.arctan2(p[:, 2], p[:, 1]) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    axial = np.minimum((u * config["axial_bins"]).astype(int), config["axial_bins"] - 1)
    cells = axial * bins + angular
    occupied = np.unique(cells)
    strongest = sorted(occupied, key=lambda cell: -w[cells == cell].sum())[:config["hotspot_count"]]
    hotspots = []
    for cell in strongest:
        mask = cells == cell
        hotspots.append(dict(position_O=np.average(p[mask], axis=0, weights=w[mask]).tolist(),
            normal_O=unit(np.average(n[mask], axis=0, weights=w[mask])).tolist(),
            strength=float(w[mask].sum())))
    coverage = float(np.mean(coverages))
    modes = (["envelope", "lateral"] if coverage >= config["envelope_coverage"]
             else ["lateral", "envelope"])
    reference = np.asarray(gripper["reference_rotation"])
    candidates = [(hand_frame @ np.diag([sign, sign, 1]) @ reference.T).tolist() for sign in (1, -1)]
    return dict(version="1.0", task_id=task,
        source=dict(dataset="ContactPose", intent="use", source_objects=sorted({s["object_name"] for s in sources}),
            sample_count=len(sources), records=sources, aggregation="equal record weight; mean joints over frames"),
        coordinates=dict(frame="tool_canonical", length_unit="m", u_zero="working_end", u_one="free_end",
            source_frame=source_frame, canonical_origin_in_source=origin.tolist(),
            canonical_axes_in_source=rotation.tolist(), handle_length_m=float(length),
            target=config["tasks"][task]["target_frame"]),
        hand_grasp=dict(closing_axis_O=x.tolist(), approach_axis_O=z.tolist()),
        contact_prior=dict(u_mean=float(mean), u_std=float(std), u_interval=interval.tolist(),
            contact_hotspots=hotspots, angular_coverage=coverage,
            coverage_definition="fraction of angular bins with measured intensity above threshold"),
        gripper=gripper,
        mapped_grasp=dict(preferred_closing_axis_O=x.tolist(), preferred_approach_axis_O=z.tolist(),
            preferred_u_RL=float(2 * mean - 1), u_sigma_RL=float(2 * std),
            mode_preference=modes, rotation_candidates_OG=candidates),
        task_context=config["tasks"][task]["context"])
