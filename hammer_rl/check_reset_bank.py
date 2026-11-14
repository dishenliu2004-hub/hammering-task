"""从已有 reset bank 继续等待，检查慢滑、倾倒及稳定窗口；不修改原数据集。

python -m hammer_rl.check_reset_bank --duration 30
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from generation.hammer_generate import (support_contacts, STABILITY_DURATION, STABILITY_INTERVAL,
    STABILITY_POSITION, STABILITY_ROTATION_DEG)
from hammer_rl.prepare import SCENE, STATE_FIELDS, MODEL_FIELDS, restore_bank, bank_simulation


DEFAULT_BANK = Path(__file__).resolve().parents[1] / "weight/hammer_rl/reset_bank_ab_256.npz"


def rotation_distance(a, b):
    """wxyz 四元数的最短转角，单位度。"""
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    qa = np.broadcast_to(a, (*shape, 4)).reshape(-1, 4)
    qb = np.broadcast_to(b, (*shape, 4)).reshape(-1, 4)
    angle = (Rotation.from_quat(qa[:, [1, 2, 3, 0]]).inv()
             * Rotation.from_quat(qb[:, [1, 2, 3, 0]])).magnitude()
    return np.rad2deg(angle).reshape(shape)


def audit(bank, indices, duration, sample_interval, device):
    sim = bank_simulation(bank, indices, device)
    dt = sim.mj_model.opt.timestep
    stride = max(1, round(sample_interval / dt))
    steps = max(1, round(duration / dt))
    body = sim.mj_model.body("hammer_Object").id
    root = sim.mj_model.body_rootid[body]
    times, positions, quats, speeds, contacts = [], [], [], [], []

    def sample(step):
        sim.forward()
        pos = sim.data.xpos[:, body]
        vel = sim.data.cvel[:, body]
        # cvel 的线速度参考点是根子树质心，换算到锤体原点。
        linear = vel[:, 3:] + torch.linalg.cross(
            vel[:, :3], pos - sim.data.subtree_com[:, root])
        speeds.append(torch.stack((linear.norm(dim=1), vel[:, :3].norm(dim=1)), dim=1)
                      .cpu().numpy())
        times.append(step * dt)
        positions.append(pos.cpu().numpy().copy())
        quats.append(sim.data.xquat[:, body].cpu().numpy().copy())
        contacts.append(support_contacts(sim))

    sample(0)
    initial_error = np.linalg.norm(positions[0] - bank["hammer_pos"][indices], axis=1)
    initial_angle = rotation_distance(quats[0], bank["hammer_quat"][indices])
    if initial_error.max() > 1e-5 or initial_angle.max() > 0.01:
        raise RuntimeError(f"恢复位姿不一致: {initial_error.max()} m / {initial_angle.max()} deg")
    print(f"Restored {len(indices)} scenes; pose error <= {initial_error.max():.3g} m / "
          f"{initial_angle.max():.3g} deg; dt={dt:g}s", flush=True)
    for step in range(1, steps + 1):
        sim.step()
        if step % stride == 0 or step == steps:
            sample(step)
            if step % max(stride, round(5 / dt)) == 0 or step == steps:
                delta = np.linalg.norm(positions[-1] - positions[0], axis=1)
                angle = rotation_distance(quats[-1], quats[0])
                print(f"+{step * dt:g}s: drift >{STABILITY_POSITION * 1000:g}mm "
                      f"or >{STABILITY_ROTATION_DEG:g}deg: "
                      f"{np.count_nonzero((delta > STABILITY_POSITION) | (angle > STABILITY_ROTATION_DEG))}/{len(indices)}; "
                      f"support contacts: {contacts[-1].sum()}/{len(indices)}", flush=True)
    wp.synchronize_device(sim.wp_device)
    return dict(time=np.array(times), position=np.stack(positions), quat=np.stack(quats),
                speed=np.stack(speeds), support_contact=np.stack(contacts),
                initial_position_error_m=initial_error, initial_rotation_error_deg=initial_angle)


def summarize(bank, indices, trace):
    times, pos, quat = (trace[k] for k in ("time", "position", "quat"))
    displacement = np.linalg.norm(pos - pos[0], axis=2) * 1000
    angle = rotation_distance(quat, quat[0])
    position_mm = STABILITY_POSITION * 1000
    moving = (displacement > position_mm) | (angle > STABILITY_ROTATION_DEG)
    window_start = max(0, np.searchsorted(times, times[-1] - .5, side="left"))
    tail_pos = np.linalg.norm(pos[window_start:] - pos[window_start], axis=2) * 1000
    tail_angle = rotation_distance(quat[window_start:], quat[window_start])
    endpoint_pos, endpoint_angle = tail_pos[-1], tail_angle[-1]
    rows = []
    for i, scene in enumerate(indices):
        crossings = np.flatnonzero(moving[:, i])
        rows.append(dict(scene_id=int(scene), split=("train", "eval")[int(bank["split"][scene])],
            source_type=int(bank["source_type"][scene]),
            max_displacement_mm=float(displacement[:, i].max()),
            max_rotation_deg=float(angle[:, i].max()),
            final_displacement_mm=float(displacement[-1, i]),
            final_rotation_deg=float(angle[-1, i]),
            first_drift_s=float(times[crossings[0]]) if len(crossings) else None,
            initial_linear_speed_mm_s=float(trace["speed"][0, i, 0] * 1000),
            initial_angular_speed_deg_s=float(np.rad2deg(trace["speed"][0, i, 1])),
            final_window_max_displacement_mm=float(tail_pos[:, i].max()),
            final_window_max_rotation_deg=float(tail_angle[:, i].max()),
            final_window_max_linear_speed_mm_s=float(trace["speed"][window_start:, i, 0].max() * 1000),
            final_window_max_angular_speed_deg_s=float(np.rad2deg(trace["speed"][window_start:, i, 1].max())),
            initial_support_contact=bool(trace["support_contact"][0, i]),
            final_support_contact=bool(trace["support_contact"][-1, i]),
            ever_drifted=bool(moving[:, i].any()),
            continuous_support_contact=bool(trace["support_contact"][:, i].all()),
            passes_stability=bool(not moving[:, i].any() and trace["support_contact"][:, i].all()),
            final_passes_original_gate=bool(endpoint_pos[i] < 5 and endpoint_angle[i] < 3
                                            and trace["support_contact"][-1, i]),
            final_quiet_1mm_1deg=bool(tail_pos[:, i].max() < 1 and tail_angle[:, i].max() < 1
                                    and trace["support_contact"][window_start:, i].all())))
    checkpoints = []
    for end in (.5, 2, 5, 10, 20, 30, 60, times[-1]):
        if end > times[-1] or any(abs(c["time_s"] - end) < 1e-8 for c in checkpoints):
            continue
        j = min(np.searchsorted(times, end), len(times) - 1)
        checkpoints.append(dict(time_s=float(times[j]),
            drifted_from_cache=int(moving[j].sum()),
            ever_drifted=int(moving[:j + 1].any(axis=0).sum()),
            without_support_contact=int((~trace["support_contact"][j]).sum())))
    summary = dict(scenes=len(indices), additional_wait_s=float(times[-1]),
        sample_interval_s=float(times[1] - times[0]),
        drift_thresholds=dict(position_mm=position_mm, rotation_deg=STABILITY_ROTATION_DEG),
        checkpoints=checkpoints, ever_drifted=sum(r["ever_drifted"] for r in rows),
        passes_stability=sum(r["passes_stability"] for r in rows),
        fails_stability=sum(not r["passes_stability"] for r in rows),
        final_window_motion_over_5mm_3deg=int(((endpoint_pos >= 5) | (endpoint_angle >= 3)).sum()),
        final_without_support_contact=int((~trace["support_contact"][-1]).sum()),
        final_fails_original_gate=sum(not r["final_passes_original_gate"] for r in rows),
        final_not_quiet_1mm_1deg=sum(not r["final_quiet_1mm_1deg"] for r in rows),
        by_source={str(s): dict(total=sum(r["source_type"] == s for r in rows),
            drifted=sum(r["source_type"] == s and r["ever_drifted"] for r in rows))
            for s in np.unique(bank["source_type"][indices]).tolist()},
        worst_scenes=sorted(rows, key=lambda r: max(r["max_displacement_mm"] / position_mm,
                                                  r["max_rotation_deg"] / STABILITY_ROTATION_DEG), reverse=True)[:10])
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--duration", type=float, default=STABILITY_DURATION, help="从缓存状态额外等待的秒数")
    parser.add_argument("--sample-interval", type=float, default=STABILITY_INTERVAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene-ids", type=int, nargs="+", help="默认检查所有场景（零起始编号）")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (not np.isfinite(args.duration) or not np.isfinite(args.sample_interval)
            or args.duration < .5 or args.sample_interval <= 0 or args.sample_interval > .5):
        parser.error("duration 须 >=0.5，sample-interval 须在 (0, 0.5] 内")
    digest = hashlib.sha256(args.data_file.read_bytes()).hexdigest()
    with np.load(args.data_file) as source:
        bank = {k: source[k] for k in (*STATE_FIELDS, "handle_size", "reference_pos",
                "reference_quat", "hammer_pos", "hammer_quat", "split", "source_type",
                "hammer_head_extra_mass")}
    indices = np.arange(len(bank["split"])) if args.scene_ids is None else np.array(args.scene_ids)
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(bank["split"])):
        parser.error("scene-ids 超出数据集范围")
    trace = audit(bank, indices, args.duration, args.sample_interval, args.device)
    rows, summary = summarize(bank, indices, trace)
    summary.update(data_file=str(args.data_file.resolve()), sha256=digest,
                   backend="mjlab / MuJoCo Warp", mujoco_version=mujoco.__version__,
                   scene_file=str(SCENE), physics_randomization=False)
    if hashlib.sha256(args.data_file.read_bytes()).hexdigest() != digest:
        raise RuntimeError("检查期间数据集文件发生变化")
    output = args.output_dir or args.data_file.parent / f"{args.data_file.stem}_stability_{args.duration:g}s"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scenes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "traces.npz", scene_ids=indices, **trace)
    print(json.dumps({k: v for k, v in summary.items() if k != "worst_scenes"}, indent=2))
    print(f"Reports: {output.resolve()}\nOriginal bank unchanged: SHA256 {digest}")


if __name__ == "__main__":
    main()
