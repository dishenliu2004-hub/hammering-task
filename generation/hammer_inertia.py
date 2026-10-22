"""普通流程与强化学习共用的锤头增重计算。"""

import mujoco
import numpy as np


def add_head_mass(spec: mujoco.MjSpec, extra_mass: float) -> None:
    """在当前锤头内部均匀增加 extra_mass 千克，合并到锤子的惯性参数。"""
    if extra_mass == 0:
        return

    hammer = spec.body("hammer_Object")
    head = spec.geom("hammer_head")
    mesh = spec.mesh(head.meshname)

    # 由 MuJoCo 根据现有网格计算新增部分的质心和惯量。
    extra = mujoco.MjSpec()
    extra.add_mesh(name="head", uservert=mesh.uservert, userface=mesh.userface,
                   scale=mesh.scale, inertia=mesh.inertia)
    body = extra.worldbody.add_body()
    body.add_freejoint()
    body.add_geom(type=head.type, meshname="head", pos=head.pos, quat=head.quat,
                  mass=extra_mass)
    model = extra.compile()
    added = model.body(1)

    total_mass = hammer.mass + extra_mass
    center = (hammer.mass * hammer.ipos + extra_mass * added.ipos) / total_mass
    inertia = np.zeros((3, 3))
    for mass, pos, quat, diagonal in (
        (hammer.mass, hammer.ipos, hammer.iquat, hammer.inertia),
        (extra_mass, added.ipos, added.iquat, added.inertia),
    ):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
        rotation = rotation.reshape(3, 3)
        offset = pos - center
        inertia += (rotation @ np.diag(diagonal) @ rotation.T
                    + mass * (offset @ offset * np.eye(3) - np.outer(offset, offset)))

    hammer.mass = total_mass
    hammer.ipos = center
    hammer.inertia = [0, 0, 0]
    hammer.iquat = [1, 0, 0, 0]
    hammer.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
