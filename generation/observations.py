"""冻结的初始观测；CPU 同步仅供离线准备/评估查看器使用。"""

import mujoco
import mujoco_warp


def privileged(env):
    return env.hammer.privileged


def public_seed(env):
    return env.hammer.public


def depth(env):
    return env.hammer.depth


def sync_data(env, data, env_id=0):
    task = env.hammer
    scene = int(task.scene_id[env_id])
    model = env.sim.mj_model
    model.body("hammer_Object").pos[:] = task.bank["reference_pos"][scene].cpu().numpy()
    model.body("hammer_Object").quat[:] = task.bank["reference_quat"][scene].cpu().numpy()
    h = task.handle_geom_id
    for name in ("geom_size", "geom_rbound", "geom_aabb"):
        getattr(model, name)[h] = (getattr(env.sim.model, name)[env_id, h].cpu().numpy()
                                  .reshape(getattr(model, name)[h].shape))
    if env.cfg.randomize_physics:
        model.body_mass[:] = env.sim.model.body_mass[env_id].cpu().numpy()
        model.body_inertia[:] = env.sim.model.body_inertia[env_id].cpu().numpy()
        model.geom_friction[:] = env.sim.model.geom_friction[env_id].cpu().numpy()
    mujoco.mj_setConst(model, data)
    mujoco_warp.get_data_into(data, model, env.sim.wp_data, world_id=env_id)
    mujoco.mj_forward(model, data)
