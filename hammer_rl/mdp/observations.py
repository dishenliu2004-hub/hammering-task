"""同步原感知、轨迹和查看器所需状态；RL 观测定义待填充。"""

import mujoco_warp


def sync_data(env, data):
    # env.reset() 和 env.step() 已刷新派生量，直接读取当前 GPU 状态。
    mujoco_warp.get_data_into(data, env.sim.mj_model, env.sim.wp_data)
