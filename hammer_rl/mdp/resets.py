"""原四类随机摆放和两秒物理稳定，通过 reset 事件执行。"""

import mujoco
import numpy as np
import torch


def reset_scene(env, env_ids):
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # 创建随机数生成器，不指定种子，所以每次调用可能得到不同的摆放。
    rng = np.random.default_rng()
    # position 和 local_quat 均相对于 L 形支架坐标系；四元数采用 wxyz。
    # 支架局部 Z 向上、X 沿底板长边、Y 沿宽度，顶部高度为 0.37 米。
    local_quat = np.zeros(4)
    # 等概率选择 0、1、2、3 四类初始摆放。
    scene = rng.integers(4)

    if scene == 0:
        # 横躺：圆形敲击面朝下，沿底板长边随机偏转、平移。
        # yaw 在正负 12 度内采样；绕锤柄轴翻转 180 度后绕支架局部 Z 轴偏转。
        yaw = rng.uniform(-np.pi / 15, np.pi / 15)
        mujoco.mju_euler2Quat(local_quat, np.array([yaw, np.pi, 0.]), "zxy")
        # 在底板上小范围平移；固定高度用于设置接近接触面的初始状态。
        position = np.array([rng.uniform(0.29, 0.33), rng.uniform(-0.02, 0.02), 0.1605])

    elif scene == 1:
        # 斜靠：锤头落在底板上，锤柄接触板面或顶部边缘。
        # theta 为倾斜角，范围 30～90 度；s、c 简写其正弦和余弦。
        theta = rng.uniform(np.pi / 6, np.pi / 2)
        s, c = np.sin(theta), np.cos(theta)
        # r 为锤柄半径 0.025 米加 0.0005 米接触间隙。
        r = 0.0255
        # 0.1405 是由底板高度和锤头半径确定的基准高度；
        # 0.24*sin(theta) 补偿锤头相对锤体原点的高度差。
        z = 0.1405 + 0.24 * s
        # h 表示锤柄端部圆弧中心相对支架顶部的高度差。
        # t 是辅助轴向距离，与锤柄圆柱段半长 0.05 米比较以区分接触形态。
        h = z + 0.05 * s - 0.37
        t = (r * c + 0.37 - z) / s
        if h <= 0:
            # 端部圆弧中心未高于顶部，按与竖直板内侧接触计算横向位置。
            x = 0.02 + 0.05 * c + r
        elif t > 0.05:
            # 端部圆弧靠在顶部边缘，用圆的几何关系求水平间距。
            x = 0.02 + 0.05 * c + np.sqrt(r * r - h * h)
        else:
            # 接触位置涉及倾斜柄身，按柄身与顶部边缘相切求位置。
            x = 0.02 + (r + c * (0.37 - z)) / s
        # Y 方向随机移动；yzx 指定欧拉角序列，结果存入 wxyz 四元数。
        position = np.array([x, rng.uniform(-0.05, 0.05), z])
        mujoco.mju_euler2Quat(local_quat, np.array([theta, np.pi, np.pi / 2]), "yzx")

    elif scene == 2:
        # 斜靠：锤柄落在底板上，锤头接触板面或顶部边缘。
        # 上限限制锤头相对顶部的高度差不超过接触半径，保证圆弧公式有效。
        theta = rng.uniform(np.pi / 6, np.arcsin(0.245 / 0.29))
        s, c = np.sin(theta), np.cos(theta)
        # r 为锤头半径 0.020 米加 0.0005 米接触间隙。
        r = 0.0205
        # 按锤柄落在底板上的几何关系设置锤体原点高度。
        z = 0.1455 + 0.05 * s
        # 锤头中心相对顶部的高度差；未超过顶部时靠板面，否则靠顶部边缘。
        h = z + 0.24 * s - 0.37
        x = 0.02 + 0.24 * c + (r if h <= 0 else np.sqrt(r * r - h * h))
        position = np.array([x, rng.uniform(-0.05, 0.05), z])
        mujoco.mju_euler2Quat(local_quat, np.array([theta, 0., np.pi / 2]), "yzx")

    else:
        # 倒立：锤柄朝上，锤头削出的 2cm×6cm 平面朝下并接触底板。
        mujoco.mju_axisAngle2Quat(local_quat, np.array([0., 1., 0.]), -np.pi / 2)
        # 底板顶面为0.12m；锤头削平面距锤体原点0.25722345m，留0.5mm间隙。
        position = np.array([rng.uniform(0.29, 0.33), rng.uniform(-0.02, 0.02),
                             0.12 + 0.25722345 + 0.0005])

    # 获取支架当前世界位姿；名称带有总场景 attach 设置的 support_ 前缀。
    support = data.body("support_L_support")
    # 此处修改 model 中锤子的初始参考位姿，用于场景初始化。
    # 仿真运行后读取实际锤子位姿应使用 data.body("hammer_Object")。
    hammer = model.body("hammer_Object")
    # 世界位置 = 支架世界位置 + 支架旋转矩阵 × 支架局部位置。
    hammer.pos[:] = support.xpos + support.xmat.reshape(3, 3) @ position
    # 世界姿态 = 支架世界姿态 × 锤子相对支架的局部姿态。
    mujoco.mju_mulQuat(hammer.quat, support.xquat, local_quat)
    # 初始参考位姿变更后，重新计算模型常量和当前状态的派生量。
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)

    # 沿用原 CPU mj_setConst 的结果，避免 GPU 重算引入精度差异。
    for name in ("body_pos", "body_quat", "body_subtreemass", "dof_invweight0",
                 "body_invweight0", "tendon_length0", "tendon_invweight0", "actuator_acc0",
                 "light_poscom0"):
        target = getattr(env.sim.model, name)
        target[0] = torch.as_tensor(getattr(model, name), dtype=target.dtype,
                                    device=env.device)
    for name in ("time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl",
                 "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "eq_active"):
        target = getattr(env.sim.data, name)
        target[env_ids] = torch.as_tensor(getattr(data, name), dtype=target.dtype,
                                          device=env.device)
    for _ in range(round(2 / model.opt.timestep)):
        env.sim.step()
    env.sim.data.time[env_ids] = 0
