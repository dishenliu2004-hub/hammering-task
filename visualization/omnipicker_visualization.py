"""Omnipicker 两侧 joint3 的角度范围与限位可视化。

运行：
/home/ubuntu20/anaconda3/envs/hammer/bin/python visualization/omnipicker_visualization.py

蓝线为角度下限，绿线为上限，灰色圆弧为允许区间，红线为当前角度。
右侧 Control 中调整原有 gripper_actuator：0 张开，1 闭合。
Joint 面板查看实际角度；空格暂停或继续。
标记仅展示局部关节角度区间，不施加额外力矩或改变物理约束。
"""

from math import cos, degrees, sin
from pathlib import Path
import sys

import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.hammer_generate import create_simulation, launch_viewer


def add_joint3_visualization(spec, side):
    joint = spec.joint(f"hand_{side}3_joint")
    parent = spec.body(f"hand_{side}2_Link")
    body = spec.body(f"hand_{side}3_Link")
    lower, upper = joint.range
    origin = body.pos + [0, 0, 0.03]
    radius = 0.035

    # 两个 joint3 的轴均为局部 -Z；区间标记随父连杆运动。
    def endpoint(angle):
        return origin + [radius * cos(angle), -radius * sin(angle), 0]

    def line(target, name, start, end, color, thickness=0.0004):
        target.add_site(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        fromto=[*start, *end], size=[thickness, 0, 0], rgba=color)

    for label, angle, color in (("min", lower, [0.1, 0.4, 1, 1]),
                                ("max", upper, [0.1, 1, 0.2, 1])):
        line(parent, f"{side}3_{label}", origin, endpoint(angle), color)
    for i in range(16):
        start = lower + (upper - lower) * i / 16
        end = lower + (upper - lower) * (i + 1) / 16
        line(parent, f"{side}3_arc_{i}", endpoint(start), endpoint(end),
             [0.7, 0.7, 0.7, 1])
    line(body, f"{side}3_current", [0, 0, 0.031], [radius, 0, 0.031],
         [1, 0.1, 0.1, 1], thickness=0.0007)

    print(f"{side}3: [{lower:.9f}, {upper:.9f}] rad "
          f"([{degrees(lower):.2f}, {degrees(upper):.2f}] deg)")


def main():
    log_config = mujoco.MjLogConfig.get()
    log_config.logto_file = False
    log_config.set()

    model_path = (Path(__file__).resolve().parents[1]
                  / "assets" / "grippers" / "omnipicker" / "ominipicker.xml")
    spec = mujoco.MjSpec.from_file(str(model_path))
    for side in ("narrow", "wide"):
        add_joint3_visualization(spec, side)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    # 仅调整查看器的初始取景。
    model.stat.center[:] = [0, 0, 0.09]
    model.stat.extent = 0.15
    model.vis.global_.azimuth = 180
    model.vis.global_.elevation = 0
    simulation = create_simulation(model, data)
    launch_viewer(model, data, simulation)


if __name__ == "__main__":
    main()
