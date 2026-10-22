"""在 MuJoCo 场景内绘制第五阶段规划轨迹：蓝色两指尖中点、红色锤头中心。"""

import mujoco
import numpy as np


class SwingTrajectory:
    def __init__(self, viewer):
        self.viewer = viewer
        self.previous = None

    def __call__(self, grip_pos, head_pos):
        points = np.array([grip_pos, head_pos])
        if self.previous is not None:
            with self.viewer.lock():
                scene = self.viewer.user_scn
                for start, end, color in zip(
                    self.previous, points, ((0, 0, 1, 1), (1, 0, 0, 1))
                ):
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom, mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3), np.zeros(3), np.eye(3).ravel(), color,
                    )
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3, start, end)
                    scene.ngeom += 1
        self.previous = points
