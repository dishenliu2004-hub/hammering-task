"""完整场景表面点云与 AnyGrasp 抓姿展示。

/home/ubuntu20/anaconda3/envs/hammer/bin/python visualization/full_scene_pointcloud.py --noise
追加 --noise 使用含噪点云，默认无噪。
"""

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualization.noise.adapter import add_depth_noise


class FullScenePointCloud:
    """缓存几何体局部表面采样，按当前仿真位姿提供完整点云。"""

    def __init__(self, model, spacing=0.004, *, noise=False):
        self.camera = model.camera("fixed").id
        self.noise = noise
        # 与相机路径的 960/4 行深度图使用相同焦距。
        self.focal = (960 / 4) / (2 * np.tan(np.deg2rad(model.cam_fovy[self.camera]) / 2))
        self.samples = []
        colors = []
        rng = np.random.default_rng(0)
        for geom_id in range(model.ngeom):
            if (model.geom_group[geom_id] == 3
                    or model.body(model.geom_bodyid[geom_id]).name.startswith(("gripper_", "camera_"))):
                continue
            material = model.geom_matid[geom_id]
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom_id]
            if rgba[3] == 0:
                continue
            kind = model.geom_type[geom_id]
            size = model.geom_size[geom_id]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                mesh = trimesh.creation.box(extents=2 * size)
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=64)
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
            else:
                mesh_id = model.geom_dataid[geom_id]
                vertex_start = model.mesh_vertadr[mesh_id]
                face_start = model.mesh_faceadr[mesh_id]
                mesh = trimesh.Trimesh(
                    vertices=model.mesh_vert[vertex_start:vertex_start + model.mesh_vertnum[mesh_id]],
                    faces=model.mesh_face[face_start:face_start + model.mesh_facenum[mesh_id]],
                    process=False,
                )
            count = int(np.ceil(mesh.area / spacing**2))
            points, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
            self.samples.append((geom_id, points))
            colors.append(np.tile(rgba[:3], (count, 1)))
        self.colors = np.concatenate(colors).astype(np.float32)

    def capture(self, data):
        world_points = np.concatenate([
            points @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            for geom_id, points in self.samples
        ])
        camera_rot = data.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1, -1, -1])
        camera_pos = data.cam_xpos[self.camera].copy()
        points = ((world_points - camera_pos) @ camera_rot).astype(np.float32)
        colors = self.colors
        if self.noise:
            # 无序采样点不具有像素邻域；仅保留逐点缺失和视差量化。
            depth = add_depth_noise(points[:, 2][None, :], focal_length=self.focal,
                                    filter_size=1)[0]
            visible = depth > 0
            points = points[visible] * (depth[visible] / points[visible, 2])[:, None]
            world_points = points @ camera_rot.T + camera_pos
            colors = colors[visible]
        above_table = world_points[:, 2] >= 0
        return {
            "points": points[above_table],
            "world_points": world_points[above_table],
            "colors": colors[above_table],
            "camera_rot": camera_rot,
            "camera_pos": camera_pos,
        }


def main():
    import matplotlib.pyplot as plt

    from generation.grasp_hammer import generate_grasps
    from generation.hammer_generate import load_random_scene
    from visualization.anygrasp_grasp_visualization import show_grasps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", action="store_true", help="为完整场景点云加噪（默认关闭）")
    args = parser.parse_args()
    log_config = mujoco.MjLogConfig.get()
    log_config.logto_file = False
    log_config.set()
    model, data, _ = load_random_scene()
    frame = FullScenePointCloud(model, noise=args.noise).capture(data)
    grasps, _ = generate_grasps(model, data, frame, full_scene=True)
    figure = show_grasps(frame, grasps)
    low, high = frame["world_points"].min(axis=0), frame["world_points"].max(axis=0)
    figure.axes[0].set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    figure.axes[0].set_box_aspect(high - low)
    plt.show()


if __name__ == "__main__":
    main()
