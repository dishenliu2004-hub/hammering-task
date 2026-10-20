
"""
运行指令：
/home/ubuntu20/anaconda3/envs/hammer/bin/python /home/ubuntu20/CodeMujoco/hammering-task/visualization/camera_rgb_pointcloud.py
追加 --noise 使用含噪点云，默认无噪。
"""

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.hammer_generate import load_random_scene, sync_data, write_data
from visualization.noise.adapter import add_depth_noise


class CameraRGBPointCloud:
    """采集并展示初始场景；接口本身不推进物理仿真。"""

    def __init__(self, model, *, noise=False):
        self.noise = noise
        self.width, self.height = 1280, 960
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
        self.camera = model.camera("fixed").id
        self.focal = self.height / (2 * np.tan(np.deg2rad(model.cam_fovy[self.camera]) / 2))
        self.v, self.u = np.mgrid[0:self.height:4, 0:self.width:4]
        self.far = model.vis.map.zfar * model.stat.extent
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

        plt.ion()
        self.rgb_figure = plt.figure(figsize=(10, 8))
        self.rgb_figure.canvas.manager.set_window_title("D435i RGB")
        rgb_axes = self.rgb_figure.add_subplot(111)
        rgb_axes.set_title("D435i RGB")
        rgb_axes.axis("off")
        self.image = rgb_axes.imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))

        self.cloud_figure = plt.figure(figsize=(9, 7))
        self.cloud_figure.canvas.manager.set_window_title("D435i Point Cloud")
        cloud_axes = self.cloud_figure.add_subplot(111, projection="3d")
        cloud_axes.set(title="D435i Point Cloud", xlim=(-0.5, 0.5),
                       ylim=(-0.7, 0.5), zlim=(0, 0.8))
        cloud_axes.set_axis_off()
        cloud_axes.set_box_aspect((1, 1.2, 0.8))
        self.cloud = cloud_axes.scatter([], [], [], s=1, depthshade=False, edgecolors="none")
        self.rgb_figure.tight_layout()
        self.cloud_figure.tight_layout()

        def zoom(event):
            cloud_axes._scale_axis_limits(0.9 ** event.step, 0.9 ** event.step, 0.9 ** event.step)
            self.cloud_figure.canvas.draw_idle()

        self.cloud_figure.canvas.mpl_connect("scroll_event", zoom)

    def capture(self, data):
        self.renderer.update_scene(data, camera=self.camera)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()[::4, ::4]
        if self.noise:
            depth = add_depth_noise(depth, focal_length=self.focal / 4)
        visible = (depth > 0) & (depth < self.far)

        # 光学相机坐标：X 向右、Y 向下、Z 向前，直接用于 AnyGrasp 推理。
        points = np.stack(((self.u - (self.width - 1) / 2) * depth / self.focal,
                           (self.v - (self.height - 1) / 2) * depth / self.focal,
                           depth), axis=-1)[visible]
        camera_rot = data.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1, -1, -1])
        camera_pos = data.cam_xpos[self.camera].copy()
        return {
            "rgb": rgb,
            "points": points.astype(np.float32),
            "world_points": points @ camera_rot.T + camera_pos,
            "colors": rgb[::4, ::4][visible].astype(np.float32) / 255.0,
            "camera_rot": camera_rot,
            "camera_pos": camera_pos,
        }

    def update(self, frame):
        self.image.set_data(frame["rgb"])
        points = frame["world_points"]
        self.cloud.set_offsets(points[:, :2])
        self.cloud.set_3d_properties(points[:, 2], "z")
        self.cloud.set_facecolor(frame["colors"])
        self.rgb_figure.canvas.draw_idle()
        self.cloud_figure.canvas.draw_idle()
        self.rgb_figure.canvas.flush_events()

    def close(self):
        self.renderer.close()
        plt.close(self.rgb_figure)
        plt.close(self.cloud_figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", action="store_true", help="为相机深度加噪（默认关闭）")
    args = parser.parse_args()
    log_config = mujoco.MjLogConfig.get()
    log_config.logto_file = False
    log_config.set()
    model, data, simulation = load_random_scene()
    camera = CameraRGBPointCloud(model, noise=args.noise)
    try:
        camera.update(camera.capture(data))
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.monotonic() - data.time
            while (viewer.is_running() and plt.fignum_exists(camera.rgb_figure.number)
                   and plt.fignum_exists(camera.cloud_figure.number)):
                frame_start = time.monotonic()
                with viewer.lock():
                    steps = int((frame_start - start_time - data.time) / model.opt.timestep)
                    write_data(simulation, data)
                    for _ in range(steps):
                        simulation.step()
                    sync_data(simulation, data)
                viewer.sync()
                camera.rgb_figure.canvas.flush_events()
                time.sleep(max(0.001, 0.1 - (time.monotonic() - frame_start)))
    finally:
        camera.close()


if __name__ == "__main__":
    # 直接运行脚本时执行主函数。
    main()
