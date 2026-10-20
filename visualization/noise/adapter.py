"""适配 D435i 深度噪声；以约 1 m 工作距离为目标，参数未经实机标定。

来源：https://github.com/leggedrobotics/sru-depth-pretraining
提交：2d153dcd9f0fadd6d1f3f00d197c6cdf6fda0e97
"""

import torch

from visualization.noise.depth_noise import DepthNoise


def add_depth_noise(depth, *, focal_length, filter_size=3):
    """为米制二维深度数组加噪，返回同形状的 NumPy 数组。"""
    noise = DepthNoise(
        focal_length=focal_length, baseline=0.05,
        min_depth=0.3, max_depth=3.0, filter_size=filter_size,
    )
    with torch.inference_mode():
        return noise(torch.as_tensor(depth, dtype=torch.float32)[None, None])[0, 0].numpy()