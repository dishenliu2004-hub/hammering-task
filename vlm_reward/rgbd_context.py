"""米单位深度转换成带尺度元数据的伪彩图；不修改原始深度。"""

import base64
from io import BytesIO

from matplotlib import colormaps
import numpy as np
from PIL import Image


def image_input(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return dict(type="input_image", detail="high",
                image_url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())


def rgbd_inputs(scene, directory, depth_range_m):
    depth = np.load(directory / scene["depth"])
    measured = depth > 0
    near, far = depth_range_m
    color = (colormaps["viridis"]((depth - near) / (far - near))[..., :3] * 255).astype(np.uint8)
    color[~measured] = 0
    metadata = dict(scene, depth_visualization=dict(colormap="viridis", min_m=near,
                                                   max_m=far, invalid="black", outside_range="clipped"))
    return metadata, [image_input(Image.open(directory / scene["rgb"])), image_input(Image.fromarray(color))]
