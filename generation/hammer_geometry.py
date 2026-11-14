"""hammer.xml 的刚体局部几何；世界位姿使用米和 wxyz。"""

import numpy as np

HEAD_CENTER_LOCAL = np.array([-0.24, 0.0, 0.0])
STRIKE_FACE_CENTER_LOCAL = HEAD_CENTER_LOCAL + np.array([0.0, 0.0, 0.04])
