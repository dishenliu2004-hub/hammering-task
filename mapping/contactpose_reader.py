"""复用官方加载器；root 是带 data/ 的 ContactPose 官方代码目录。"""

import sys

import numpy as np
import open3d as o3d


def read_records(root, object_name, participants):
    sys.path.insert(0, str(root))
    from utilities.dataset import ContactPose
    from utilities.misc import texture_proc

    for participant in participants:
        cp = ContactPose(participant, "use", object_name, load_mano=False)
        mesh = o3d.io.read_triangle_mesh(cp.contactmap_filename)
        mesh.compute_vertex_normals()
        raw = np.asarray(mesh.vertex_colors)[:, 0].copy()
        # 有效手是数据的语义分支；每条记录只统计一次，时间帧取均值。
        hands = [np.mean([cp.hand_joints(i)[side] for i in range(len(cp))], axis=0)
                 for side, hand in enumerate(cp.hand_joints(0)) if hand is not None]
        yield dict(participant_id=participant, object_name=object_name, intent="use",
            hands=hands, vertices=np.asarray(mesh.vertices), normals=np.asarray(mesh.vertex_normals),
            intensity=texture_proc(raw.copy()), measured=raw > 0,
            annotation=cp.annotation_filename, contact_map=cp.contactmap_filename,
            frame_count=len(cp))
