"""原生 actor/critic 加载与单次 6D 完整抓取生成。"""

import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.utils import resolve_callable

from .mdp.actions import direct_grasp_pose
from .mdp.resets import public_observation


def load_models(checkpoint, device="cuda:0", *, with_critic=True):
    """按旁置 run_config 构建现有 MLP/CNN；归一化统计随 state_dict 加载。"""
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint.parent / "run_config.json").read_text())
    weights = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = saved["rl"]
    obs = TensorDict({"privileged": torch.zeros(1, 66, device=device),
        "public_seed": torch.zeros(1, 34, device=device),
        "depth": torch.zeros(1, 1, 120, 160, device=device)}, batch_size=[1])

    def build(model_cfg, groups, name, output_dim):
        kwargs = {key: model_cfg[key] for key in
                  ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")}
        if model_cfg["class_name"] == "CNNModel":
            kwargs["cnn_cfg"] = model_cfg["cnn_cfg"]
        model = resolve_callable(model_cfg["class_name"])(obs, groups, name, output_dim, **kwargs)
        return model.to(device).eval()

    distill = saved["stage"] == "distill"
    actor_cfg = cfg["student"] if distill else cfg["actor"]
    groups = {"actor": cfg["obs_groups"]["student"]} if distill else cfg["obs_groups"]
    actor = build(actor_cfg, groups, "actor", 6)
    actor.load_state_dict(weights["student_state_dict" if distill else "actor_state_dict"])
    critic = None
    if with_critic:
        critic = build(cfg["critic"], groups, "critic", 1)
        critic.load_state_dict(weights["critic_state_dict"])
    return actor, critic


@torch.no_grad()
def select_grasp(actor, depth, scene):
    """学生在线入口；scene 提供公开观测及工具位姿、锤柄尺寸和 TCP 偏移。"""
    device = next(actor.parameters()).device
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    fields = {key: tensor(scene[key]).unsqueeze(0) for key in
              ("finger_pos", "finger_vel", "camera_pos", "camera_quat", "intrinsics",
               "target_pos", "target_quat")}
    public = public_observation(**fields)
    observations = TensorDict({"public_seed": public,
        "depth": tensor(depth).unsqueeze(0)}, batch_size=[1])
    action = actor(observations)
    pos, quat, pregrasp = direct_grasp_pose(
        tensor(scene["hammer_pos"]).unsqueeze(0),
        tensor(scene["hammer_quat"]).unsqueeze(0),
        tensor(scene["handle_size"]).unsqueeze(0),
        tensor(scene["tcp_offset"]).unsqueeze(0), action)
    return dict(action=action, mount_pos=pos, mount_quat=quat,
                pregrasp_pos=pregrasp, gripper_open_ctrl=0.0)
