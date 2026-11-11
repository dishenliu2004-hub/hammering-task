"""原生 9D actor/critic 加载、候选评分与 A 抓取、B 调姿动作输出。"""

import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.utils import resolve_callable

from generation.grasp_hammer import candidate_to_mount
from .mdp.actions import GraspResidualActionCfg, refine_pose
from .mdp.resets import public_observation, candidate_observations, bind_candidates


def load_models(checkpoint, device="cuda:0", *, with_critic=True):
    """按旁置 run_config 构建现有 MLP/CNN；归一化统计随 state_dict 加载。"""
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint.parent / "run_config.json").read_text())
    weights = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = saved["rl"]
    obs = TensorDict({"privileged": torch.zeros(1, 94, device=device),
        "public_seed": torch.zeros(1, 57, device=device),
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
    actor = build(actor_cfg, groups, "actor", 9)
    actor.load_state_dict(weights["student_state_dict" if distill else "actor_state_dict"])
    critic = None
    if with_critic:
        critic = build(cfg["critic"], groups, "critic", 1)
        critic.load_state_dict(weights["critic_state_dict"])
    return actor, critic


@torch.no_grad()
def select_and_refine(actor, critic, observations, offsets, action_cfg):
    """offsets 分组变长候选；同分沿用 torch.argmax 的首个结果。"""
    scores = critic(observations).squeeze(-1)
    index = torch.stack([scores[start:end].argmax() + start
                         for start, end in zip(offsets[:-1], offsets[1:])])
    public = observations["public_seed"][index]
    action = actor(observations[index])
    pos, quat, pregrasp = refine_pose(public[:, :3], public[:, 3:7], action[:, :6], action_cfg)
    return dict(index=index, candidate_index=index - offsets[:-1], scores=scores,
        score=scores[index], action=action, mount_pos=pos, mount_quat=quat,
        pregrasp_pos=pregrasp, width=public[:, 7], gripper_open_ctrl=public[:, 8])


@torch.no_grad()
def select_candidates(env, actor, critic, env_ids):
    """评估/回放：从当前场景的全部缓存候选中选择，然后绑定并刷新仿真。"""
    task = env.hammer
    scene = task.scene_id[env_ids]
    starts = task.bank["offsets"][scene]
    counts = task.bank["offsets"][scene + 1] - starts
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    repeated = env_ids.repeat_interleave(counts)
    candidate = (starts.repeat_interleave(counts) + torch.arange(offsets[-1], device=env.device)
                 - offsets[:-1].repeat_interleave(counts))
    obs = candidate_observations(env, repeated, candidate)
    result = select_and_refine(actor, critic, obs, offsets, env.cfg.actions["grasp"])
    result["candidate_id"] = candidate[result["index"]]
    bind_candidates(env, env_ids, result["candidate_id"])
    env.sim.forward()
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True, env_ids=env_ids)
    return result


@torch.no_grad()
def select_grasp(actor, critic, depth, scene, grasps):
    """学生在线入口。depth 为 (1,120,160)，scene 为未批量的公开字段，候选为世界坐标。"""
    device = next(actor.parameters()).device
    poses = [candidate_to_mount(grasps, i, extra_depth=0.03) for i in range(len(grasps["scores"]))]
    count = len(poses)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    seed_pos = torch.stack([tensor(pose["mount_pos"]) for pose in poses])
    seed_quat = torch.stack([tensor(pose["mount_quat"]) for pose in poses])
    open_ctrl = tensor([pose["gripper_open_ctrl"] for pose in poses])
    fields = {key: tensor(scene[key]).unsqueeze(0).expand(count, -1) for key in
              ("finger_pos", "finger_vel", "camera_pos", "camera_quat", "intrinsics",
               "target_pos", "target_quat")}
    public = public_observation(seed_pos, seed_quat, tensor(grasps["widths"]), open_ctrl, **fields)
    b_target = torch.cat((tensor(scene["b_target_pos"]), tensor(scene["b_target_quat"])))
    public = torch.cat((public, b_target.expand(count, -1)), dim=-1)
    observations = TensorDict({"public_seed": public,
        "depth": tensor(depth).unsqueeze(0).expand(count, -1, -1, -1)}, batch_size=[count])
    return select_and_refine(actor, critic, observations,
        torch.tensor([0, count], device=device), GraspResidualActionCfg(entity_name="gripper_mount"))
