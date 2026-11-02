"""教师四类摆放同时开放，按近期完整任务成功率分配采样份额。"""

import torch


PLACEMENT_NAMES = (
    "flat",
    "lean_head_down",
    "lean_head_up",
    "inverted",
)

SGS_PARAMS = {
    "history_size": 100,
    "target_success": 0.5,
    "concentration": 1.0,
    "temperature": 2.0,
    "epsilon": 1e-4,
}


class PlacementCurriculum:
    def __init__(self, cfg, env):
        self.task = env.hammer
        bank = self.task.bank
        self.train_mask = bank["split"] == 0
        self.train_placement = bank["placement"][self.train_mask].long()
        self.scene_counts = torch.bincount(self.train_placement, minlength=4)
        self.history = [
            torch.empty(0, dtype=torch.bool, device=env.device)
            for _ in PLACEMENT_NAMES
        ]

    def __call__(
        self,
        env,
        env_ids,
        history_size,
        target_success,
        concentration,
        temperature,
        epsilon,
    ):
        completed = env.episode_length_buf[env_ids] >= 1
        old_scene = self.task.scene_id[env_ids][completed]
        placement = self.task.bank["placement"][old_scene]
        success = (
            env.action_manager.get_term("grasp")
            .values["task_success"][env_ids][completed]
            .bool()
        )

        success_rate = torch.full(
            (4,), 0.5, dtype=torch.float32, device=env.device
        )
        for k in range(4):
            self.history[k] = torch.cat(
                (self.history[k], success[placement == k])
            )[-history_size:]
            if self.history[k].numel():
                success_rate[k] = self.history[k].float().mean()

        weights = (
            (success_rate + epsilon) ** (concentration * target_success)
            * (1.0 - success_rate + epsilon)
            ** (concentration * (1.0 - target_success))
        )
        probabilities = torch.softmax(
            torch.log(weights + epsilon) / temperature, dim=0
        )

        self.task.sampling_weights.zero_()
        self.task.sampling_weights[self.train_mask] = (
            probabilities[self.train_placement]
            / self.scene_counts[self.train_placement]
        )

        return {
            **{
                f"success_rate_{name}": success_rate[k]
                for k, name in enumerate(PLACEMENT_NAMES)
            },
            **{
                f"probability_{name}": probabilities[k]
                for k, name in enumerate(PLACEMENT_NAMES)
            },
        }

    def state_dict(self):
        return {"history": self.history}

    def load_state_dict(self, state):
        self.history = state["history"]
