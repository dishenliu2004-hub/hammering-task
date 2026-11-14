"""教师按训练场景的近期失败率分配 80% 采样概率，其余 20% 均匀分配。"""

import torch


SGS_PARAMS = {
    "history_size": 100,
    "update_interval": 10,
    "target_success": 0.0,
    "concentration": 1.0,
    "temperature": 1.0,
    "epsilon": 1e-4,
    "uniform_mix": 0.2,
}


class SceneSgsCurriculum:
    def __init__(self, cfg, env):
        self.task = env.hammer
        self.train_mask = self.task.bank["split"] == 0
        self.train_ids = self.train_mask.nonzero().flatten().tolist()
        # 历史按缓存行号索引；评估场景的历史始终为空。
        self.history = [
            torch.empty(0, dtype=torch.float32, device=env.device)
            for _ in self.task.bank["split"]
        ]
        # 概率顺序对应 train_ids 中递增的缓存行号。
        self.probabilities = torch.full(
            (len(self.train_ids),), 1.0 / len(self.train_ids),
            dtype=torch.float32, device=env.device,
        )
        self.completed_batches = 0

    def __call__(
        self,
        env,
        env_ids,
        history_size,
        update_interval,
        target_success,
        concentration,
        temperature,
        epsilon,
        uniform_mix,
    ):
        completed = env.episode_length_buf[env_ids] >= 1
        if completed.any():
            completed_ids = env_ids[completed]
            # 课程在 reset 覆盖 scene_id 和动作评分前读取旧回合。
            old_scene = self.task.scene_id[completed_ids]
            success = env.action_manager.get_term("grasp").values["task_success"][completed_ids].float()
            for scene in old_scene.unique().tolist():
                if self.train_mask[scene]:
                    # 同场景的并行完成回合全部追加，再保留最近 H 条。
                    self.history[scene] = torch.cat(
                        (self.history[scene], success[old_scene == scene])
                    )[-history_size:]
            self.completed_batches += 1

        success_rate = torch.full_like(self.probabilities, 0.5)
        observed = torch.zeros_like(self.probabilities, dtype=torch.bool)
        for i, scene in enumerate(self.train_ids):
            if len(self.history[scene]):
                success_rate[i] = self.history[scene].mean()
                observed[i] = True

        if completed.any() and self.completed_batches % update_interval == 0:
            weight = (success_rate + epsilon).pow(concentration * target_success) * (
                1.0 - success_rate + epsilon
            ).pow(concentration * (1.0 - target_success))
            logits = (weight.clamp_min(epsilon) + epsilon).log() / temperature
            self.probabilities = torch.softmax(logits, dim=0)
            self.probabilities = (
                (1.0 - uniform_mix) * self.probabilities
                + uniform_mix / len(self.train_ids)
            )

        self.task.sampling_weights.zero_()
        self.task.sampling_weights[self.train_mask] = self.probabilities

        return {
            "sgs_mean_success": success_rate[observed].mean() if observed.any()
                                else success_rate.new_zeros(()),
            "sgs_mean_probability": self.probabilities.mean(),
            "sgs_min_probability": self.probabilities.min(),
            "sgs_max_probability": self.probabilities.max(),
            "sgs_observed_scenes": observed.sum(),
        }

    def state_dict(self):
        return {
            "history": self.history,
            "probabilities": self.probabilities,
            "completed_batches": self.completed_batches,
        }

    def load_state_dict(self, state):
        self.history = state["history"]
        self.probabilities = state["probabilities"]
        self.completed_batches = state["completed_batches"]
