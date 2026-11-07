"""教师四类摆放同时开放，按完整成功缺口和成功回合力度分配采样份额。"""

import torch


PLACEMENT_NAMES = (
    "flat",
    "lean_head_down",
    "lean_head_up",
    "inverted",
)

SGS_PARAMS = {
    "history_size": 512,
    "warmup_samples_per_class": 256,
    "update_interval": 10,
    "target_success": 0.9,
    "force_scale_n": 100.0,
    "force_mix": 0.7,
    "strength_weight": 0.5,
    "priority_power": 2.0,
    "min_probability": 0.075,
    "max_probability": 0.775,
    "probability_smoothing": 0.3,
    "min_success_samples": 32,
}


class PlacementCurriculum:
    def __init__(self, cfg, env):
        self.task = env.hammer
        bank = self.task.bank
        self.train_mask = bank["split"] == 0
        self.train_placement = bank["placement"][self.train_mask].long()
        self.scene_counts = torch.bincount(self.train_placement, minlength=4)
        self.history = [
            torch.empty((0, 3), dtype=torch.float32, device=env.device)
            for _ in PLACEMENT_NAMES
        ]
        self.probabilities = torch.full(
            (4,), 0.25, dtype=torch.float32, device=env.device
        )
        self.completed_batches = 0

    def __call__(
        self,
        env,
        env_ids,
        history_size,
        warmup_samples_per_class,
        update_interval,
        target_success,
        force_scale_n,
        force_mix,
        strength_weight,
        priority_power,
        min_probability,
        max_probability,
        probability_smoothing,
        min_success_samples,
    ):
        completed = env.episode_length_buf[env_ids] >= 1
        if completed.any():
            completed_ids = env_ids[completed]
            old_scene = self.task.scene_id[completed_ids]
            placement = self.task.bank["placement"][old_scene]
            values = env.action_manager.get_term("grasp").values
            samples = torch.stack(
                [values[name][completed_ids] for name in
                 ("task_success", "mean_force", "impact_impulse")], dim=-1
            ).float()
            for k in range(4):
                self.history[k] = torch.cat(
                    (self.history[k], samples[placement == k])
                )[-history_size:]
            self.completed_batches += 1

        success_rate = torch.full(
            (4,), 0.5, dtype=torch.float32, device=env.device
        )
        success_samples = torch.zeros(4, dtype=torch.long, device=env.device)
        strength_deficit = torch.zeros(4, dtype=torch.float32, device=env.device)
        for k in range(4):
            history = self.history[k]
            if len(history):
                success_rate[k] = history[:, 0].mean()
            successful = history[history[:, 0] == 1]
            success_samples[k] = len(successful)
            if len(successful) >= min_success_samples:
                force = successful[:, 1].median()
                impulse = successful[:, 2].median()
                d_force = force_scale_n / (force + force_scale_n)
                d_impulse = env.cfg.impulse_scale / (impulse + env.cfg.impulse_scale)
                strength_deficit[k] = force_mix * d_force + (1.0 - force_mix) * d_impulse

        if (
            completed.any()
            and all(len(history) >= warmup_samples_per_class for history in self.history)
            and self.completed_batches % update_interval == 0
        ):
            d_success = ((target_success - success_rate) / target_success).clamp(0.0, 1.0)
            priority = d_success + strength_weight * (1.0 - d_success) * strength_deficit
            score = priority.pow(priority_power)
            if score.sum() == 0:
                target = torch.full((4,), 0.25, dtype=torch.float32, device=env.device)
            else:
                raw = min_probability + (1.0 - 4.0 * min_probability) * score / score.sum()
                excess = (raw - max_probability).clamp_min(0.0)
                receivers = (raw <= max_probability).to(raw.dtype)
                target = raw - excess + excess.sum() * receivers / receivers.sum()
            self.probabilities = (
                (1.0 - probability_smoothing) * self.probabilities
                + probability_smoothing * target
            )

        self.task.sampling_weights.zero_()
        self.task.sampling_weights[self.train_mask] = (
            self.probabilities[self.train_placement]
            / self.scene_counts[self.train_placement]
        )

        return {
            **{
                f"success_rate_{name}": success_rate[k]
                for k, name in enumerate(PLACEMENT_NAMES)
            },
            **{
                f"probability_{name}": self.probabilities[k]
                for k, name in enumerate(PLACEMENT_NAMES)
            },
            **{
                f"strength_deficit_{name}": strength_deficit[k]
                for k, name in enumerate(PLACEMENT_NAMES)
            },
            **{
                f"success_samples_{name}": success_samples[k]
                for k, name in enumerate(PLACEMENT_NAMES)
            },
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
