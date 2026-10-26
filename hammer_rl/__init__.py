"""锤子初始抓姿残差任务。"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import make_env_cfg
from .rl_cfg import ppo_cfg


def register_hammer_task(*, data_file="weight/hammer_rl/reset_bank.npz", stage="teacher",
                         hammer_head_extra_mass=0.0):
    register_mjlab_task(task_id=f"Mjlab-Hammer-{stage.title()}",
        env_cfg=make_env_cfg(hammer_head_extra_mass, data_file=data_file, stage=stage),
        play_env_cfg=make_env_cfg(hammer_head_extra_mass, data_file=data_file, stage=stage, training=False),
        rl_cfg=ppo_cfg(stage))
