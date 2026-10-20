"""锤击仿真的 manager-based 环境；训练配置确定后显式注册任务。"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import make_env_cfg


def register_hammer_task(rl_cfg: RslRlOnPolicyRunnerCfg) -> None:
    register_mjlab_task(
        task_id="Mjlab-Hammer-Omnipicker",
        env_cfg=make_env_cfg(),
        play_env_cfg=make_env_cfg(),
        rl_cfg=rl_cfg,
    )
