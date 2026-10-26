"""仅改变缓存摆放/倾角的采样范围；里程碑由 run 配置固定。"""


def placement_curriculum(env, env_ids):
    del env_ids
    task = env.hammer
    bank = task.bank
    step = env.common_step_counter
    first, second = env.cfg.curriculum_milestones

    inverted = bank["placement"] == 3
    leaning = (bank["placement"] == 1) | (bank["placement"] == 2)
    lying = bank["placement"] == 0

    if step < first:
        allowed = inverted
        phase = 0
    elif step < second:
        allowed = inverted | leaning
        phase = 1
    else:
        allowed = inverted | leaning | lying
        phase = 2

    task.sampling_weights = (
        allowed & (bank["split"] == 0)
    ).float()

    return {"phase": phase}
