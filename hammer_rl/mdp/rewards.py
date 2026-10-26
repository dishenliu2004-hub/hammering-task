"""仅五项任务评分；有效性由固定物理评价时点锁存。"""


def score(env, name):
    return env.action_manager.get_term("grasp").values[name]
