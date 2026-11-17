"""读取固定评价时点的评分缓存；Recipe 仅组合已有数值。"""


def score(env, name):
    return env.action_manager.get_term("grasp").values[name]


def recipe(env):
    values = env.action_manager.get_term("grasp").values
    return sum(term["weight"] * values[term["id"]]
               * (1.0 if term["gate"] == "none" else values[term["gate"]])
               for term in env.cfg.reward_recipe["terms"])
