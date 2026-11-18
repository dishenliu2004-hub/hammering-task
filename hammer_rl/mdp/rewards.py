"""读取固定评价时点的评分缓存；Recipe 仅组合已有数值。"""

import torch


def score(env, name):
    return env.action_manager.get_term("grasp").values[name]


def compute_term(values, term, mapping_prior):
    formula, params = term["formula"], term["params"]
    if formula == "identity":
        return values[term["id"]]
    if formula == "power":
        return values[term["id"]].clamp(0, 1).pow(params[0])
    if formula == "prior_gaussian":
        sigma = mapping_prior["u_sigma_RL"] * params[0]
        return torch.exp(-0.5 * ((values["grasp_u_rl"] - mapping_prior["u_mean_RL"]) / sigma).square())
    if formula == "prior_axis_power":
        return values["grasp_axis_cos2"].pow(params[0])
    if formula == "force_mean_min":
        alpha, force_ref_n = params
        forces = torch.stack([values[f"strike_force_{i}"] for i in (1, 2, 3)], dim=-1)
        scaled = torch.log1p(forces / force_ref_n)
        return alpha * scaled.mean(-1) + (1 - alpha) * scaled.amin(-1)
    if formula == "hold_drift":
        pos_scale, rot_scale = params
        return values["hold_fraction"] * torch.exp(
            -values["translation_drift"] / pos_scale - values["rotation_drift"] / rot_scale)


def recipe(env):
    values = env.action_manager.get_term("grasp").values
    cfg = env.cfg.reward_recipe
    terms = cfg["terms"]
    if cfg.get("version") == "2.0":
        scores = (compute_term(values, term, cfg["mapping_prior"]) for term in terms)
    else:
        scores = (values[term["id"]] for term in terms)
    return sum(term["weight"] * value * (1.0 if term["gate"] == "none" else values[term["gate"]])
               for term, value in zip(terms, scores))
