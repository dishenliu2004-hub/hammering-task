"""由真实评估、阶段关键帧和上轮对照修订 hammer RewardRecipe。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .api_client import design
from .reward_schema import AVAILABLE_TERMS, revision_schema
from .rgbd_context import image_input

ROOT = Path(__file__).resolve().parent
PHYSICAL_METRICS = (
    "capture", "lift_height", "lift_progress", "lift_success", "hold_fraction", "grasp_before_swing",
    "seat_closure", "seat_lift", "seat_before_swing", "translation_drift", "rotation_drift",
    "valid_strike", "hit_count", "hit_progress", "recovery_hold", "drop", "task_success",
    "strike_force_1", "strike_force_2", "strike_force_3", "peak_force", "mean_force", "impact_impulse",
)


def physical_metrics(evaluation):
    return {key: float(evaluation[key].mean()) for key in PHYSICAL_METRICS}


def build_feedback(recipe, mapping, scene, evaluation, history, directory):
    metrics = physical_metrics(evaluation)
    closed = evaluation["capture"] == 1
    lifted = evaluation["lift_success"] == 1
    held = evaluation["grasp_before_swing"] == 1
    dropped = evaluation["drop"] == 1
    hits = evaluation["hit_count"]
    forces = np.stack([evaluation[f"strike_force_{i}"] for i in (1, 2, 3)], axis=-1)
    # 参考力用于解释力度分布，与任务成功的判定独立。
    force_term = next((t for t in recipe["terms"] if t.get("formula") == "force_mean_min"), None)
    ref_n = force_term["params"][1] if force_term else 10.0
    stages = dict(
        closure_failed=int((~closed & ~lifted).sum()),
        lift_failed_after_closure=int((closed & ~lifted).sum()),
        lifted_missing_at_swing_with_drop=int((lifted & ~held & dropped).sum()),
        lifted_missing_at_swing_without_drop=int((lifted & ~held & ~dropped).sum()),
        held_but_incomplete_hits=int((held & (hits < 3)).sum()),
        three_hits_lost_before_completion=int(((hits == 3) & (evaluation["task_success"] == 0)).sum()),
    )
    feedback = dict(episodes=len(hits), metrics=metrics, stage_failure_counts=stages,
                    hit_force_reference_n=ref_n, positive_strikes=int((forces > 0).sum()),
                    positive_strikes_below_reference=int(((forces > 0) & (forces < ref_n)).sum()))
    previous = None
    if history:
        last = history[-1]
        old = json.loads(Path(last["old_recipe"]).read_text())
        proposed = json.loads(Path(last["new_recipe"]).read_text())
        previous = dict(selected_branch=last["selected_branch"], previous_recipe_id=old["recipe_id"],
                        previous_terms=old["terms"], proposed_recipe_id=proposed["recipe_id"],
                        proposed_terms=proposed["terms"], old_metrics=last["old_metrics"],
                        new_metrics=last["new_metrics"], failure_mode=last["failure_mode"],
                        change_since_previous_feedback={
                            key: value - last["source_metrics"][key] for key, value in metrics.items()})
    frames = json.loads((directory / "index.json").read_text())
    context = dict(task="一次决定 6D 抓姿，固定轨迹完成三次有效敲击并保持工具。",
                   current_recipe=recipe, available_terms=AVAILABLE_TERMS, mapping_prior=mapping["mapped_grasp"],
                   initial_scene=scene, feedback=feedback, previous_round=previous, frames_in_image_order=frames)
    images = []
    for frame in frames:
        with Image.open(directory / frame["file"]) as image:
            images.append(image_input(image))
    return context, images


def prepare_request(recipe_path, mapping_path, scene_path, evaluation_path, history_path, directory):
    with np.load(evaluation_path) as data:
        context, images = build_feedback(
            json.loads(recipe_path.read_text()), json.loads(mapping_path.read_text()), json.loads(scene_path.read_text()),
            data, [json.loads(line) for line in history_path.read_text().splitlines()], directory,
        )
    (directory / "summary.json").write_text(json.dumps(context["feedback"], indent=2) + "\n")
    return context, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "mapping", "scene-context", "evaluation", "history", "feedback-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    context, images = prepare_request(args.recipe, args.mapping, args.scene_context,
                                      args.evaluation, args.history, args.feedback_dir)
    recipe = context["current_recipe"]
    config = yaml.safe_load((ROOT / "configs/vlm.yaml").read_text())
    result, response_id = design(config, (ROOT / "prompts/reward_revision.md").read_text(),
                                 context, images, schema=revision_schema())
    revised = recipe.copy()  # 保留初始来源元数据。
    revised.update(result, version="2.0", recipe_id=args.output.stem,
                   revision=dict(parent_recipe_id=recipe["recipe_id"], model=config["model"], response_id=response_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(revised, ensure_ascii=False, indent=2) + "\n")
    print(f"recipe: {args.output.resolve()}")


if __name__ == "__main__":
    main()
