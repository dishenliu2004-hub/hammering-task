"""离线读取 MappingCode、场景 RGB-D 和任务文字，导出 RewardRecipe。"""

import argparse
import json
from pathlib import Path

import yaml

from .api_client import design
from .reward_schema import AVAILABLE_TERMS
from .rgbd_context import rgbd_inputs

ROOT = Path(__file__).resolve().parent


def assemble_recipe(design_result, mapping, *, recipe_id, mapping_ref, profile_id):
    mapped = mapping["mapped_grasp"]
    return dict(version="1.0", recipe_id=recipe_id, task_id=mapping["task_id"],
        mapping_ref=mapping_ref, profile_id=profile_id, **design_result,
        mapping_prior=dict(u_mean_RL=mapped["preferred_u_RL"], u_sigma_RL=mapped["u_sigma_RL"],
                           closing_axis_O=mapped["preferred_closing_axis_O"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 仅锤击已实现真实物理算子。扫帚/刀映射可离线导出，但尚无对应奖励任务。
    parser.add_argument("--task", choices=("hammer",), required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scene-context", type=Path, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vlm.yaml")
    parser.add_argument("--without-mapping", action="store_true", help="VLM-only 对照：不向模型提供映射或先验算子")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    mapping = json.loads(args.mapping.read_text())
    scene = json.loads(args.scene_context.read_text())
    scene, images = rgbd_inputs(scene, args.scene_context.parent, config["depth_range_m"])
    available = {key: value for key, value in AVAILABLE_TERMS.items()
                 if not args.without_mapping or not key.startswith("prior_")}
    context = dict(task_id=args.task, description=args.description, scene=scene, available_terms=available)
    if not args.without_mapping:
        context["mapping"] = mapping
    result, response_id = design(config, (ROOT / config["prompt"]).read_text(), context, images)
    recipe = assemble_recipe(result, mapping, recipe_id=args.output.stem,
        mapping_ref=str(args.mapping), profile_id=scene["profile_id"])
    recipe["generation"] = dict(model=config["model"], response_id=response_id,
        scene_context=str(args.scene_context), description=args.description,
        mapping_visible_to_vlm=not args.without_mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    print(f"recipe: {args.output.resolve()}")


if __name__ == "__main__":
    main()
