"""评估当前 Teacher，用 VLM 修正奖励并沿最新 checkpoint 续训。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from .api_client import design
from .reward_schema import AVAILABLE_FORMULAS, AVAILABLE_TERMS, reflection_schema
from .rgbd_context import image_input

ROOT = Path(__file__).resolve().parent
FEEDBACK_METRICS = (
    "task_success", "valid_strike", "hit_progress", "strike_force_1", "strike_force_2",
    "strike_force_3", "mean_force", "drop", "lift_success", "grasp_before_swing",
    "hold_fraction", "capture", "wrap", "orientation", "stability", "translation_drift",
    "rotation_drift", "seat_closure", "seat_lift", "seat_before_swing", "recovery_hold",
    "prior_region", "prior_axis", "grasp_u_rl", "grasp_axis_cos2",
)


def train(args):
    command = [sys.executable, "-u", "-m", "hammer_rl.train", *args]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("checkpoint: "):
                checkpoint = Path(line.removeprefix("checkpoint: ").strip())
        subprocess.CompletedProcess(command, process.wait()).check_returncode()
    return checkpoint


def reflect(config, mapping, recipe, feedback_dir, recipe_id):
    with np.load(feedback_dir / "metrics.npz") as metrics:
        summary = {}
        episode_metrics = {name: metrics[name].tolist() for name in ("scene_id", "candidate_id")}
        for name in FEEDBACK_METRICS:
            values = metrics[name]
            summary[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                 p90=float(np.quantile(values, 0.9)))
            episode_metrics[name] = values.tolist()
    frames = feedback_dir / "frames"
    index = json.loads((frames / "index.json").read_text())
    generation = recipe["generation"]
    with Image.open(generation["hand_image"]) as photo:
        images = [image_input(ImageOps.exif_transpose(photo).convert("RGB"))]
    for frame in index:
        with Image.open(frames / frame["file"]) as image:
            images.append(image_input(image))
    context = dict(task_id=recipe["task_id"], description=generation["description"],
                   available_terms=AVAILABLE_TERMS, available_formulas=AVAILABLE_FORMULAS,
                   mapping_latent=mapping, current_recipe=recipe, metrics_summary=summary,
                   episode_metrics=episode_metrics, frame_index=index)
    result, response_id = design(config, (ROOT / config["reflection_prompt"]).read_text(), context,
                                 images, schema=reflection_schema(), name="reward_reflection")
    return dict(recipe, **result, recipe_id=recipe_id, version="2.0",
                generation=dict(generation, model=config["model"], response_id=response_id,
                                parent_recipe_id=recipe["recipe_id"], feedback_dir=str(feedback_dir),
                                rationale=result["rationale"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-recipe", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vlm.yaml")
    parser.add_argument("--evolution-rounds", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--hammer-head-extra-mass", type=float, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    mapping = json.loads(args.mapping.read_text())
    output = args.output_dir.resolve()
    recipe_path = args.initial_recipe
    runtime = ["--seed", str(args.seed), "--device", args.device]
    data_args = ["--data-file", str(args.data_file)]
    recipe_args = [*data_args, "--split", "train", "--reward-recipe", str(recipe_path),
                   "--hammer-head-extra-mass", str(args.hammer_head_extra_mass)]
    for version in range(1, args.evolution_rounds + 2):
        checkpoint = train(["--stage", "teacher", *recipe_args, *runtime,
                            "--max-iterations", str(args.max_iterations), "--num-envs", str(args.num_envs),
                            "--log-dir", str(output / f"teacher_v{version}")])
        feedback_dir = output / f"feedback_v{version}"
        subprocess.run([sys.executable, "-m", "hammer_rl.evaluate", *data_args, *runtime,
                        "--split", "eval", "--policy", "trained", "--selection", "learned",
                        "--checkpoint", str(checkpoint), "--episodes", "0", "--headless",
                        "--output", str(feedback_dir / "metrics.npz"),
                        "--feedback-dir", str(feedback_dir / "frames")], check=True)
        if version <= args.evolution_rounds:
            recipe = json.loads(recipe_path.read_text())
            recipe_path = output / f"reward_v{version + 1}.json"
            next_recipe = reflect(config, mapping, recipe, feedback_dir, recipe_path.stem)
            recipe_path.write_text(json.dumps(next_recipe, ensure_ascii=False, indent=2) + "\n")
            recipe_args = ["--resume", str(checkpoint), "--reward-recipe-override", str(recipe_path)]


if __name__ == "__main__":
    main()
