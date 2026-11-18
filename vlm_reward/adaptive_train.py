"""按同一组物理指标比较同源 Teacher 的旧、新奖励续训分支。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .reward_refiner import physical_metrics, prepare_request


def run(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def evaluate(args, checkpoint, directory, *, frames=False):
    output = directory / "evaluation.npz"
    run("hammer_rl.evaluate", "--data-file", args.data_file, "--checkpoint", checkpoint,
        "--policy", "trained", "--selection", "sdk", "--split", "eval", "--episodes", 0,
        "--num-envs", args.eval_num_envs, "--seed", args.seed, "--headless", "--output", output,
        *(("--feedback-dir", directory) if frames else ()))
    with np.load(output) as data:
        return physical_metrics(data)


def train_branch(args, checkpoint, recipe, directory):
    run("hammer_rl.train", "--stage", "teacher", "--resume", checkpoint,
        "--reward-recipe-override", recipe, "--log-dir", directory, "--seed", args.seed,
        "--num-envs", args.num_envs, "--max-iterations", args.train_iters_per_round)
    return max(directory.glob("model_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def choose_by_physics(old, new):
    def rank(m):
        return (m["task_success"], m["grasp_before_swing"], m["hit_progress"], -m["drop"], m["mean_force"])
    return "new" if rank(new) > rank(old) else "old"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-file", "initial-checkpoint", "initial-recipe", "mapping", "scene-context", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--train-iters-per-round", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--eval-num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline-recipes", type=Path, nargs="+", help="离线验证时按轮提供人工 v2 配方")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    history = root / "rounds.jsonl"
    recipes = root / "recipes"
    recipes.mkdir(parents=True)
    history.touch()
    checkpoint = args.initial_checkpoint.resolve()
    recipe = args.initial_recipe.resolve()
    for k in range(1, args.rounds + 1):
        directory = root / f"round_{k:03d}"
        feedback = directory / "feedback"
        source_metrics = evaluate(args, checkpoint, feedback, frames=True)
        proposed = recipes / f"recipe_{k:03d}.json"
        if args.offline_recipes:
            shutil.copyfile(args.offline_recipes[k - 1], proposed)
            response_id = failure_mode = None
            prepare_request(recipe, args.mapping, args.scene_context, feedback / "evaluation.npz", history, feedback)
        else:
            run("vlm_reward.reward_refiner", "--recipe", recipe, "--mapping", args.mapping,
                "--scene-context", args.scene_context, "--evaluation", feedback / "evaluation.npz",
                "--history", history, "--feedback-dir", feedback, "--output", proposed)
            revision = json.loads(proposed.read_text())
            response_id, failure_mode = revision["revision"]["response_id"], revision["failure_mode"]

        old_checkpoint = train_branch(args, checkpoint, recipe, directory / "old")
        new_checkpoint = train_branch(args, checkpoint, proposed, directory / "new")
        old_metrics = evaluate(args, old_checkpoint, directory / "old")
        new_metrics = evaluate(args, new_checkpoint, directory / "new")
        selected = choose_by_physics(old_metrics, new_metrics)
        next_checkpoint, next_recipe = ((new_checkpoint, proposed) if selected == "new" else (old_checkpoint, recipe))
        updates = k * args.train_iters_per_round
        record = dict(round=k, source_checkpoint=str(checkpoint), old_recipe=str(recipe), new_recipe=str(proposed),
                      source_metrics=source_metrics, old_metrics=old_metrics, new_metrics=new_metrics,
                      selected_branch=selected, selected_checkpoint=str(next_checkpoint), selected_recipe=str(next_recipe),
                      response_id=response_id, failure_mode=failure_mode, seed=args.seed, num_envs=args.num_envs,
                      updates_per_branch=args.train_iters_per_round, cumulative_selected_updates=updates,
                      cumulative_extra_training_updates=updates, cumulative_extra_training_attempts=updates * args.num_envs)
        with history.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint, recipe = next_checkpoint, next_recipe
        print(f"round {k}: {selected}, selected_checkpoint={checkpoint}, selected_recipe={recipe}", flush=True)


if __name__ == "__main__":
    main()
