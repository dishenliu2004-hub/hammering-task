"""从映射 latent 和抓锤照片设计一份奖励，确认后启动 Teacher。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from visualization.keypoint_preview import save_preview

from .api_client import design
from .reward_schema import AVAILABLE_TERMS, design_schema
from .rgbd_context import image_input

ROOT = Path(__file__).resolve().parent


def assemble_recipe(design_result, mapping, *, recipe_id, mapping_ref, profile_id):
    mapped = mapping["mapped_grasp"]
    return dict(version="1.0", recipe_id=recipe_id, task_id="hammer",
        mapping_ref=mapping_ref, profile_id=profile_id, **design_result,
        mapping_prior=dict(u_mean_RL=mapped["preferred_u_RL"], u_sigma_RL=mapped["u_sigma_RL"],
                           closing_axis_O=mapped["preferred_closing_axis_O"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-image", type=Path, required=True)
    parser.add_argument("--data-file", type=Path,
                        default=ROOT.parent / "weight/hammer_rl/reset_bank_v1_256.npz")
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vlm.yaml")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data_file = args.data_file.resolve()
    output = args.output_dir.resolve()
    mapping_path = output / "mapping.json"
    recipe_path = output / "reward_v1.json"
    preview_path = output / "grasp_preview.png"
    subprocess.run([sys.executable, "-m", "mapping.export_mapping", "--data-file",
                    str(data_file), "--output", str(mapping_path)], check=True)
    mapping = json.loads(mapping_path.read_text())
    with Image.open(args.hand_image) as photo:
        images = [image_input(ImageOps.exif_transpose(photo).convert("RGB"))]
    context = {"task_id": "hammer", "description": args.description, "available_terms": AVAILABLE_TERMS,
               "mapping_latent": mapping}
    result, response_id = design(config, (ROOT / config["prompt"]).read_text(), context, images,
        schema=design_schema(), name="reward_design")
    candidate_index = result.pop("preview_candidate")
    save_preview(mapping, candidate_index, preview_path)
    recipe = assemble_recipe(result, mapping, recipe_id=recipe_path.stem,
        mapping_ref=str(mapping_path), profile_id="default")
    recipe["generation"] = {"model": config["model"], "response_id": response_id,
        "mapping_image": str(preview_path), "hand_image": str(args.hand_image.resolve()),
        "preview_candidate": candidate_index,
        "data_file": str(data_file), "description": args.description}
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    print(f"三维关键点对照图：{preview_path}\n奖励配方：{recipe_path}")
    if input("查看预览后输入 yes 开始训练，直接回车取消：").strip() == "yes":
        with np.load(data_file) as bank:
            extra_mass = float(bank["hammer_head_extra_mass"])
        subprocess.run([sys.executable, "-m", "hammer_rl.train", "--stage", "teacher",
                        "--data-file", str(data_file), "--split", "all", "--reward-recipe", str(recipe_path),
                        "--hammer-head-extra-mass", str(extra_mass),
                        "--num-envs", str(args.num_envs), "--max-iterations", str(args.max_iterations),
                        "--seed", str(args.seed), "--device", args.device,
                        "--log-dir", str(output / "teacher")], check=True)


if __name__ == "__main__":
    main()
