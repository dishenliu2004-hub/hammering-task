"""把 ContactPose 锤击先验映射到 OmniPicker，导出 JSON 和抓姿图。"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .contactpose_reader import read_records
from .grasp_preview import save_preview
from .mapping_encoder import encode
from .omnipicker_adapter import gripper_description

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "mapping/configs/mapping.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = json.loads((ROOT / "data/human_grasp/processed/hammer_mapping.json").read_text())
    prior = source["contact_prior"]
    hand = dict(source["hand_grasp"], u_mean=prior["u_mean"], u_std=prior["u_std"],
                source="contactpose")
    with np.load(args.data_file) as bank:
        handle_size = bank["handle_size"].mean(axis=0).tolist()
    code = encode(hand, gripper_description(ROOT / config["gripper_xml"]), handle_size)

    participant = source["source"]["records"][0]["participant_id"]
    record = next(read_records(ROOT / "data/human_grasp/source/contactpose", "hammer", [participant]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(code, indent=2, ensure_ascii=False) + "\n")
    save_preview(code, source, record, args.output.with_name("grasp_preview.png"))


if __name__ == "__main__":
    main()
