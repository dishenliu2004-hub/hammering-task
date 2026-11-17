"""从真实 ContactPose use 子集导出 MappingCode JSON。"""

import argparse
import json
from pathlib import Path

import yaml

from .contactpose_reader import read_records
from .mapping_encoder import encode
from .omnipicker_adapter import gripper_description

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("hammer", "broom", "knife"), required=True)
    parser.add_argument("--contactpose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "mapping/configs/mapping.yaml")
    parser.add_argument("--tool-frames", type=Path, default=ROOT / "mapping/configs/tool_frames.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frames = yaml.safe_load(args.tool_frames.read_text())
    obj = config["tasks"][args.task]["source_object"]
    records = read_records(args.contactpose_root.resolve(), obj, config["participants"])
    code = encode(args.task, records, frames["contactpose"][obj],
                  gripper_description(ROOT / config["gripper_xml"]), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(code, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.output}: {code['source']['sample_count']} use records")


if __name__ == "__main__":
    main()
