# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import json
from pathlib import Path
from typing import Union

from tqdm import tqdm


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs",
        type=str,
        required=True,
        help="Path to VQAv2 questions json.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output jsonl file.",
    )
    parser.add_argument(
        "--coco-split",
        type=str,
        default="test2015",
        choices=["train2014", "val2014", "test2015"],
        help="Which COCO split the images belong to.",
    )
    return parser.parse_args()


def convert_vga2_to_jsonl(question_file: Union[Path, str], output_file: Union[Path, str], coco_split: str = "test2015") -> None:
    question_file = Path(question_file)

    if output_file is None:
        output_file = question_file.with_suffix(".jsonl")
    else:
        output_file = Path(output_file)

    with question_file.open("r", encoding="utf-8") as file:
        json_data = json.load(file)

    questions = json_data["questions"]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as fout:
        for question in tqdm(questions, desc="Converting"):
            question_id = question["question_id"]
            image_id = question["image_id"]
            image_file = f"COCO_{coco_split}_{image_id:012d}.jpg"
            text = question["question"].strip() + "\nAnswer the question using a single word or phrase."
            entry = {
                "question_id": question_id,
                "image": image_file,
                "text": text,
                "category": "default"
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Wrote '{len(questions)}' questions to '{output_file}'.")


def main() -> None:
    opts = get_opts()

    convert_vga2_to_jsonl(opts.inputs, opts.output, opts.coco_split)


if __name__ == "__main__":
    main()
