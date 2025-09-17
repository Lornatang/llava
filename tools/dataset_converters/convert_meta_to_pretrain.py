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
import random
from pathlib import Path
from typing import Union

HUMAN_PROMPTS = [
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "What is this?",
    "What is in the photo?",
    "Describe the image concisely.",
    "Share a concise interpretation of the image provided.",
    "Give a brief description of the image.",
    "Present a compact description of the photo's key features.",
    "Provide a brief description of the given image.",
    "Summarize the visual content of the image.",  # Origin LLaVA.
    "Give a short and clear explanation of the subsequent image.",
    "Offer a concise summary of what the picture shows.",
    "Briefly explain the main content of the photo.",
    "Provide a short overview of the image.",
    "Summarize the picture in a few words.",
    "Write a compact description of the image’s subject.",
    "What does the image depict?",
    "Give a succinct account of the visual scene.",
    "Provide a short interpretation of the photo.",
    "Explain briefly what can be seen in the picture."
]


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs",
        type=str,
        required=True,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def convert_meta_to_pretrain(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        prompt = random.choice(HUMAN_PROMPTS)

        if random.random() > 0.5:
            human_value = f"{prompt}\n<image>"
        else:
            human_value = f"<image>\n{prompt}"

        new_item = {
            "id": item["id"],
            "image": item["image"],
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": item.get("blip_caption", "")
                }
            ]
        }
        new_data.append(new_item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Done, saved to `{output_path}`.")


def main() -> None:
    opts = get_opts()

    convert_meta_to_pretrain(
        opts.inputs,
        opts.output,
    )


if __name__ == "__main__":
    main()
