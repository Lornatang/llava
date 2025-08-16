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

from transformers import LlavaProcessor

from llava.utils.checkpoint import load_pretrained


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--patch-size",
        default=14,
        type=int,
        help="Patch size for the image encode. Defaults to 14.",
    )
    parser.add_argument(
        "--vision-feature",
        default="default",
        type=str,
        help="Vision feature to use. Defaults to ``default``",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to the output directory.",
    )
    return parser.parse_args()


def save_llavaprocesser(
        model_path: str,
        patch_size: int,
        vision_feature: str,
        output: str,
) -> None:
    tokenizer, model, image_processor, _ = load_pretrained(model_path)

    llava_processor = LlavaProcessor(
        image_processor,
        tokenizer,
        patch_size,
        vision_feature_select_strategy=vision_feature,
    )
    llava_processor.save_pretrained(output)


def main() -> None:
    opts = get_opts()
    if not opts.output:
        opts.output = opts.model_path

    save_llavaprocesser(**vars(opts))


if __name__ == "__main__":
    main()
