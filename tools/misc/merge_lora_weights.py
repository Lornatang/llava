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

from llava.utils.checkpoint import load_pretrained


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-base",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        required=True,
    )
    return parser.parse_args()


def merge_lora_weights(
        model_path: str,
        model_base: str,
        save_model_path: str,
) -> None:
    tokenizer, model, image_processor, _ = load_pretrained(model_path, model_base)
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


def main() -> None:
    opts = get_opts()

    merge_lora_weights(opts.model_path, opts.model_base, opts.save_model_path, )


if __name__ == "__main__":
    main()
