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
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import orjson
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, Blip2Processor, Blip2ForConditionalGeneration


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs",
        type=str,
        required=True,
        help="Root directory containing subfolders to process.",
    )
    parser.add_argument(
        "--model-weights-path",
        default="Salesforce/blip2-opt-2.7b",
        type=str,
        help="Path to the pretrained BLIP model weights. Defaults to ``Salesforce/blip2-opt-2.7b``.",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="Number of image to process per batch. Defaults to 128.",
    )
    return parser.parse_args()


def generate_blip_caption_for_folder(root: Union[Path, str], model_weights_path: Union[Path, str], batch_size: int = 128):
    root = Path(root)
    model_weights_path = str(model_weights_path)

    # Load BLIP processor and model,
    processor = Blip2Processor.from_pretrained(model_weights_path, use_fast=True)
    model_kwargs = {
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        "device_map": "auto",
    }
    model = Blip2ForConditionalGeneration.from_pretrained(model_weights_path, **model_kwargs)

    pending_images = []
    pending_records = []
    # Recursively iterate through all JSON files in the root directory,
    for json_file in sorted(root.rglob("*.json")):
        json_file = Path(json_file)
        with json_file.open("rb") as f:
            data = orjson.loads(f.read())

        # Skip if the BLIP caption already exists,
        if data.get("blip_caption"):
            continue

        image_file = json_file.with_suffix(".jpg")
        if not image_file.exists():
            print(f"Image not found: {image_file}.")
            continue

        # Append image and its corresponding record to batch lists,
        pending_images.append(Image.open(image_file).convert("RGB"))
        pending_records.append((data, json_file))

        # Generate BLIP captions in batch.
        if len(pending_images) >= batch_size:
            batch_generate(model, processor, pending_images, pending_records)
            pending_images.clear()
            pending_records.clear()

    # Process any remaining images that didn't fill a full batch.
    if pending_images:
        batch_generate(model, processor, pending_images, pending_records)


def batch_generate(
        model: Any,
        processor: Any,
        images: List[Image.Image],
        records: List[Tuple[Dict, Path]]
) -> None:
    # Preprocess images and move tensors to GPU.
    inputs = processor(images=images, return_tensors="pt").to("cuda", torch.float16)

    # Generate captions without tracking gradients.
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode the generated captions.
    captions = [processor.decode(out, skip_special_tokens=True).strip() for out in outputs]

    # Update JSON data with BLIP captions and write back to file.
    for (data, json_file), blip_caption in zip(records, captions):
        data["blip_caption"] = blip_caption
        with json_file.open("wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def main() -> None:
    opts = get_opts()

    generate_blip_caption_for_folder(
        opts.inputs,
        opts.model_weights_path,
        opts.batch_size,
    )


if __name__ == "__main__":
    main()
