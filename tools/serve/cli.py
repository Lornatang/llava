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

import torch
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils.checkpoint import load_pretrained
from llava.utils.ops import load_image, process_images, tokenizer_image_token
from llava.utils.torch_utils import disable_torch_init


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-file",
        type=str,
        required=True,
        help="Path to the image file to be processed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pretrained model or model name.",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        required=True,
        help="Conversation mode to use.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Whether to load the model in 8-bit mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for text generation. Default is 0.2.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate. Default is 512.",
    )
    return parser.parse_args()


def cli(
        image_file: str,
        model: str,
        conv_mode: str,
        load_in_8bit: bool = False,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        device: str = "cuda",
) -> None:
    disable_torch_init()

    tokenizer, model, image_processor = load_pretrained(model, load_in_8bit, device)

    image = load_image(image_file)
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    while True:
        try:
            inputs = input(f"{roles[0]}: ")
        except EOFError:
            inputs = ""
        if not inputs:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None and len(conv.messages) == 0:
            if model.config.mm_use_im_start_end:
                inputs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inputs
            else:
                inputs = DEFAULT_IMAGE_TOKEN + "\n" + inputs

        conv.append_message(conv.roles[0], inputs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.VICUNA_V1 else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                attention_mask=attention_mask,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs


def main() -> None:
    args = get_opts()

    cli(**vars(args))


if __name__ == "__main__":
    main()
