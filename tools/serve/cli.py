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
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils.checkpoint import load_pretrained
from llava.utils.ops import KeywordsStoppingCriteria, load_image, tokenizer_image_token
from llava.utils.torch_utils import disable_torch_init
from transformers import TextStreamer


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-path",
        type=str,
        required=True,
        help="Path to the image file to be processed.",
    )
    parser.add_argument(
        "--model-path",
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
        "--load-8bit",
        action="store_true",
        help="Whether to load the model in 8-bit mode.",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Whether to load the model in 4-bit mode.",
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
        default=1024,
        help="Maximum number of new tokens to generate. Default is 1024.",
    )
    return parser.parse_args()


def cli(
        image_path: str,
        model_path: str,
        conv_mode: str,
        load_8bit: bool = False,
        load_4bit: bool = False,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
) -> None:
    disable_torch_init()

    tokenizer, model, image_processor, _ = load_pretrained(model_path, load_8bit, load_4bit)

    image = load_image(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()

    conv = conv_templates[conv_mode].copy()
    while True:
        try:
            inputs = input(f"User: ")
        except EOFError:
            inputs = ""
        if not inputs:
            print("exit...")
            break

        print(f"Assistant: ", end="")

        if image is not None and len(conv.messages) == 0:
            if model.config.mm_use_im_start_end:
                inputs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inputs
            else:
                inputs = DEFAULT_IMAGE_TOKEN + "\n" + inputs
            conv.append_message(conv.roles[0], inputs)
        else:
            conv.append_message(conv.roles[0], inputs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.VICUNA_V1 else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                image_tensor,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs


def main() -> None:
    args = get_opts()

    cli(**vars(args))


if __name__ == "__main__":
    main()
