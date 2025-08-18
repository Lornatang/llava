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
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import shortuuid
import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.utils.checkpoint import load_pretrained
from llava.utils.ops import process_images, tokenizer_image_token
from llava.utils.torch_utils import disable_torch_init


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="./results/stage2_finetune/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-stage2_data",
        type=str,
        help="Path to model path. Defaults to ``./results/stage2_finetune/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-stage2_data``."
    )
    parser.add_argument(
        "--image-folder",
        default="./results/eval/vqav2/test2015.",
        type=str,
        help="Path to image folder. Defaults to ``./results/eval/vqav2/test2015.``."
    )
    parser.add_argument(
        "--question-file",
        default="./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions.jsonl.",
        type=str,
        help="Path to VQAv2 questions jsonl. Defaults to ``./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions.jsonl.``."
    )
    parser.add_argument(
        "--answer-file",
        default="./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions/vicuna-13b-v1.5.jsonl.",
        type=str,
        help="Path to VQAv2 answers jsonl. Defaults to ``./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions/vicuna-13b-v1.5.jsonl.``."
    )
    parser.add_argument(
        "--conv-mode",
        default="llava_v1",
        type=str,
        help="Conversation mode. Defaults to ``llava_v1``."
    )
    parser.add_argument(
        "--temperature",
        default=0.2,
        type=float,
        help="Temperature for sampling. Defaults to 0.2."
    )
    parser.add_argument(
        "--top_p",
        default=None,
        type=float,
        help="Top-p for sampling. Defaults to ``None``."
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams for beam search. Defaults to 1."
    )
    parser.add_argument(
        "--max_new_tokens",
        default=128,
        type=int,
        help="The maximum number of new tokens to be generated. Defaults to 128."
    )
    return parser.parse_args()


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset for multi-modal VQA tasks.

    Prepares text prompts and processes images for a given VQA model configuration.
    Each item returned is a tuple of tokenized input IDs, image tensor, and image size.

    Attributes:
        questions (List[dict]): List of question dictionaries, each containing "image" and "text".
        image_folder (str): Path to the folder containing images.
        tokenizer (Any): Tokenizer object for converting text prompts to input IDs.
        image_processor (Any): Image processor/preprocessor for model input.
        model_config (Any): Model configuration object containing flags like mm_use_im_start_end.
        conv_mode (str): Conversation template mode to use for prompt formatting.
    """

    def __init__(
            self,
            questions: List[dict],
            image_folder: str,
            tokenizer: Any,
            image_processor: Any,
            model_config: Any,
            conv_mode: str
    ):
        """Initializes the CustomDataset.

        Args:
            questions (List[dict]): List of question dictionaries.
            image_folder (str): Path to the image folder.
            tokenizer (Any): Tokenizer object.
            image_processor (Any): Image processor object.
            model_config (Any): Model configuration.
            conv_mode (str): Conversation template mode.
        """
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Fetches a single item from the dataset.

        Converts the text question into token IDs, processes the corresponding image,
        and returns them along with the image size.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
                - input_ids: Tokenized input IDs of the text prompt.
                - image_tensor: Processed image tensor ready for model input.
                - image_size: Original (width, height) of the image.
        """
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self) -> int:
        """Returns the total number of questions in the dataset.

        Returns:
            int: Number of questions.
        """
        return len(self.questions)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[int, int], ...]]:
    """Collate function for DataLoader to combine a batch of samples.

    This function takes a list of dataset items and stacks the input IDs and image tensors
    along the batch dimension. Image sizes are returned as a tuple of (width, height).

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]):
            List of tuples, each containing:
            - input_ids (torch.Tensor): Tokenized input IDs of the text prompt.
            - image_tensor (torch.Tensor): Processed image tensor.
            - image_size (Tuple[int, int]): Original (width, height) of the image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[int, int], ...]]:
            - input_ids (torch.Tensor): Batched input IDs stacked along dim=0.
            - image_tensors (torch.Tensor): Batched image tensors stacked along dim=0.
            - image_sizes (Tuple[Tuple[int, int], ...]): Tuple of original image sizes for each sample.
    """
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def eval_vqa_model(
        model_path: str,
        image_folder: str,
        question_file: str,
        answer_file: str,
        conv_mode: str,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 128,
) -> None:
    question_file = Path(question_file)
    answer_file = Path(answer_file)

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained(model_path)

    with question_file.open("r", encoding="utf-8") as file:
        questions = [json.loads(q) for q in file]

    answer_file.parent.mkdir(parents=True, exist_ok=True)
    answer_fout = answer_file.open("w", encoding="utf-8")

    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model.config, conv_mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

    model_id = Path(model_path).name

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        index = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device="cuda", non_blocking=True).long()
        do_sample = True if temperature > 0 else False
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                image_tensor,
                image_sizes,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answer_id = shortuuid.uuid()
        answer_fout.write(
            json.dumps(
                {
                    "question_id": index,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": answer_id,
                    "model_id": model_id,
                    "metadata": {},
                }
            ) + "\n")
    answer_fout.close()


def main() -> None:
    opts = get_opts()

    eval_vqa_model(
        opts.model_path,
        opts.image_folder,
        opts.question_file,
        answer_file=opts.answer_file,
        conv_mode=opts.conv_mode,
        temperature=opts.temperature,
        top_p=opts.top_p,
        num_beams=opts.num_beams,
        max_new_tokens=opts.max_new_tokens,
    )


if __name__ == "__main__":
    main()
