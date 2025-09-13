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
import base64
import dataclasses
from enum import auto, Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from llava.utils.ops import convert_expand_to_square

__all__ = [
    "SeparatorStyle", "Conversation",
    "conv_llava_plain", "conv_llava_vicuna_v1", "conv_llava_vicuna_v1_mmtag", "conv_llava_llama", "conv_llava_mistral_direct",
    "conv_llava_mistral_instruct", "conv_llava_qwen1_5", "conv_llava_qwen2", "conv_llava_qwen2_5", "conv_vicuna_v1", "conv_llama",
    "conv_mistral_direct", "conv_mistral_instruct", "conv_qwen1_5", "conv_qwen2", "conv_qwen2_5", "conv_templates", "default_conversation",
]


class SeparatorStyle(Enum):
    """Enum for different separator styles used in conversations."""
    PLAIN = auto()  # Text style.
    VICUNA_V1 = auto()  # Vicuna-V1 style.
    LLAMA = auto()  # Llama style.
    MPT = auto()  # MPT style.
    CHATML = auto()  # ChatML/Qwen style.


@dataclasses.dataclass
class Conversation:
    """Class representing a conversation with a system prompt, roles, and messages."""
    system_message: str = ""
    roles: Tuple[str, str] = ("USER", "ASSISTANT")
    messages: Any = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.VICUNA_V1
    sep: Optional[str] = None
    sep2: Optional[str] = None
    tokenizer_id: str = ""
    tokenizer: Any = None
    stop_str: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    skip_next: bool = False
    version: str = "Unknown"

    def append_message(self, role: Any, message: Any) -> None:
        """Append a message to the conversation.

        Args:
            role (Any): The role of the message sender.
            message (Any): The message content.
        """
        self.messages.append([role, message])

    def get_prompt(self) -> str:
        """Generate the prompt string from the conversation messages.

        Returns:
            str: The constructed prompt string.
        """
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_message = messages[0].copy()
            init_message = init_message[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_message)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_message)

        if self.sep_style == SeparatorStyle.PLAIN:  # Text style.
            seps = [self.sep, self.sep2]
            ret = self.system_message

            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.VICUNA_V1:  # Vicuna-V1 style.
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]

            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.LLAMA:  # Llama style.
            wrap_sys = lambda _message: f"<<SYS>>\n{_message}\n<</SYS>>\n\n" if len(_message) > 0 else _message
            wrap_inst = lambda _message: f"[INST] {_message} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system_message) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.MPT:  # MPT style.
            ret = self.system_message + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.CHATML:  # ChatML/Qwen/Hermes style.
            ret = "" if self.system_message == "" else self.system_message + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
        return ret

    def get_images(self, return_pil: bool = False) -> List[Any]:
        """Extract images from the conversation messages.

        Args:
            return_pil (bool, optional): Whether to return PIL images. Defaults to ``False``.

        Returns:
            List[Any]: List of processed images.
        """
        images = []
        for i, (role, message) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(message) is tuple:
                    message, image, image_process_mode = message
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        images.append(img)
        return images

    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Check if a filename corresponds to an image file.

        Args:
            filename (str): The name of the file to check.

        Returns:
            bool: True if the file is an image, False otherwise.
        """
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    @staticmethod
    def is_video_file(filename: str) -> bool:
        """Check if a filename corresponds to a video file.

        Args:
            filename (str): The name of the file to check.

        Returns:
            bool: True if the file is a video, False otherwise.
        """
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"]
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    @staticmethod
    def process_image(
            image: Union[Image.Image, str],
            image_process_mode: str,
            return_pil: bool = False,
            image_format: str = "PNG",
    ) -> Union[str, Image.Image]:
        """Process an image according to the specified mode.

        Args:
            image (Union[Image.Image, str]): The input image.
            image_process_mode (str): The image processing mode ('Pad', 'Default', 'Crop', 'Resize').
            return_pil (bool, optional): Whether to return a PIL image. Defaults to False.
            image_format (str, optional): The format for saving the image. Defaults to "PNG".

        Returns:
            Union[str, Image.Image]: Base64 string if return_pil is False, otherwise PIL image.
        """
        if image_process_mode == "Pad":
            image = convert_expand_to_square(image, background_color=(114, 114, 114))
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        image_width, image_height = image.size
        if image_height > image_width:
            image_height, image_width = longest_edge, shortest_edge
        else:
            image_height, image_width = shortest_edge, longest_edge

        image = image.resize((image_width, image_height))

        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            image_base64_str = base64.b64encode(buffered.getvalue()).decode()
            return image_base64_str

    def to_gradio_chatbot(self) -> List[List[Any]]:
        """Convert the conversation to Gradio chatbot format.

        Returns:
            List[List[Any]]: List of message pairs for Gradio chatbot.
        """
        ret = []
        for i, (role, message) in enumerate(self.messages[self.offset:]):
            if isinstance(message, tuple):
                text, image, image_process_mode = message
                if not isinstance(image, list):
                    image = [image]

                image_str_list = []
                for img in image:
                    if isinstance(img, Image.Image):
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        image_base64_str = base64.b64encode(buffered.getvalue()).decode()
                        image_str = f'<img src="data:image/jpeg;base64,{image_base64_str}" style="max-width:256px; max-height:256px; object-fit:contain;"/>'
                        image_str_list.append(image_str)
                    elif self.is_image_file(img):
                        image_base64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                        image_str = f'<img src="data:image/jpeg;base64,{image_base64_str}" style="max-width:256px; max-height:256px; object-fit:contain;"/>'
                        image_str_list.append(image_str)

                content = "\n\n".join(image_str_list) + "\n\n" + text.strip()
            else:
                content = str(message)

            if i % 2 == 0:
                ret.append([content, None])
            else:
                ret[-1][-1] = content
        return ret

    def copy(self) -> "Conversation":
        """Create a copy of the conversation.

        Returns:
            Conversation: A new Conversation object with the same data.
        """
        return Conversation(
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            tokenizer_id=self.tokenizer_id,
            tokenizer=self.tokenizer,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            skip_next=self.skip_next,
            version=self.version,
        )

    def dict(self) -> Dict:
        """Convert the conversation to a dictionary.

        Returns:
            Dict: Dictionary representation of the conversation.
        """
        if len(self.get_images()) > 0:
            return {
                "system_message": self.system_message,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep_style": self.sep_style.name,
                "sep": self.sep,
                "sep2": self.sep2,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer": self.tokenizer_id,
                "stop_str": self.stop_str,
                "stop_token_ids": self.stop_token_ids,
                "skip_next": self.skip_next,
                "version": self.version,
            }
        return {
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep_style": self.sep_style.name,
            "sep": self.sep,
            "sep2": self.sep2,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer": self.tokenizer_id,
            "stop_str": self.stop_str,
            "stop_token_ids": self.stop_token_ids,
            "skip_next": self.skip_next,
            "version": self.version,
        }


conv_llava_plain = Conversation(
    system_message="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_plain",
)
conv_llava_vicuna_v1 = Conversation(
    system_message="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA_V1,
    sep=" ",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_vicuna_v1",
)
conv_llava_vicuna_v1_mmtag = Conversation(
    system_message="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
                   "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA_V1,
    sep=" ",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_vicuna_v1_mmtag",
)
conv_llava_llama = Conversation(
    system_message="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, "
                   "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA,
    sep="<s>",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_llama",
)
conv_llava_mistral_direct = Conversation(
    system_message="<|im_start|>system\nAnswer the questions.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_mistral_direct",
)
conv_llava_mistral_instruct = Conversation(
    system_message="",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA,
    sep="",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_mistral_instruct",
)
conv_llava_qwen1_5 = Conversation(
    system_message="<|im_start|>system\nYou are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_qwen1_5",
)
conv_llava_qwen2 = Conversation(
    system_message="<|im_start|>system\nYou are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_qwen2",
)
conv_llava_qwen2_5 = Conversation(
    system_message="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llava_qwen2_5",
)

conv_vicuna_v1 = Conversation(
    system_message="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.VICUNA_V1,
    sep=" ",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="vicuna_v1",
)
conv_llama = Conversation(
    system_message="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                   "Please ensure that your responses are socially unbiased and positive in nature. "
                   "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                   "If you don't know the answer to a question, please don't share false information.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA,
    sep="<s>",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="llama",
)
conv_mistral_direct = Conversation(
    system_message="<|im_start|>system\nAnswer the questions.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="mistral_direct",
)
conv_mistral_instruct = Conversation(
    system_message="",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA,
    sep="",
    sep2="</s>",
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="mistral_instruct",
)
conv_qwen1_5 = Conversation(
    system_message="<|im_start|>system\nYou are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="qwen1_5",
)
conv_qwen2 = Conversation(
    system_message="<|im_start|>system\nYou are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="qwen2",
)
conv_qwen2_5 = Conversation(
    system_message="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
    sep2=None,
    tokenizer_id="",
    tokenizer=None,
    stop_str=None,
    stop_token_ids=None,
    skip_next=False,
    version="qwen2_5",
)

conv_templates = {
    # pretrain.
    "llava_plain": conv_llava_plain,
    "llava_vicuna_v1": conv_llava_vicuna_v1,
    "llava_vicuna_v1_mmtag": conv_llava_vicuna_v1_mmtag,
    "llava_llama": conv_llava_llama,
    "llava_mistral_direct": conv_llava_mistral_direct,
    "llava_mistral_instruct": conv_llava_mistral_instruct,
    "llava_qwen1_5": conv_llava_qwen1_5,
    "llava_qwen2": conv_llava_qwen2,
    "llava_qwen2_5": conv_llava_qwen2_5,

    # finetune.
    "vicuna_v1": conv_vicuna_v1,
    "llama": conv_llama,
    "mistral_direct": conv_mistral_direct,
    "mistral_instruct": conv_mistral_instruct,
    "qwen1_5": conv_qwen1_5,
    "qwen2": conv_qwen2,
    "qwen2_5": conv_qwen2_5,
}
default_conversation = conv_templates["vicuna_v1"]
