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
from typing import Any, Dict, List, Tuple, Union

from PIL import Image
from llava.utils.ops import convert_expand_to_square

__all__ = [
    "SeparatorStyle", "Conversation",
    "conv_llava_plain", "conv_llava_vicuna_v1", "conv_llava_vicuna_v1_mmtag", "conv_llava_llama", "conv_llava_deepseek_v3", "conv_vicuna_v1",
    "conv_llama", "conv_deepseek_v3", "conv_templates", "default_conversation",
]


class SeparatorStyle(Enum):
    """Enum for different separator styles used in conversations."""
    PLAIN = auto()  # Text style.
    VICUNA_V1 = auto()  # Vicuna-V1 style.
    LLAMA = auto()  # Llama style.
    DEEPSEEK_V3 = auto()  # DeepSeek-V3 style.
    QWEN2 = auto()  # Qwen2 style.


@dataclasses.dataclass
class Conversation:
    """Class representing a conversation with a system prompt, roles, and messages."""
    system_message: str = ""
    roles: Tuple[str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.VICUNA_V1
    sep: str = "###"
    sep2: str = None
    stop_str: Union[str, List[str]] = None
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
            wrap_sys = lambda message: f"<<SYS>>\n{message}\n<</SYS>>\n\n" if len(message) > 0 else message
            wrap_inst = lambda message: f"[INST] {message} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system_message) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.DEEPSEEK_V3:  # DeepSeek-V3 style.
            seps = [self.sep, self.sep2]
            ret = ""
            
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.QWEN2:  # Qwen2 style.
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def get_images(self, return_pil: bool = False) -> List[Any]:
        """Extract images from the conversation messages.

        Args:
            return_pil (bool, optional): Whether to return PIL images. Defaults to False.

        Returns:
            List[Any]: List of processed images.
        """
        images = []
        for i, (role, message) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(message) is tuple:
                    message, image, image_process_mode = message
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    @staticmethod
    def process_image(
            image: Image.Image,
            image_process_mode: str,
            return_pil: bool = False,
            image_format: str = "PNG",
            max_len: int = 1344,
            min_len: int = 672
    ) -> Union[str, Image.Image]:
        """Process an image according to the specified mode.

        Args:
            image (PIL.Image.Image): The input image.
            image_process_mode (str): The image processing mode ('Pad', 'Default', 'Crop', 'Resize').
            return_pil (bool, optional): Whether to return a PIL image. Defaults to False.
            image_format (str, optional): The format for saving the image. Defaults to "PNG".
            max_len (int, optional): Maximum allowed image size. Defaults to 1344.
            min_len (int, optional): Minimum allowed image size. Defaults to 672.

        Returns:
            Union[str, PIL.Image.Image]: Base64 string if return_pil is False, otherwise PIL image.
        """
        if image_process_mode == "Pad":
            image = convert_expand_to_square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            width, height = image.size
            if height > width:
                height, width = longest_edge, shortest_edge
            else:
                height, width = shortest_edge, longest_edge
            image = image.resize((width, height))

        if not return_pil:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            image_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return image_b64_str
        return image

    def to_gradio_chatbot(self) -> List[List[Any]]:
        """Convert the conversation to Gradio chatbot format.

        Returns:
            List[List[Any]]: List of message pairs for Gradio chatbot.
        """
        ret = []
        for i, (role, message) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(message) is tuple:
                    message, image, image_process_mode = message
                    image_b64_str = self.process_image(image, "Default", return_pil=False, image_format="JPEG")
                    image_str = f"<image src='data:image/jpeg;base64,{image_b64_str}' alt='user upload image' />"
                    message = image_str + message.replace("<image>", "").strip()
                    ret.append([message, None])
                else:
                    ret.append([message, None])
            else:
                ret[-1][-1] = message
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
            stop_str=self.stop_str,
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
                "system_template": self.system_message,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep_style": self.sep_style.name,
                "sep": self.sep,
                "sep2": self.sep2,
                "stop_str": self.stop_str,
                "skip_next": self.skip_next,
                "version": self.version,
            }
        return {
            "system_template": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep_style": self.sep_style.name,
            "sep": self.sep,
            "sep2": self.sep2,
            "stop_str": self.stop_str,
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
    stop_str=None,
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
    stop_str=None,
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
    stop_str=None,
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
    stop_str=None,
    skip_next=False,
    version="llava_llama",
)
conv_llava_deepseek_v3 = Conversation(
    system_message="<｜begin▁of▁sentence｜>",
    roles=("User", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DEEPSEEK_V3,
    sep="\n\n",
    sep2="<｜end▁of▁sentence｜>",
    stop_str="<｜end▁of▁sentence｜>",
    skip_next=False,
    version="llava_deepseek_v3",
)
conv_llava_qwen2 = Conversation(
    system_message="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN2,
    sep=" ",
    sep2="<|endoftext|>",
    stop_str=None,
    skip_next=False,
    version="llava_qwen2",
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
    stop_str=None,
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
    stop_str=None,
    skip_next=False,
    version="llama",
)
conv_deepseek_v3 = Conversation(
    system_message="<｜begin▁of▁sentence｜>",
    roles=("User", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DEEPSEEK_V3,
    sep="\n\n",
    sep2="<｜end▁of▁sentence｜>",
    stop_str="<｜end▁of▁sentence｜>",
    skip_next=False,
    version="deepseek_v3",
)
conv_qwen2 = Conversation(
    system_message="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN2,
    sep=" ",
    sep2="<|endoftext|>",
    stop_str=None,
    skip_next=False,
    version="qwen2",
)

conv_templates = {
    # pretrain.
    "llava_plain": conv_llava_plain,
    "llava_vicuna_v1": conv_llava_vicuna_v1,
    "llava_vicuna_v1_mmtag": conv_llava_vicuna_v1_mmtag,
    "llava_llama": conv_llava_llama,
    "llava_deepseek_v3": conv_llava_deepseek_v3,
    "llava_qwen2": conv_llava_qwen2,

    # finetune.
    "vicuna_v1": conv_vicuna_v1,
    "llama": conv_llama,
    "deepseek_v3": conv_deepseek_v3,
    "qwen2": conv_qwen2,
}
default_conversation = conv_templates["vicuna_v1"]
