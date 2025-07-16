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
from typing import Any, Dict, List, Union

from PIL import Image

__all__ = [
    "SeparatorStyle", "Conversation",
    "conv_templates", "conv_vicuna_v0", "conv_vicuna_v1", "conv_llama_2", "conv_llava_llama_2", "conv_mistral_instruct",
    "conv_chatml_direct", "conv_llava_plain", "conv_llava_v0", "conv_llava_v0_mmtag", "conv_llava_v1", "conv_llava_v1_mmtag", "conv_llava_llama_2",
    "conv_mpt", "default_conversation",
]


class SeparatorStyle(Enum):
    """Enum for different separator styles used in conversations."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """Class representing a conversation with a system prompt, roles, and messages."""
    system: str
    roles: Union[List[str], Any]
    messages: Union[List[str], Any]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    skip_next: bool = False

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

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
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
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
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
            def expand2square(pil_image, background_color=(122, 116, 104)):
                width, height = pil_image.size
                if width == height:
                    return pil_image
                elif width > height:
                    result = Image.new(pil_image.mode, (width, width), background_color)
                    result.paste(pil_image, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_image.mode, (height, height), background_color)
                    result.paste(pil_image, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
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

        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            image_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return image_b64_str

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
                    image_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format="JPEG")
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
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self) -> Dict:
        """Convert the conversation to a dictionary.

        Returns:
            Dict: Dictionary representation of the conversation.
        """
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
         "Renewable energy sources are those that can be replenished naturally in a relatively "
         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
         "renewable and non-renewable energy sources:\n"
         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
         "energy sources are finite and will eventually run out.\n"
         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
         "and other negative effects.\n"
         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
         "have lower operational costs than non-renewable sources.\n"
         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
         "locations than non-renewable sources.\n"
         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don"t know the answer to a question, please don"t share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}
default_conversation = conv_vicuna_v1
