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
import time
from threading import Thread
from typing import Dict, List, Generator, Tuple

import gradio as gr
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, LlavaProcessor, LlavaForConditionalGeneration, TextIteratorStreamer

model_id = "./results/stage2_finetune/llava-qwen1.5_0.5b_chat-clip_vit_large_patch14_336-stage2_coco_data/"
processor = LlavaProcessor.from_pretrained(model_id, use_fast=True)
kwargs = {
    "device_map": "auto",
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
}
model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
device = next(model.parameters()).device
temperature = 0.2
top_p = 0.7
max_new_tokens = 1024


def stream(message: Dict, history: List[Tuple[str, str]]) -> Generator[str, None, None]:
    """Stream responses from a LLaVA model given user input and conversation history.

    This function handles multimodal inputs (text + image), prepares the prompt,
    feeds it into the LLaVA model, and yields partial outputs in a streaming fashion.

    Args:
        message (Dict): The current user message. Expected to contain:
            - "text" (str): The user's text input.
            - "files" (List[str]): A list of file paths to uploaded images/videos.
        history (List[Tuple[str, str]]): The chat history, where each entry is a tuple
            (user_message, assistant_response).

    Yields:
        str: The assistant's generated response in streaming chunks.
    """
    # A buffer string used to help strip prompt text from generated outputs later.
    ext_buffer = f"user\n{message['text']} assistant"

    # Get the uploaded image from the current message.
    image = [message["files"][0]]

    # If no new image is uploaded, try to retrieve the last image from history.
    if not message["files"]:
        for hist in history:
            if type(hist[0]) == tuple:  # history may contain multimodal data.
                image = hist[0][0]

    if message["files"] is None:
        gr.Error("You need to upload an image or video for LLaVA to work.")

    image = Image.open(image[0]).convert("RGB")
    prompt = f"<|im_start|>user <image>\n{message['text']}<|im_end|><|im_start|>assistant"
    inputs = processor(text=prompt, images=image, return_tensors="pt", truncation=False).to(device)
    streamer = TextIteratorStreamer(
        processor,
        **{
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": True,
        },
    )
    generation_kwargs = dict(inputs, streamer=streamer, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer_text = ""
    for new_text in streamer:
        buffer_text += new_text
        generated_text_without_prompt = buffer_text[len(ext_buffer):]
        time.sleep(0.01)
        yield generated_text_without_prompt


app = gr.ChatInterface(
    fn=stream,
    multimodal=True,
    type="messages",
    examples=[
        {
            "text": "Do the cats in these two videos have same breed? What breed is each cat?",
            "files": ["./cats_1.mp4", "./cats_2.mp4"],
        },
        {
            "text": "These are the tech specs of two laptops I am choosing from. Which one should I choose for office work?",
            "files": ["./dell-tech-specs.jpeg", "./asus-tech-specs.png"],
        },
        {
            "text": "Here are several images from a cooking book, showing how to prepare a meal step by step. Can you write a recipe for the meal, describing each step in details?",
            "files": [
                "./step0.png",
                "./step1.png",
                "./step2.png",
                "./step3.png",
                "./step4.png",
                "./step5.png",
            ],
        },
        {"text": "What is on the flower?", "files": ["./bee.jpg"]},
        {
            "text": "What is this video about? Describe all the steps taken in the video so I can follow them, be very detailed",
            "files": ["./tutorial.mp4"],
        },
    ],
    title="LLaVA: Large Language and Vision Assistant",
    textbox=gr.MultimodalTextbox(file_count="multiple"),
    stop_btn="Stop Generation",
)
app.launch()
