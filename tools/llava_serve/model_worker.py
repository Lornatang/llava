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
import asyncio
import json
import threading
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import SERVER_ERROR_MSG, WORKER_HEART_BEAT_INTERVAL
from llava.utils.checkpoint import load_pretrained
from llava.utils.events import LOGGER
from llava.utils.ops import load_image_from_base64, process_images, pretty_print_semaphore, tokenizer_image_token

global_counter = 0
model_semaphore = None


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="Host IP to listen on. Defaults to ``127.0.0.1``.",
    )
    parser.add_argument(
        "--port",
        default=40000,
        type=int,
        help="Port to listen on. Defaults to 40000.",
    )
    parser.add_argument(
        "--worker",
        default="http://127.0.0.1:40000",
        type=str,
        help="Publicly accessible worker address. Defaults to ``http://127.0.0.1:40000``.",
    )
    parser.add_argument(
        "--controller",
        default="http://127.0.0.1:10000",
        type=str,
        help="Controller server address. Defaults to ``http://127.0.0.1:10000``.",
    )
    parser.add_argument(
        "--model-path",
        default="./results/stage2_finetune_ov/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-stage2_ov_data",
        type=str,
        help="Path to the model checkpoint directory. Defaults to ``./results/stage2_finetune_ov/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-stage2_ov_data``.",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        type=str,
        help="Base model name. Defaults to ``None``.",
    )
    parser.add_argument(
        "--limit-model-concurrency",
        default=5,
        type=int,
        help="Maximum number of concurrent model requests allowed.",
    )
    parser.add_argument(
        "--stream-interval",
        default=1,
        type=int,
        help="Interval in seconds between streaming outputs.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization for model loading.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Enable 8-bit quantization for model loading.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        type=str,
        help="Attention implementation. Defaults to ``None``.",
    )
    return parser.parse_args()


class ModelWorker:
    """A worker node that loads a model and communicates with a central controller."""

    def __init__(
            self,
            controller: str,
            worker: str,
            worker_id: str,
            model_path: str,
            model_base: str,
            load_in_4bit: bool,
            load_in_8bit: bool,
            attn_implementation: Optional[str] = None,
    ):
        """Initializes a ModelWorker by loading the pretrained model and tokenizer.

        Args:
            controller (str): Controller HTTP endpoint.
            worker (str): This worker's HTTP endpoint.
            worker_id (str): Unique worker identifier.
            model_path (str): Path to the pretrained model.
            model_base (str): Path to base model.
            load_in_4bit (bool): Whether to load the model in 4-bit precision.
            load_in_8bit (bool): Whether to load the model in 8-bit precision.
            attn_implementation (Optional[str], optional): Whether to use flash_attn_implementation. Defaults to ``None``.
        """
        self.controller: str = controller
        self.worker: str = worker
        self.worker_id: str = worker_id
        self.model_path: str = model_path
        self.model_base: str = model_base

        LOGGER.info(f"Loading the model {self.model_path} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_length = load_pretrained(
            model_path,
            model_base,
            load_in_4bit,
            load_in_8bit=load_in_8bit,
            attn_implementation=attn_implementation,
        )

        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
        self.heart_beat_thread.start()

    def register_to_controller(self) -> None:
        """Registers this worker to the controller.

        Sends a POST request to the controller's `/register_worker` endpoint,
        including the worker's status and name.
        """
        LOGGER.info("Register to controller.")
        url = self.controller + "/register_worker"
        data = {
            "worker_name": self.worker,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self) -> None:
        """Sends periodic heartbeat signals to the controller.

        This method continuously attempts to send a POST request to the
        `/receive_heart_beat` endpoint of the controller to report the current
        queue length. If the worker is not registered, it will attempt to
        re-register.

        Retries every 5 seconds upon failure.
        """
        LOGGER.info(
            f"Send heart beat. Models: {[self.model_path]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}."
        )

        url = self.controller + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={"worker_name": self.worker, "queue_length": self.get_queue_length()},
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                LOGGER.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    @staticmethod
    def get_queue_length() -> int:
        """Gets the number of pending requests in the model's queue.

        Returns:
            int: The total number of requests waiting to be processed, including
            currently active ones.
        """
        if model_semaphore is None:
            return 0
        else:
            return opts.limit_model_concurrency - model_semaphore._value + (
                len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self) -> Dict[str, Any]:
        """Retrieves the current status of the worker.

        Returns:
            Dict[str, Any]: A dictionary containing model name(s), processing speed,
            and queue length.
        """
        return {
            "model_names": [Path(self.model_path).name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params: Any) -> Generator[bytes, None, None]:
        """Generates text from a given prompt in a streaming fashion.

        Handles both text-only prompts and prompts that include images.

        Args:
            params (Any): Inference parameters.

        Yields:
            bytes: JSON-encoded partial output messages, terminated by b"\\0".

        Raises:
            ValueError: If the number of provided images does not match the number of image tokens in the prompt.
        """
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        original_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0

        # Process images if provided.
        if images is not None and len(images) > 0:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                image_sizes = [image.size for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, "mm_use_im_start_end", False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
                image_sizes = None
            image_args = {"images": images, "image_sizes": image_sizes}
        else:
            image_args = {}

        # Tokenize and prepare input.
        device = next(model.parameters()).device
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        temperature = float(params.get("temperature", 0.2))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 2048)
        max_context_length = getattr(model.config, "max_position_embeddings", 2048)
        stop_str = params.get("stop", None)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
        if max_new_tokens < 1:
            yield json.dumps(
                {"text": original_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        # Start model generation in a separate thread.
        thread = threading.Thread(
            target=model.generate,
            kwargs=dict(
                inputs=input_ids,
                attention_mask=attention_mask,
                do_sample=True if temperature > 0.001 else False,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                **image_args,
            ),
        )
        thread.start()

        start_time = time.time()
        generated_text = original_prompt
        for new_text in streamer:
            generated_text += new_text
            if isinstance(stop_str, str) and generated_text.endswith(stop_str):
                generated_text = generated_text[: -len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

        end_time = time.time()

        new_generated = generated_text[len(original_prompt):]
        new_generated_tokens = tokenizer(new_generated).input_ids
        token_per_second = len(new_generated_tokens) / (end_time - start_time)
        print(f"token_per_second: {token_per_second}.")

    def generate_stream_gate(self, params: Dict[str, Any]) -> Generator[bytes, None, None]:
        """Wrapper for `generate_stream` with error handling.

        Catches ValueError, CUDA errors, and general exceptions,
        logging the errors and returning a standard error message.

        Args:
            params (Dict[str, Any]): Same parameters as `generate_stream`.

        Yields:
            bytes: JSON-encoded messages with either partial output or error text.
        """
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            LOGGER.exception(e)
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            LOGGER.exception(e)
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            LOGGER.exception(e)
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


def heart_beat_worker(controller: Any) -> None:
    """Continuously sends heartbeat signals to the controller at regular intervals.

    Args:
        controller (Any): The controller object that manages the worker and provides the `send_heart_beat` method.

    Note:
        This function never returns; it is intended to run in a separate thread or process to maintain worker liveness.
    """
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def release_model_semaphore(fn: Optional[Callable[[], None]] = None) -> None:
    """Release the global model semaphore and optionally execute a callback.

    This function is used to signal that a model slot has become available
    and can also call an optional function, e.g., sending a heartbeat
    to indicate the worker is alive.

    Args:
        fn (Optional[Callable[[], None]]): A callback function to be called
            after releasing the semaphore. Defaults to None.
    """
    model_semaphore.release()
    if fn is not None:
        fn()


app = FastAPI()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request) -> StreamingResponse:
    """Handle streaming generation requests from clients.

    This endpoint acquires the model semaphore to control concurrency,
    triggers a heartbeat to indicate worker liveness, and returns a streaming
    response with generated data. The semaphore is released asynchronously
    when streaming finishes.

    Args:
        request (Request): The incoming FastAPI request containing JSON parameters.

    Returns:
        StreamingResponse: A streaming response yielding generation outputs.
    """
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(opts.limit_model_concurrency)

    await model_semaphore.acquire()
    model_worker.send_heart_beat()

    generator = model_worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=model_worker.send_heart_beat))

    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request) -> Dict:
    """Return the current status of the worker.

    Args:
        request (Request): The incoming FastAPI request.

    Returns:
        Dict: A dictionary representing the worker's current status.
    """
    return model_worker.get_status()


if __name__ == "__main__":
    opts = get_opts()

    model_worker = ModelWorker(
        opts.controller,
        opts.worker,
        str(uuid.uuid4())[:6],
        opts.model_path,
        opts.model_base,
        opts.load_in_4bit,
        opts.load_in_8bit,
        opts.attn_implementation,
    )
    uvicorn.run(
        app,
        host=opts.host,
        port=opts.port,
        log_level="info",
    )
