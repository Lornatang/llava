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
from threading import Thread
from typing import Any, Dict, Callable, Optional

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
from llava.utils.ops import KeywordsStoppingCriteria, load_image_from_base64, process_images, pretty_print_semaphore, tokenizer_image_token

worker_id = str(uuid.uuid4())[:6]
global_counter = 0

model_semaphore = None


class ModelWorker:
    def __init__(self, controller_addr, worker_addr, worker_id, no_register, model_path, load_8bit, load_4bit):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_path = model_path
        if model_path.endswith("/"):
            model_path = model_path[:-1]

        LOGGER.info(f"Loading the model {self.model_path} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained(model_path, load_8bit, load_4bit)
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        LOGGER.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {"worker_name": self.worker_addr, "check_heart_beat": True, "worker_status": self.get_status()}
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        LOGGER.info(
            f"Send heart beat. Models: {[self.model_path]}. " f"Semaphore: {pretty_print_semaphore(model_semaphore)}. " f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={"worker_name": self.worker_addr, "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                LOGGER.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    @staticmethod
    def get_queue_length() -> int:
        if model_semaphore is None:
            return 0
        return args.limit_model_concurrency - model_semaphore._value + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [Path(self.model_path).name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
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

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, "max_position_embeddings", 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps(
                {"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(
            target=model.generate,
            kwargs=dict(
                inputs=input_ids,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                stopping_criteria=[stopping_criteria],
                use_cache=True,
                **image_args,
            ),
        )
        thread.start()

        start_time = time.time()
        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[: -len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

        end_time = time.time()

        new_generated = generated_text[len(ori_prompt):]
        new_generated_tokens = tokenizer(new_generated).input_ids
        token_per_second = len(new_generated_tokens) / (end_time - start_time)
        print(f"token_per_second: {token_per_second}")

    def generate_stream_gate(self, params):
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
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    await model_semaphore.acquire()
    worker.send_heart_beat()

    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))

    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request) -> Dict:
    """Return the current status of the worker.

    Args:
        request (Request): The incoming FastAPI request.

    Returns:
        Dict: A dictionary representing the worker's current status.
    """
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=40000)
    parser.add_argument("--worker-address", type=str, default="http://127.0.0.1:40000")
    parser.add_argument("--controller-address", type=str, default="http://127.0.0.1:10000")
    parser.add_argument("--model-path", type=str, default="./results/stage2_finetune_ov/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-stage2_ov_data")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.load_8bit,
        args.load_4bit,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
