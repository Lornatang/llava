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
import dataclasses
import enum
import json
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from llava.constants import SERVER_ERROR_MSG
from llava.utils.events import LOGGER

CONTROLLER_HEART_BEAT_EXPIRATION = 30


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="Host to listen on. Defaults to ``127.0.0.1``.",
    )
    parser.add_argument(
        "--port",
        default=10000,
        type=int,
        help="Port to listen on. Defaults to 10000.",
    )
    parser.add_argument(
        "--dispatch-method",
        default="shortest_queue",
        type=str,
        choices=["lottery", "shortest_queue"],
        help="Method to dispatch requests to workers. Defaults to ``shortest_queue``.",
    )
    return parser.parse_args()


class DispatchMethod(Enum):
    """Specifies the dispatch method for assigning tasks to workers.

    Attributes:
        LOTTERY: A probabilistic method where workers with higher speed
            are more likely to be chosen.
        SHORTEST_QUEUE: A greedy method that assigns tasks to the worker
            with the fewest tasks in its queue.
    """
    LOTTERY = enum.auto()
    SHORTEST_QUEUE = enum.auto()

    @classmethod
    def from_str(cls, name: str) -> "DispatchMethod":
        """Converts a string to its corresponding DispatchMethod enum.

        Args:
            name: The string representation of the dispatch method (e.g., "lottery",
                "shortest_queue").

        Returns:
            The DispatchMethod enum instance.

        Raises:
            ValueError: If the provided `name` does not correspond to
                a valid dispatch method.
        """
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method: {name}")


@dataclasses.dataclass
class WorkerInfo:
    """Stores information about a worker in the system.

    Attributes:
        model_names: A list of model names that the worker can serve.
        speed: The processing speed of the worker, used for dispatching.
        queue_length: The number of pending tasks in the worker's queue.
        check_heart_beat: A boolean indicating whether to check the worker's
            heartbeat.
        last_heart_beat: A string representing the timestamp of the last
            received heartbeat from the worker.
    """
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str


class Controller:
    """The central component that manages workers and dispatches tasks.

    This class handles the registration of workers, monitors their status via heartbeats, and determines the best worker for a given task based on the chosen dispatch method.
    """

    def __init__(self, dispatch_method: str):
        """Initializes the controller with a dispatch method.

        Args:
            dispatch_method: The name of the dispatch method to use (e.g., "lottery", "shortest_queue").
        """
        self.worker_info: Dict[str, WorkerInfo] = {}
        self.dispatch_method: DispatchMethod = DispatchMethod.from_str(dispatch_method)
        self.heart_beat_thread: threading.Thread = threading.Thread(
            target=heart_beat_controller,
            args=(self,),
            daemon=True,
        )
        self.heart_beat_thread.start()

    def register_worker(self, worker_name: str, check_heart_beat: bool, worker_status: Optional[Dict[str, Any]]) -> bool:
        """Registers or updates a worker's information.

        If the worker is new, it adds it to the system. If it already exists, it updates its status. It also fetches the worker's status if not provided.

        Args:
            worker_name (str): The unique name of the worker.
            check_heart_beat (bool): A boolean indicating if the controller should monitor this worker's heartbeat.
            worker_status (Optional[Dict[str, Any]]): An optional dictionary containing the worker's status (model_names, speed, queue_length). If `None`, the status is fetched from the worker.

        Returns:
            bool: A boolean indicating whether the registration was successful.
        """
        if worker_name not in self.worker_info:
            LOGGER.info(f"Register a new worker: {worker_name}.")
        else:
            LOGGER.info(f"Register an existing worker: {worker_name}.")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            model_names=worker_status["model_names"],
            speed=worker_status["speed"],
            queue_length=worker_status["queue_length"],
            check_heart_beat=check_heart_beat,
            last_heart_beat=str(time.time()),
        )

        LOGGER.info(f"Register done: {worker_name}, {worker_status}.")
        return True

    @staticmethod
    def get_worker_status(worker_name: str) -> Optional[Dict[str, Any]]:
        """Fetches the status of a worker by making an HTTP request.

        Args:
            worker_name (str): The name (and address) of the worker to query.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the worker's status if successful, otherwise `None`.
        """
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            LOGGER.exception(f"Get status fails: {worker_name}, {e}.")
            return None

        if r.status_code != 200:
            LOGGER.error(f"Get status fails: {worker_name}, {r.status_code}.")
            return None

        return r.json()

    def remove_worker(self, worker_name: str) -> None:
        """Removes a worker from the controller's registry.

        Args:
            worker_name (str): The name of the worker to remove.
        """
        if worker_name in self.worker_info:
            del self.worker_info[worker_name]
            LOGGER.info(f"Removed worker: {worker_name}")

    def refresh_all_workers(self) -> None:
        """Refreshes the status of all registered workers."""
        old_worker_info = dict(self.worker_info)
        self.worker_info = {}

        for work_name, work_info in old_worker_info.items():
            if not self.register_worker(work_name, work_info.check_heart_beat, None):
                LOGGER.info(f"Remove stale worker: {work_name}")

    def list_models(self) -> List[str]:
        """Lists all unique model names available across all workers.

        Returns:
            List[str]: A list of all unique model names.
        """
        model_names = set()
        for _, work_info in self.worker_info.items():
            model_names.update(work_info.model_names)
        return list(model_names)

    def get_worker_address(self, model_name: str) -> str:
        """Selects a worker to handle a request for a specific model based on the dispatch method.

        Args:
            model_name (str): The name of the model requested by the client.

        Returns:
            str: The address of the selected worker, or an empty string if no suitable worker is found.
        """
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for work_name, work_info in self.worker_info.items():
                if model_name in work_info.model_names:
                    worker_names.append(work_name)
                    worker_speeds.append(work_info.speed)

            if not worker_names:
                return ""

            worker_speeds_np = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds_np)
            if norm < 1e-4:
                return ""

            worker_probabilities = worker_speeds_np / norm

            if True:  # A fast path to directly return a worker address without additional checks.
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_probabilities)
                worker_name = worker_names[pt]
                return worker_name

            # An alternative path to check worker status before returning, though the fast path is used here.
            while True:
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_probabilities)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    return worker_name
                else:
                    self.remove_worker(worker_name)
                    worker_probabilities[pt] = 0
                    norm = np.sum(worker_probabilities)
                    if norm < 1e-4:
                        return ""
                    worker_probabilities = worker_probabilities / norm
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for work_name, work_info in self.worker_info.items():
                if model_name in work_info.model_names:
                    worker_names.append(work_name)
                    worker_qlen.append(work_info.queue_length / work_info.speed)

            if not worker_names:
                return ""

            min_index = np.argmin(worker_qlen)
            work_name = worker_names[min_index]
            self.worker_info[work_name].queue_length += 1
            LOGGER.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {work_name}.")
            return work_name

    def receive_heart_beat(self, worker_name: str, queue_length: int) -> bool:
        """Updates a worker's heartbeat and queue length.

        Args:
            worker_name (str): The name of the worker sending the heartbeat.
            queue_length (int): The current queue length of the worker.

        Returns:
            bool: A boolean indicating if the worker still exists in the controller's registry.
        """
        if worker_name not in self.worker_info:
            LOGGER.info(f"Receive unknown heart beat. {worker_name}.")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        LOGGER.info(f"Receive from {worker_name} heart beat.")
        return True

    def remove_stable_workers_by_expiration(self) -> None:
        """Removes workers whose heartbeats have expired."""
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, work_info in self.worker_info.items():
            if work_info.check_heart_beat and float(work_info.last_heart_beat) < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_generate_stream(self, params: Dict[str, Any]) -> Any:
        """Handles streaming text generation requests by dispatching them to a worker.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for the text generation request, including the "model" name.

        Yields:
            Any: Chunks of the streaming response from the worker.
        """
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            LOGGER.info(f"no worker: {params['model']}")
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        try:
            response = requests.post(worker_addr + "/worker_generate_stream", json=params, stream=True, timeout=5)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            LOGGER.exception(f"worker timeout: {worker_addr}")
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 3,
            }
            yield json.dumps(ret).encode() + b"\0"

    def worker_api_get_status(self) -> Dict[str, Any]:
        """Aggregates and returns the status of all registered workers.

        Returns:
            Dict[str, Any]: A dictionary containing the combined status of all workers, including a list of available model names, total speed, and total queue length.
        """
        model_names = set()
        total_speed = 0
        total_queue_length = 0

        for work_name in self.worker_info:
            worker_status = self.get_worker_status(work_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                total_speed += worker_status["speed"]
                total_queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": total_speed,
            "queue_length": total_queue_length,
        }


def heart_beat_controller(controller: Any):
    """Manages the heartbeats of workers and removes unresponsive ones.

    This function runs in a loop, periodically checking for expired worker
    heartbeats and removing inactive workers from the controller.

    Args:
        controller (Any): The controller object that manages the workers. It is expected to have a `remove_stable_workers_by_expiration` method.
    """
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request) -> None:
    """Registers a new worker with the controller.

    Args:
        request (Request): The FastAPI Request object containing worker registration data.
            The JSON body is expected to contain "worker_name", "check_heart_beat" and an optional "worker_status".
    """
    data: Dict[str, Any] = await request.json()
    controller.register_worker(data["worker_name"], data["check_heart_beat"], data.get("worker_status", None))


@app.post("/refresh_all_workers")
async def refresh_all_workers() -> None:
    """Refreshes the status of all workers registered with the controller."""
    controller.refresh_all_workers()


@app.post("/list_models")
async def list_models() -> Dict[str, List[str]]:
    """Retrieves a list of all models available from the workers.

    Returns:
        Dict[str, List[str]]: A dictionary containing a list of model names.
    """
    models: List[str] = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request) -> Dict[str, str]:
    """Gets the address of a worker serving a specific model.

    Args:
        request (Request): The FastAPI Request object. The JSON body is expected to contain the "model" name.

    Returns:
        Dict[str, str]: A dictionary containing the worker's address.
    """
    data: Dict[str, str] = await request.json()
    addr: str = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request) -> Dict[str, bool]:
    """Receives a heartbeat from a worker and updates its status.

    Args:
        request (Request): The FastAPI Request object. The JSON body is expected to contain the "worker_name" and "queue_length".

    Returns:
        Dict[str, bool]: A dictionary with a boolean value indicating if the worker still exists.
    """
    data: Dict[str, Any] = await request.json()
    exist: bool = controller.receive_heart_beat(data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request) -> StreamingResponse:
    """Initiates a streaming text generation from a worker.

    Args:
        request (Request): The FastAPI Request object containing generation parameters in its JSON body.

    Returns:
        StreamingResponse: A StreamingResponse object that yields generated text chunks.
    """
    params: Dict[str, Any] = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request) -> Any:
    """Retrieves the status of the workers.

    Args:
        request (Request): The FastAPI Request object.

    Returns:
        Any: A dictionary or other data structure containing the status of the workers.
    """
    return controller.worker_api_get_status()


if __name__ == "__main__":
    opts = get_opts()

    controller = Controller(opts.dispatch_method)
    uvicorn.run(
        app,
        host=opts.host,
        port=opts.port,
        log_level="info",
    )
