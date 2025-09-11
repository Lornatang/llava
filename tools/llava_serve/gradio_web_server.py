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
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import requests

from llava.constants import GET_QUERY_PARAMS_FROM_WINDOW, MODERATION_MSG, SERVER_ERROR_MSG
from llava.conversation import default_conversation, conv_templates
from llava.utils.events import LOGGER
from llava.utils.ops import violates_moderation

HEADERS = {"User-Agent": "LLaVA Web Client"}
TITLE_MARKDOWN = """
# LLaVA: Large Language and Vision Assistant
"""

BLOCK_CSS = """

#buttons button {
    min-width: min(120px,100%);
}

"""
NO_CHANGE_BTN = gr.update()
ENABLE_BTN = gr.update(interactive=True)
DISABLE_BTN = gr.update(interactive=False)


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to listen on. Defaults to ``127.0.0.1``."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to listen on. Defaults to 7860."
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="http://127.0.0.1:10000",
        help="Controller URL to fetch worker addresses. Defaults to ``http://127.0.0.1:10000``."
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="Maximum number of concurrent requests. Defaults to 10."
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="reload",
        choices=["once", "reload"],
        help="How to load model list. Choices: 'once', 'reload'. Defaults to ``reload``."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing."
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable text moderation for user input."
    )
    return parser.parse_args()


def add_text(
        state: Any,
        text: Union[List[str], str],
        image: Any,
        image_process_mode: Any,
        request: gr.Request
) -> Any:
    """Add user text (and optionally image) to the conversation.

    Handles moderation checks and updates the chatbot messages.

    Args:
        state (Any): Current conversation state.
        text (Union[List[str], str]): User input text.
        image (Any): Optional image input.
        image_process_mode (Any): Selected image preprocessing mode.
        request (gr.Request): Gradio request object.

    Returns:
        Any: Tuple containing updated state, chatbot messages, message string, None, and 5 button states.
    """
    LOGGER.info(f"add_text. IP: {request.client.host}. Length: {len(text)}.")

    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (NO_CHANGE_BTN,) * 5

    if opts.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), MODERATION_MSG, None) + (NO_CHANGE_BTN,) * 5

    text = text[:1536]
    if image is not None:
        text = text[:1200]
        if "<image>" not in text:
            text += "\n<image>"
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (DISABLE_BTN,) * 5


def clear_history(request: gr.Request) -> Any:
    """Clear the conversation history and reset state.

    Args:
        request (gr.Request): Gradio request object.

    Returns:
        Any: Tuple containing reset state, empty chatbot, empty string, None, and 5 disabled buttons.
    """
    LOGGER.info(f"clear_history. IP: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (DISABLE_BTN,) * 5


def downvote_last_response(state: Any, model_selector: str, request: gr.Request) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Register a downvote for the last response.

    Args:
        state (Any): Current conversation state.
        model_selector (str): Name of the model being voted on.
        request (gr.Request): Gradio request object.

    Returns:
        Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Tuple of output values for Gradio: empty string and 3 disabled buttons.
    """
    LOGGER.info(f"downvote_last_response. IP: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (DISABLE_BTN,) * 3


def flag_last_response(state: Any, model_selector: str, request: gr.Request) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Flag the last response for moderation.

    Args:
        state (Any): Current conversation state.
        model_selector (str): Name of the model being voted on.
        request (gr.Request): Gradio request object.

    Returns:
        Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Tuple of output values for Gradio: empty string and 3 disabled buttons.
    """
    LOGGER.info(f"flag_last_response. IP: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (DISABLE_BTN,) * 3


def get_conv_log_file_path() -> Path:
    """Generate a conversation log filename based on the current date.

    Returns:
        Path: log file path.
    """
    today = datetime.now()
    filename = f"{today:%Y-%m-%d}-conv.json"
    return Path(".", filename)


def get_model_list() -> List[str]:
    """Fetch the list of available models from the controller service.

    This function first refreshes all worker states by calling the
    controller's `/refresh_all_workers` endpoint, then retrieves the
    current list of models from `/list_models` endpoint. The list is
    sorted alphabetically before returning.

    Returns:
        List[str]: A sorted list of model names available from the controller.

    Raises:
        AssertionError: If the refresh request fails (status code != 200).
        requests.RequestException: If the network request encounters an error.
        KeyError: If the returned JSON does not contain the "models" key.
    """
    ret = requests.post(f"{opts.controller}/refresh_all_workers")
    assert ret.status_code == 200, "Failed to refresh all workers"

    ret = requests.post(f"{opts.controller}/list_models")
    ret.raise_for_status()
    models = ret.json()["models"]

    models.sort()
    LOGGER.info(f"Models: {models}.")
    return models


def load_demo(url_params: Dict[str, str], request: gr.Request) -> Tuple[Any, Dict[str, Any]]:
    """Initialize the conversation state and update the model dropdown based on URL parameters.

    This function is typically used as the Gradio `load` callback to:
    1. Log the client IP and URL parameters.
    2. Initialize a new conversation state.
    3. Optionally pre-select a model in the dropdown if specified in URL parameters.

    Args:
        url_params (Dict[str, str]): URL query parameters from the browser, e.g., {'model': 'vicuna-13b'}.
        request (gr.Request): Gradio request object containing client information such as IP address.

    Returns:
        Tuple[Any, Dict[str, Any]]:
            - `state`: A copy of the default conversation state.
            - `dropdown_update`: Gradio Dropdown update object, optionally setting `value` and `visible`.
    """
    LOGGER.info(f"load_demo. IP: {request.client.host}. Params: {url_params}.")

    dropdown_update: Dict[str, Any] = {"visible": True}

    model_name = url_params.get("model")
    if model_name in models:
        dropdown_update["value"] = model_name

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request) -> Tuple[Any, Dict[str, Any]]:
    """Reload the model list from the controller and initialize conversation state.

    Args:
        request (gr.Request): Gradio request object containing client information.

    Returns:
        Tuple[Any, dict[str, Any]]:
            - `state`: Initialized conversation state.
            - `dropdown_update`: Dictionary to update the model dropdown choices and selected value.
    """
    models = get_model_list()

    LOGGER.info(
        f"load_demo_refresh_model_list. "
        f"IP: {request.client.host}. "
        f"Found {len(models)} models: {models if len(models) <= 5 else models[:5] + ['...']}",
    )

    state = default_conversation.copy()
    dropdown_update = gr.update(choices=models, value=models[0] if models else None)
    return state, dropdown_update


def regenerate(state: Any, image_process_mode: str, request: gr.Request) -> Any:
    """Regenerate the last assistant response.

    Args:
        state (Any): Current conversation state.
        image_process_mode (str): Selected image preprocessing mode.
        request (gr.Request): Gradio request object.

    Returns:
        Any: Tuple containing updated state, chatbot messages, empty string, None, and 5 disabled buttons.
    """
    LOGGER.info(f"regenerate. IP: {request.client.host}.")

    # Ensure there is at least one assistant message to clear.
    if state.messages and isinstance(state.messages[-1], list) and len(state.messages[-1]) > 1:
        state.messages[-1][-1] = None

    # Update the last human message's image mode if it contains image data.
    if len(state.messages) >= 2:
        prev_human_msg = state.messages[-2]
        if isinstance(prev_human_msg[1], (tuple, list)) and len(prev_human_msg[1]) >= 2:
            prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)

    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (DISABLE_BTN,) * 5


def upvote_last_response(state: Any, model_selector: str, request: gr.Request) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Register an upvote for the last response.

    Args:
        state (Any): Current conversation state.
        model_selector (str): Name of the model being voted on.
        request (gr.Request): Gradio request object.

    Returns:
        Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Tuple of output values for Gradio: empty string and 3 disabled buttons.
    """
    LOGGER.info(f"upvote_last_response. IP: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (DISABLE_BTN,) * 3


def vote_last_response(state: Any, vote_type: str, model_selector: str, request: gr.Request) -> None:
    """Record a vote for the last model response to a log file.

    Args:
        state (Any): The conversation state object, expected to have a `.dict()` method.
        vote_type (str): Type of vote, e.g., "upvote" or "downvote".
        model_selector (str): Name of the model being voted on.
        request (gr.Request): Gradio request object containing client information.
    """
    log_file_path = get_conv_log_file_path()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "tstamp": round(time.time(), 4),
        "type": vote_type,
        "model": model_selector,
        "state": state.dict(),
        "ip": request.client.host,
    }
    with log_file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def http_bot(
        state: Any,
        model_selector: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        request: gr.Request,
) -> Any:
    """Handles asynchronous streaming chat responses for a selected LLM model.

    This function queries a remote worker to generate text (and handle images),
    streams partial outputs to the Gradio frontend, and logs conversation data.

    Args:
        state (Any): The current conversation state object.
        model_selector (str): The name of the selected model.
        temperature (float): Sampling temperature for text generation.
        top_p (float): Top-p sampling probability for text generation.
        max_new_tokens (int): Maximum number of tokens to generate.
        request (gr.Request): Gradio request object for retrieving client info.

    Yields:
        Any: A tuple containing the updated conversation state, the Gradio chatbot representation, and button states.
    """
    LOGGER.info(f"http_bot. IP: {request.client.host}.")
    start_tstamp = time.time()

    # Skip generation if flagged.
    if state.skip_next:
        yield (state, state.to_gradio_chatbot()) + (NO_CHANGE_BTN,) * 5
        return

    # Initialize conversation template if new.
    if len(state.messages) == state.offset + 2:
        if "qwen2.5" in model_selector.lower():
            template_name = "qwen2_5"
        elif "qwen2" in model_selector.lower():
            template_name = "qwen2"
        elif "qwen1.5" in model_selector.lower():
            template_name = "qwen1_5"
        elif "llama" in model_selector.lower():
            template_name = "llama"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address from controller.
    ret = requests.post(opts.controller + "/get_worker_address", json={"model": model_selector})
    worker_address = ret.json()["address"]
    LOGGER.info(f"model_name: {model_selector}, worker_address: {worker_address}.")

    # No available worker.
    if worker_address == "":
        state.messages[-1][-1] = SERVER_ERROR_MSG
        yield state, state.to_gradio_chatbot(), DISABLE_BTN, DISABLE_BTN, DISABLE_BTN, ENABLE_BTN, ENABLE_BTN
        return

    # Construct prompt and save images.
    prompt = state.get_prompt()
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash_str in zip(all_images, all_image_hash):
        t = datetime.now()
        save_dir = Path("results", "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}")
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = Path(save_dir, f"{hash_str}.jpg")
        if not file_path.is_file():
            image.save(file_path)

    # Prepare payload.
    pload = {
        "model": model_selector,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep2,
        "images": f"List of {len(state.get_images())} images: {all_image_hash}.",
    }
    LOGGER.info(f"==== request ====\n{pload}")
    pload["images"] = state.get_images()

    # Show streaming indicator.
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (DISABLE_BTN,) * 5

    # Stream response from worker.
    try:
        response = requests.post(worker_address + "/worker_generate_stream", headers=HEADERS, json=pload, stream=True, timeout=100)
        last_print_time = time.time()
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    if time.time() - last_print_time > 0.05:
                        last_print_time = time.time()
                        yield (state, state.to_gradio_chatbot()) + (DISABLE_BTN,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (DISABLE_BTN, DISABLE_BTN, DISABLE_BTN, ENABLE_BTN, ENABLE_BTN)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:  # noqa.
        state.messages[-1][-1] = SERVER_ERROR_MSG
        yield (state, state.to_gradio_chatbot()) + (DISABLE_BTN, DISABLE_BTN, DISABLE_BTN, ENABLE_BTN, ENABLE_BTN)
        return

    # Remove streaming indicator.
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (ENABLE_BTN,) * 5

    # Log conversation.
    finish_tstamp = time.time()
    LOGGER.info(f"{output}")
    log_file_path = get_conv_log_file_path()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "tstamp": round(finish_tstamp, 4),
        "type": "chat",
        "model": model_selector,
        "start": round(start_tstamp, 4),
        "finish": round(start_tstamp, 4),
        "state": state.dict(),
        "images": all_image_hash,
        "ip": request.client.host,
    }
    with log_file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def build_demo() -> gr.Blocks:
    """Builds and returns the Gradio demo interface for the LLaVA chatbot.

    Returns:
        gr.Blocks: A Gradio Blocks object representing the full chatbot interface.

    The interface contains:
        - Model selector dropdown
        - Textbox for user input
        - Image uploader
        - Preprocessing mode selector for images
        - Example images
        - Sliders for generation parameters (temperature, top_p, max_output_tokens)
        - Chatbot display
        - Action buttons (upvote, downvote, flag, regenerate, clear)
        - Event listeners for all buttons and input submission
        - Dynamic loading of models based on `args.model_list_mode`
    """
    # User input textbox.
    textbox = gr.Textbox(
        placeholder="Enter text and press ENTER",
        show_label=False,
        container=False
    )

    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=BLOCK_CSS) as demo:
        # Stores chatbot state.
        state = gr.State()

        gr.Markdown(TITLE_MARKDOWN)

        with gr.Row():
            # Left column: model selection, image input, parameters.
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        show_label=False,
                        container=False,
                        interactive=True,
                    )

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False,
                )

                gr.Examples(
                    examples=[
                        [f"./assets/0.jpg", "Who and what activity can you see in the snowy scene?"],
                        [f"./assets/1.jpg", "What are the things I should be cautious about when I visit here?"],
                    ],
                    inputs=[imagebox, textbox],
                )

                # Generation parameters accordion.
                with gr.Accordion("Parameters", open=False):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=2048,
                        value=1024,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            # Right column: chatbot and buttons.
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="LLaVA Chatbot",
                    height=550,
                    type="tuples"
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")

                # Buttons: upvote, downvote, flag, regenerate, clear.
                with gr.Row(elem_id="buttons"):
                    upvote_btn = gr.Button(value="Upvote", interactive=False)
                    downvote_btn = gr.Button(value="Downvote", interactive=False)
                    flag_btn = gr.Button(value="Flag", interactive=False)
                    regenerate_btn = gr.Button(value="Regenerate", interactive=False)
                    clear_btn = gr.Button(value="Clear", interactive=False)

        # Hidden JSON for URL parameters.
        url_params = gr.JSON(visible=False)

        # Register button listeners.
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False,
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False,
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False,
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens], [state, chatbot] + btn_list)

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens], [state, chatbot] + btn_list)

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens], [state, chatbot] + btn_list)

        # Load model list depending on mode.
        if opts.model_list_mode == "once":
            demo.load(load_demo, [url_params], [state, model_selector], js=GET_QUERY_PARAMS_FROM_WINDOW, queue=False)
        elif opts.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector], queue=False)
        else:
            raise ValueError(f"Unknown model list mode: {opts.model_list_mode}")

    return demo


if __name__ == "__main__":
    opts = get_opts()

    models = get_model_list()
    demo = build_demo()
    demo.queue(
        api_open=False,
        max_size=opts.concurrency_count,
    ).launch(server_name=opts.host, server_port=opts.port, share=opts.share)
