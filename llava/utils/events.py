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
import logging
import os

__all__ = [
    "LOGGER",
]


def configure_logging(name: str = None) -> logging.Logger:
    """Configure the logger.

    Args:
        name (str, optional): The name of the logger. Defaults to None.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    # Get the RANK environment variable, default to -1 if not set.
    rank = int(os.getenv("RANK", -1))

    # Set the logging level.
    logger_level = logging.INFO if (rank in (-1, 0)) else logging.WARNING
    logger.setLevel(logger_level)

    # Set the logging format.
    format_string = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - llava - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Create a stream handler for logging to the console.
    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(format_string)
    cli_handler.setLevel(logger_level)
    logger.handlers.clear()
    logger.addHandler(cli_handler)
    logger.propagate = False

    return logger


LOGGER = configure_logging(__name__)
