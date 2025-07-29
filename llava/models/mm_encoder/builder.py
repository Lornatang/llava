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
from pathlib import Path
from typing import Dict

from .clip_encoder import CLIPVisionTower

__all__ = [
    "build_vision_tower",
]


def build_vision_tower(vision_tower_cfg: Dict, **kwargs) -> CLIPVisionTower:
    """Builds a vision tower (encoder) instance based on the given configuration.

    Args:
        vision_tower_cfg (Dict): Configuration dictionary for the vision tower.

    Returns:
        CLIPVisionTower: CLIPVisionTower or CLIPVisionTowerS2: An instance of the selected vision tower.

    Raises:
        ValueError: If the vision tower type is unknown or unsupported.
    """
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if any([
        Path(vision_tower).exists(),
        vision_tower.startswith("openai"),
        vision_tower.startswith("laion"),
        "ShareGPT4V" in vision_tower
    ]):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
