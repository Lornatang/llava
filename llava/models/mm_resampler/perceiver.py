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
from typing import Any, Optional, Dict

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn


def feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    """Create a feed-forward neural network block.

    Args:
        dim (int): Input and output dimension.
        mult (int, optional): Multiplication factor for hidden layer. Defaults to 4.

    Returns:
        nn.Sequential: The feed-forward block.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        """Initialize PerceiverAttention.

        Args:
            dim (int): Input dimension.
            dim_head (int, optional): Dimension per attention head. Defaults to 64.
            heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass for PerceiverAttention.

        Args:
            x (torch.Tensor): Image features of shape (b, T, n1, D).
            latents (torch.Tensor): Latent features of shape (b, T, n2, D).

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=self.heads)
        q = q * self.scale

        # attention.
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)
        return self.to_out(out)


class PerceiverResamplerModule(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            depth: int = 6,
            dim_head: int = 64,
            heads: int = 8,
            num_latents: int = 64,
            max_num_media: Optional[int] = None,
            max_num_frames: Optional[int] = None,
            ff_mult: int = 4,
    ) -> None:
        """Initialize PerceiverResamplerModule.

        Args:
            dim (int): Input dimension.
            depth (int, optional): Number of layers. Defaults to 6.
            dim_head (int, optional): Dimension per attention head. Defaults to 64.
            heads (int, optional): Number of attention heads. Defaults to 8.
            num_latents (int, optional): Number of latent vectors. Defaults to 64.
            max_num_media (Optional[int], optional): Max number of media. Defaults to None.
            max_num_frames (Optional[int], optional): Max number of frames. Defaults to None.
            ff_mult (int, optional): Feed-forward multiplier. Defaults to 4.
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if max_num_frames is not None else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if max_num_media is not None else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        feed_forward(dim=dim, mult=ff_mult) if ff_mult > 0 else nn.Identity(),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for PerceiverResamplerModule.

        Args:
            x (torch.Tensor): Image features of shape (b, T, F, v, D).

        Returns:
            torch.Tensor: Output tensor of shape (b, T, n, D), where n is num_latents.
        """
        b, time_stamp, frame, dim = x.shape[:4]

        # frame and media time embeddings.
        if self.frame_embs is not None:
            frame_embs = repeat(self.frame_embs[:frame], "F d -> b T F v d", b=b, T=time_stamp, v=dim)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions.
        if self.media_time_embs is not None:
            x = x + self.media_time_embs[:time_stamp]

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=time_stamp)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class PerceiverResampler(nn.Module):
    def __init__(self, model_args: Any, vision_tower: Any) -> None:
        """Initialize PerceiverResampler.

        Args:
            model_args (Any): Model arguments containing perceiver configuration.
            vision_tower (Any): Vision tower with hidden_size attribute.
        """
        super().__init__()
        self.depth = model_args.mm_perceiver_depth
        self.num_latents = model_args.mm_perceiver_latents
        self.ff_mult = model_args.mm_perceiver_ff_mult
        self.pretrained = model_args.mm_perceiver_pretrained

        self.perceiver = PerceiverResamplerModule(dim=vision_tower.hidden_size, depth=self.depth, num_latents=self.num_latents, ff_mult=self.ff_mult)

        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained))

    def forward(self, image_features: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass for PerceiverResampler.

        Args:
            image_features (torch.Tensor): Image features tensor.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor after resampling.
        """
        return self.perceiver(image_features[:, None, None]).squeeze(1)

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration dictionary for PerceiverResampler.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        return {
            "mm_resampler_type": "perceiver",
            "mm_perceiver_depth": self.depth,
            "mm_perceiver_latents": self.num_latents,
            "mm_perceiver_ff_mult": self.ff_mult,
            "mm_perceiver_pretrained": self.pretrained,
        }
