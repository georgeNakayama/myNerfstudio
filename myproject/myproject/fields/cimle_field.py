# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base class for the graphs.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type, Literal


import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.field_heads import FieldHeadNames


class cIMLEField(nn.Module):
    """Base class for fields."""

    def __init__(
        self, 
        in_cimle_ch: int, 
        out_cimle_ch: int, 
        cimle_type: Literal["concat", "add"] = "concat",
        cimle_pretrain: bool = False
        ) -> None:
        super().__init__()
        self.cimle_type=cimle_type
        self.in_cimle_ch = in_cimle_ch
        self.out_cimle_ch = out_cimle_ch
        self.cimle_pretrain=cimle_pretrain
        self.cimle_linear = nn.Sequential(nn.Linear(self.in_cimle_ch, self.out_cimle_ch), nn.ReLU())
        
    @property
    def in_dim(self):
        return self.in_cimle_ch 

    @property
    def out_dim(self):
        return self.out_cimle_ch 
    
    @property
    def is_pretraining(self):
        return self.cimle_pretrain

    def forward(self, ray_samples: RaySamples, original_latent: Tensor) -> Tensor:
        """Returns only the density. Used primarily with the density grid.

        Args:
            ray_samples: raysamples. needs cimleletents in metadata attributes 
            original_latents: the latent to incorporate cimle to
        """
        assert ray_samples.metadata is not None and "cimle_latent" in ray_samples.metadata.keys(), "cimle_latent must be one of the key of ray_samples metadata"
        zs = ray_samples.metadata["cimle_latent"]
        assert zs.shape[-1] == self.in_cimle_ch, f"the latent dimension of cimle latent must match with {self.in_cimle_ch}. Got {zs.shape[-1]} instead"
        # print(h_table_latents.shape, cimle_embedding.shape)
        sh = original_latent.shape[:-1]
        cimle_embedding = self.cimle_linear(zs) if not self.cimle_pretrain else torch.zeros(list(sh) + [self.out_cimle_ch]).to(original_latent.device)
        if self.cimle_type == "concat":
            original_latent = torch.cat([original_latent, cimle_embedding.view(list(sh) + [self.out_cimle_ch])], dim=-1)
        elif self.cimle_type == "add":
            original_latent = original_latent + cimle_embedding.view(list(sh) + [self.out_cimle_ch])
        return original_latent