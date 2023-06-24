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

from typing import Literal


import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import  RaySamples
from nerfstudio.field_components.mlp import MLP


class cIMLEField(nn.Module):
    """Base class for fields."""

    def __init__(
        self, 
        in_cimle_ch: int, 
        num_layers: int,
        out_cimle_ch: int, 
        cimle_type: Literal["concat", "add"] = "concat",
        cimle_activation: Literal["relu", "none"] = "relu"
        ) -> None:
        super().__init__()
        self.cimle_type=cimle_type
        self.in_cimle_ch = in_cimle_ch
        self.out_cimle_ch = out_cimle_ch
        cimle_activation = nn.ReLU() if cimle_activation == "relu" else nn.Identity()
        self.cimle_linear = MLP(in_dim=in_cimle_ch, num_layers=num_layers, layer_width=in_cimle_ch, out_dim=out_cimle_ch, activation=cimle_activation, out_activation=None)
        
    @property
    def in_dim(self):
        return self.in_cimle_ch 

    @property
    def out_dim(self):
        return self.out_cimle_ch 

    def forward(self, ray_samples: RaySamples, original_latent: Tensor) -> Tensor:
        """Returns only the density. Used primarily with the density grid.

        Args:
            ray_samples: raysamples. needs cimleletents in metadata attributes 
            original_latents: the latent to incorporate cimle to
        """
        assert "cimle_latent" in ray_samples.metadata.keys(), "cimle_latent must be one of the key of ray_samples metadata"
        zs = ray_samples.metadata["cimle_latent"]
        assert zs.shape[-1] == self.in_cimle_ch, f"the latent dimension of cimle latent must match with {self.cimle_ch}. Got {zs.shape[-1]} instead"
        cimle_embedding = self.cimle_linear(zs)
        # print(h_table_latents.shape, cimle_embedding.shape)
        sh = original_latent.shape[:-1]
        if self.cimle_type == "concat":
            original_latent = torch.cat([original_latent, cimle_embedding.view(list(sh) + [self.out_cimle_ch])], dim=-1)
        elif self.cimle_type == "add":
            original_latent = original_latent + cimle_embedding.view(list(sh) + [self.out_cimle_ch])
        return original_latent