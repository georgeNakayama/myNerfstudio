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

from typing import Literal, Optional


import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import  RaySamples
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import HashEncoding

class cIMLEField(nn.Module):
    """Base class for fields."""

    def __init__(
        self, 
        in_cimle_ch: int, 
        feature_in_dim: int,
        num_layers: int,
        cimle_type: Literal["concat", "add", "cat_linear", "add_linear", "cat_linear_res", "add_linear_res"] = "concat",
        cimle_pretrain: bool = False,
        cimle_act: Literal["relu", "none"] = "relu",
        use_cimle_grid: bool = False,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        ) -> None:
        super().__init__()
        self.use_cimle_grid = use_cimle_grid
        if self.use_cimle_grid:
            self.cimle_grid = HashEncoding(            
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
            no_grad=True
        )
        self.cimle_type=cimle_type
        in_cimle_ch = self.cimle_grid.get_out_dim() if self.use_cimle_grid else in_cimle_ch
        self.in_cimle_ch = in_cimle_ch + int(cimle_type in ["cat_linear", "cat_linear_res"]) * feature_in_dim
        self.out_cimle_ch = in_cimle_ch if cimle_type == "concat" else feature_in_dim
        cimle_activation = nn.ReLU() if cimle_act == "relu" else nn.Identity()
        self.cimle_pretrain = cimle_pretrain
        self.cimle_linear = MLP(in_dim=self.in_cimle_ch, num_layers=num_layers, layer_width=self.out_cimle_ch, out_dim=self.out_cimle_ch, activation=cimle_activation, out_activation=None)
        
        
        
    @property
    def in_dim(self):
        return self.in_cimle_ch 

    @property
    def out_dim(self):
        return self.out_cimle_ch 
    
    @property
    def is_pretraining(self):
        return self.cimle_pretrain

    def forward(self, ray_samples: RaySamples, original_latent: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Returns only the density. Used primarily with the density grid.

        Args:
            ray_samples: raysamples. needs cimleletents in metadata attributes 
            original_latents: the latent to incorporate cimle to
        """
        _original_latent = original_latent
        if self.cimle_pretrain:
            return original_latent
        
        if self.use_cimle_grid:
            assert positions is not None
            zs = self.cimle_grid(positions).to(torch.float32)
        else: 
            assert ray_samples.metadata is not None and "cimle_latent" in ray_samples.metadata.keys(), "cimle_latent must be one of the key of ray_samples metadata"
            zs = ray_samples.metadata["cimle_latent"]
            assert zs.shape[-1] == self.in_cimle_ch, f"the latent dimension of cimle latent must match with {self.in_cimle_ch}. Got {zs.shape[-1]} instead"
            # print(h_table_latents.shape, cimle_embedding.shape)
            sh = original_latent.shape[:-1]
            zs = zs.reshape(list(sh) + [self.in_cimle_ch])
        
        if self.cimle_type == "concat":
            cimle_embedding = self.cimle_linear(zs) if not self.cimle_pretrain else torch.zeros(list(sh) + [self.out_cimle_ch]).to(original_latent.device)
            original_latent = torch.cat([original_latent, cimle_embedding], dim=-1)
        elif self.cimle_type == "cat_linear":
            cimle_embedding = torch.cat([original_latent, zs], dim=-1)
            original_latent = self.cimle_linear(cimle_embedding)
        elif self.cimle_type == "cat_linear_res":
            cimle_embedding = torch.cat([original_latent, zs], dim=-1)
            original_latent = original_latent + self.cimle_linear(cimle_embedding)
        elif self.cimle_type == "add_linear_res":
            cimle_embedding = original_latent + zs
            original_latent = original_latent + self.cimle_linear(cimle_embedding)
        elif self.cimle_type == "add_linear":
            cimle_embedding = original_latent + zs
            original_latent = self.cimle_linear(cimle_embedding)
        elif self.cimle_type == "denoise":
            cimle_embedding = original_latent + zs
            original_latent = cimle_embedding + self.cimle_linear(cimle_embedding)
        elif self.cimle_type == "add":
            cimle_embedding = self.cimle_linear(zs) if not self.cimle_pretrain else torch.zeros(list(sh) + [self.out_cimle_ch]).to(original_latent.device)
            original_latent = original_latent + cimle_embedding
        diff = 0.0
        if _original_latent.shape == original_latent.shape:
            diff = (_original_latent - original_latent) ** 2
            diff = diff.mean(-1).reshape(ray_samples.shape)
        return original_latent, diff