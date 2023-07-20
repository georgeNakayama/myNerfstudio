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
Proposal network field.
"""


from typing import Literal, Optional, Tuple, Dict

import torch
from torch import Tensor, nn
from jaxtyping import Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.density_fields import HashMLPDensityField
from myproject.fields.cimle_field import cIMLEField
from nerfstudio.cameras.rays import Frustums


class cIMLEHashMLPDensityField(HashMLPDensityField):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        implementation: Literal["tcnn", "torch"] = "torch",
        cimle_ch: int = 32,
        num_layers_cimle: int=1,
        cimle_injection_type: Literal["add", "concat", "cat_linear"]="add",
        cimle_activation: Literal["relu", "none"]="relu",
        cimle_pretrain: bool = False,
        use_cimle_grid: bool = False,
    ) -> None:
        super().__init__(
            aabb,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            spatial_distortion=spatial_distortion,
            use_linear=use_linear,
            num_levels=num_levels,
            max_res=max_res,
            base_res=base_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
            )
        self.cimle = cIMLEField(
            cimle_ch, 
            self.mlp_base_grid.get_out_dim(),
            num_layers_cimle,
            cimle_injection_type,
            cimle_pretrain=cimle_pretrain,
            cimle_act=cimle_activation,
            use_cimle_grid=use_cimle_grid,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size, 
            features_per_level=features_per_level,
            implementation=implementation
            )
        if not self.use_linear:
            self.mlp_base_mlp = MLP(
                in_dim=self.cimle.out_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
        else:
            self.mlp_base_mlp = torch.nn.Linear(self.cimle.out_dim, 1)

    def density_fn(
        self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None, metadata: Optional[Dict[str, Tensor]] = None
    ) -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        del times
        # Need to figure out a better way to describe positions with a ray.
        metadata_dict = dict() 
        if metadata is not None and "cimle_latent" in metadata.keys():
            metadata_dict["cimle_latent"] = metadata["cimle_latent"]
            
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            metadata=metadata_dict
        )
        density, _, metric_dict = self.get_density(ray_samples)
        return density, metric_dict

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        
        x = self.mlp_base_grid(positions_flat).to(positions)
        x, latent_diff = self.cimle(ray_samples, x, positions=positions_flat)
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base_mlp(x).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            density_before_activation = self.mlp_base_mlp(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None, {"latent_diff": latent_diff}

