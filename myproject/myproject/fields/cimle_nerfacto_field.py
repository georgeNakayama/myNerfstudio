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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Literal, Optional, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.fields.nerfacto_field import NerfactoField
from myproject.fields.cimle_field import cIMLEField
from nerfstudio.utils.rich_utils import CONSOLE

class cIMLENerfactoField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_pred_normals: bool = False,
        
        # hash grid specs
        base_res: int = 16,
        features_per_level: int = 2,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        
        # base mlp related
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        
        # color channel related
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        
        # transient embedding related
        num_layers_transient: int = 2,
        hidden_dim_transient: int = 64,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        
        # semantic related
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        
        # appearance embedding related
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,
        
        # cIMLE related
        cimle_injection_type: Literal["add", "concat", "cat_linear"] = "concat",
        cimle_ch: int=32,
        num_layers_cimle: int=1,
        color_cimle: bool=False,
        cimle_activation: Literal["relu", "none"] = "relu",
        cimle_pretrain: bool = False,
        use_cimle_grid: bool = False,
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            base_res=base_res, 
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            implementation=implementation
            )
        
        self.color_cimle=color_cimle
        self.use_cimle_grid=use_cimle_grid
        color_in_dim = self.direction_encoding.get_out_dim() + self.geo_feat_dim
        self.cimle = cIMLEField(
            cimle_ch,
            color_in_dim if color_cimle else self.mlp_base_grid.get_out_dim(),
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

        if color_cimle:
            self.mlp_head = MLP(
                in_dim=self.cimle.out_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )
        else:
            self.mlp_base_mlp = MLP(
                in_dim=self.cimle.out_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

    def get_param_group(self) -> Dict[str, List[Parameter]]:
        param_groups = {'fields':[], 'cimle':[]}
        for name, params in self.named_parameters():
            if name.split(".")[0] == 'cimle':
                param_groups['cimle'].append(params)
            else:
                param_groups['fields'].append(params)
        if self.cimle.is_pretraining:
            del param_groups["cimle"]
        return param_groups


    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, metric_dict = self.get_density(ray_samples)
        else:
            density, density_embedding, metric_dict = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs["field.latent_diff"] = metric_dict["latent_diff"]  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Computes and returns the densities."""
        if self.color_cimle:
            return super().get_density(ray_samples)
        
        # else 
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h_table_latents = self.mlp_base_grid(positions_flat)
        
        # density cimle
        h_table_latents, latent_diff = self.cimle(ray_samples, h_table_latents, positions=positions_flat)

        h = self.mlp_base_mlp(h_table_latents).view(*ray_samples.frustums.shape, -1)
        
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out, {"latent_diff": latent_diff}
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        if not self.color_cimle:
            return super().get_outputs(ray_samples, density_embedding)
        
        # else
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        


        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)
        h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                ],
                dim=-1,
            )
        h_table_latents = self.cimle(ray_samples, h)
        
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
