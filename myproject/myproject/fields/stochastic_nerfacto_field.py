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


from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
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
from myproject.model_components.losses import neg_nll_loss, gaussian_entropy, log_normal_entropy, logistic_normal_entropy, log_normal_log_prob, logistic_normal_log_prob


class StochasticNerfactoField(NerfactoField):
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
        num_layers: int = 2,
        hidden_dim: int = 64,
        stochastic_samples: int = 32,
        geo_feat_dim: int = 14,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 0,
        transient_embedding_dim: int = 16,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_gaussian_ent: bool = False,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        compute_kl: bool = False,
        deterministic_color: bool = False,
        init_glob_density: float = -10,
        global_density_std: float = 3,
        global_rgb_std: float = 1
    ) -> None:
        super().__init__(
            aabb, 
            num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            use_transient_embedding=False,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            implementation=implementation,
            )

        self.stochastic_samples=stochastic_samples
        self.use_gaussian_ent=use_gaussian_ent
        self.compute_kl=compute_kl
        self.deterministic_color=deterministic_color
        self.global_rgb_mean = nn.Parameter(torch.zeros(3).to(torch.float32), requires_grad=True)
        self.global_rgb_std = global_rgb_std
        self.global_rgb_logvar = math.log(self.global_rgb_std) * 2
        self.global_density_std = global_density_std
        self.global_density_logvar = math.log(self.global_density_std) * 2
        self.global_density_mean = nn.Parameter(torch.ones(1).to(torch.float32) * init_glob_density, requires_grad=False)
        # global_rgb_mean.requires_grad = True
        # self.register_buffer("global_rgb_mean", global_rgb_mean)
        # self.register_buffer("global_density_mean", global_density_mean)
        
        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=2 + self.geo_feat_dim, # mean + std + feat_dim
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )


        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim())

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3 + 3, # mean + std
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )


    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, compute_global_entropy: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_dict, density_embedding = self.get_density(ray_samples)
        else:
            density, density_dict, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs.update(density_dict)  # type: ignore

        if compute_global_entropy:
            res = 128
            out_dict = self.get_global_entropy(res=res, device=density_embedding.device)
            field_outputs.update(out_dict)

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    

    def get_global_entropy(self, res: int, device:str) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Computes and returns the densities."""
        out_dict = dict()
        positions = SceneBox.sample_uniform_points(res, torch.tensor([[0, 0, 0], [1,1, 1]]).to(device).to(torch.float32)) # [512, 512, 512, 3]
        uniform_direction = torch.randn([res, res, res, 3]).to(device)
        uniform_direction = F.normalize(uniform_direction, dim=-1)
        directions_flat = get_normalized_directions(uniform_direction).view(-1, 3)
        d = self.direction_encoding(directions_flat)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base_grid(positions_flat)
        h = self.mlp_base_mlp(h).view(res, res, res, -1)
        
        density_before_activate_mean, density_before_activate_logvar, base_mlp_out = torch.split(h, [1, 1, self.geo_feat_dim], dim=-1)
        if self.use_gaussian_ent:
            density_entropy = gaussian_entropy(density_before_activate_logvar, dim=1) 
        else:
            density_entropy = log_normal_entropy(density_before_activate_mean, density_before_activate_logvar, dim=1) 
        out_dict["density_entropy"] = density_entropy
        if self.compute_kl:
            samples = density_before_activate_mean + torch.randn([res, res, res, 1]).to(positions) * trunc_exp(0.5 * density_before_activate_logvar)
            log_prob_p = log_normal_log_prob(samples, self.global_density_mean.to(positions), self.global_density_std * self.global_density_std, dim=1)
            density_kl = -1.0 * (density_entropy + log_prob_p)
            out_dict["density_kl"] = density_kl
            
        if self.deterministic_color:
            return out_dict
        
        d = self.direction_encoding(directions_flat)
        if self.appearance_embedding_dim > 0:
            embedded_appearance = torch.ones(
                        (res, res, res, self.appearance_embedding_dim), device=device
                    ) * self.embedding_appearance.mean(dim=0)
            h = torch.cat(
                [
                    d,
                    base_mlp_out.view(-1, self.geo_feat_dim),
                    embedded_appearance.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            h = torch.cat(
                [d, base_mlp_out.view(-1, self.geo_feat_dim)],
                dim=-1,
            )
        rgb_out = self.mlp_head(h).view(res, res, res, -1).to(uniform_direction)
        rgb_mean, rgb_logvar = torch.split(rgb_out, 3, dim=-1)

            
        if self.use_gaussian_ent:
            rgb_entropy = gaussian_entropy(rgb_logvar, dim=3)[..., None]
        else:
            rgb_entropy = logistic_normal_entropy(rgb_mean, rgb_logvar, dim=3)[..., None]
        
        out_dict["rgb_entropy"] = rgb_entropy
        if self.compute_kl:
            q_samples = rgb_mean + torch.randn([res, res, res, 3]).to(positions) * trunc_exp(0.5 * rgb_logvar)
            p_samples = self.global_rgb_mean + torch.randn([res, res, res, 3]).to(positions) * self.global_rgb_std
            log_prob_p = logistic_normal_log_prob(p_samples, self.global_rgb_mean, self.global_rgb_std * self.global_rgb_std, dim=3)
            log_prob_q = logistic_normal_log_prob(q_samples, rgb_mean, trunc_exp(rgb_logvar), dim=3)
            logr = log_prob_p - log_prob_q
            rgb_kl = trunc_exp(logr) - 1 - logr
            out_dict["rgb_kl"] = rgb_kl.mean(-2)
            
        return out_dict

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Computes and returns the densities."""
        out_dict = {}
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
        h = self.mlp_base_grid(positions_flat)
        h = self.mlp_base_mlp(h).view(*ray_samples.shape, -1)
        density_before_activation_mean, density_before_activation_logvar, base_mlp_out = torch.split(h, [1, 1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation_mean

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density_before_activation_std = trunc_exp(0.5 * density_before_activation_logvar.to(positions))
        noise = torch.randn(list(ray_samples.shape) + [self.stochastic_samples, 1]).to(density_before_activation_mean)
        density_before_activation_samples = density_before_activation_mean.unsqueeze(-2) + noise * density_before_activation_std.unsqueeze(-2)
        density_samples = trunc_exp(density_before_activation_samples)
        density_samples = density_samples * selector[..., None, None]
        density_samples = density_samples.transpose(-2, -3)
        if self.use_gaussian_ent:
            density_entropy = gaussian_entropy(density_before_activation_logvar, dim=1)[..., None]
        else:
            density_entropy = log_normal_entropy(density_before_activation_mean, density_before_activation_logvar, dim=1)[..., None]
        
        out_dict["density_entropy"] = density_entropy
        
        if self.compute_kl:
            global_samples = self.global_density_mean.reshape(1, 1, 1).to(positions) + torch.randn(list(ray_samples.shape) + [1]).to(positions) * self.global_density_std
            log_prob_p = log_normal_log_prob(density_before_activation_samples, self.global_density_mean.to(positions), self.global_density_std*self.global_density_std, dim=1).mean(-2)
            density_kl = -1.0 * (density_entropy + log_prob_p)
            out_dict["density_kl"] = density_kl
        
        out_dict["density_std"] = density_before_activation_std
        out_dict["density_mean"] = density_before_activation_mean
        density_before_activation_mean = trunc_exp(density_before_activation_mean)
        return density_samples, out_dict, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.appearance_embedding_dim > 0:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )
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

        
        if self.appearance_embedding_dim > 0:
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_appearance.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            h = torch.cat(
                [d, density_embedding.view(-1, self.geo_feat_dim)],
                dim=-1,
            )
        rgb_out = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        rgb_mean, rgb_logvar = torch.split(rgb_out, 3, dim=-1)
        rgb_std = trunc_exp(rgb_logvar * 0.5).to(rgb_mean)
        if self.deterministic_color:
            noise = 0.0
        else:
            noise = torch.randn(list(outputs_shape) + [self.stochastic_samples, 3]).to(rgb_mean)
        pre_sigmoid_rgb_samples = rgb_mean.unsqueeze(-2) + noise * rgb_std.unsqueeze(-2)
        rgb_samples = F.sigmoid(pre_sigmoid_rgb_samples)
        if self.deterministic_color:
            outputs[FieldHeadNames.RGB] = rgb_samples.transpose(-2, -3)
            return outputs
        if self.use_gaussian_ent:
            rgb_entropy = gaussian_entropy(rgb_logvar, dim=3)[..., None]
        else:
            rgb_entropy = logistic_normal_entropy(rgb_mean, rgb_logvar, samples=self.stochastic_samples, dim=3)[..., None]
        
        if self.compute_kl:
            q_samples = pre_sigmoid_rgb_samples
            p_samples = self.global_rgb_mean + torch.randn(list(outputs_shape) + [self.stochastic_samples, 3]).to(rgb_mean) * self.global_rgb_std
            log_prob_p = logistic_normal_log_prob(p_samples, self.global_rgb_mean, self.global_rgb_std * self.global_rgb_std, dim=3)
            log_prob_q = logistic_normal_log_prob(q_samples, rgb_mean.unsqueeze(-2), trunc_exp(rgb_logvar.unsqueeze(-2)), dim=3)
            logr = log_prob_p - log_prob_q
            rgb_kl = trunc_exp(logr) - 1 - logr
            outputs["rgb_kl"] = rgb_kl.mean(-2)
            
        rgb_mean = F.sigmoid(rgb_mean)
        outputs.update({FieldHeadNames.RGB: rgb_samples.transpose(-2, -3), "rgb_std": rgb_std, "rgb_mean": rgb_mean, "rgb_entropy": rgb_entropy})

        return outputs
