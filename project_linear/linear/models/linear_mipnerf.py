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
Implementation of mip-NeRF.
"""
from __future__ import annotations

from typing import Type, Literal
from dataclasses import dataclass, field

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.utils import colors
from linear.model_components.renderers import LinearRGBRenderer
from linear.model_components.ray_samplers import LinearPDFSampler
from nerfstudio.model_components.ray_samplers import UniformSampler, PDFSampler
from nerfstudio.fields.vanilla_nerf_field import NeRFField


@dataclass
class LinearMipNerfModelConfig(VanillaModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: LinearMipNerfModel)
    """target class to instantiate"""
    color_mode : Literal["midpoint", "left"] = "midpoint"
    farcolorfix: bool = False
    include_original: bool = False
    use_same_field: bool = True
    



class LinearMipNerfModel(MipNerfModel):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    config: LinearMipNerfModelConfig

    def __init__(
        self,
        config: LinearMipNerfModelConfig,
        **kwargs,
    ) -> None:
        if config.num_importance_samples <= 0:
            config.loss_coefficients["rgb_loss_fine"] = 0.0
            config.loss_coefficients["rgb_loss_coarse"] = 1.0
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        if self.config.num_importance_samples > 0:
            if not self.config.use_same_field:
                self.fine_field = NeRFField(
                    position_encoding=self.position_encoding, direction_encoding=self.direction_encoding, use_integrated_encoding=True
                )
            self.sampler_pdf = LinearPDFSampler(num_samples=self.config.num_importance_samples, include_original=self.config.include_original)

        # renderers
        self.renderer_rgb = LinearRGBRenderer(background_color=colors.WHITE, color_mode=self.config.color_mode, farcolorfix=self.config.farcolorfix)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if not self.config.use_same_field:
            param_groups['fields'].extend(list(self.fine_field.parameters()))
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, **kwargs):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform: RaySamples
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        mid_points = (ray_samples_uniform.frustums.ends + ray_samples_uniform.frustums.starts) / 2.0
        spacing_mid_points = (ray_samples_uniform.spacing_ends + ray_samples_uniform.spacing_starts) / 2.0
        ray_samples_uniform_shifted = ray_bundle.get_ray_samples(
            bin_starts=mid_points[:, :-1], 
            bin_ends=mid_points[:, 1:], 
            spacing_starts=spacing_mid_points[:, :-1], 
            spacing_ends=spacing_mid_points[:, 1:], 
            spacing_to_euclidean_fn=ray_samples_uniform.spacing_to_euclidean_fn,
            sample_method="piecewise_linear")
        weights_coarse, densities_coarse, transmittance_coarse = ray_samples_uniform_shifted.get_weights_linear(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        
        # pdf sampling
        if self.config.num_importance_samples > 0:
            ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform_shifted, weights_coarse, densities_coarse, transmittance_coarse)

            # Second pass:
            if not self.config.use_same_field:
                field_outputs_fine = self.fine_field.forward(ray_samples_pdf)
            else:
                field_outputs_fine = self.field.forward(ray_samples_pdf)
            weights_fine, _, _ = ray_samples_pdf.get_weights_linear(field_outputs_fine[FieldHeadNames.DENSITY])
            rgb_fine = self.renderer_rgb(
                rgb=field_outputs_fine[FieldHeadNames.RGB],
                weights=weights_fine,
            )
            accumulation_fine = self.renderer_accumulation(weights_fine)
            depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        else:
            rgb_fine = rgb_coarse
            accumulation_fine = accumulation_coarse
            depth_fine = depth_coarse
            
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs