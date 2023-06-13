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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.vanilla_nerf import VanillaModelConfig, NeRFModel
from nerfstudio.utils import colormaps, colors, misc
from myproject.fields.vanilla_cimle_nerf_field import cIMLENeRFField
from collections import defaultdict
from nerfstudio.model_components.scene_colliders import NearFarCollider

@dataclass
class cIMLEVanillaModelConfig(VanillaModelConfig):
    """cIMLE Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: cIMLENeRFModel)
    """targeting nerf field to initialize"""
    
    cimle_ch: int = 32
    """cimle latent dimension. Need to match with the specification in cIMLEPipeline"""
    
    head_only: bool = True
    """Whether to apply cIMLE only at head mlps"""
    


class cIMLENeRFModel(NeRFModel):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: cIMLEVanillaModelConfig

    def __init__(
        self,
        config: cIMLEVanillaModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = cIMLENeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            cimle_ch=self.config.cimle_ch,
            head_only=self.config.head_only
        )

        self.field_fine = cIMLENeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            cimle_ch=self.config.cimle_ch,
            head_only=self.config.head_only
        )


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        fine_param_groups = self.field_fine.get_param_group()
        coarse_param_groups = self.field_coarse.get_param_group()
        param_groups.update(fine_param_groups)
        for k in param_groups:
            param_groups[k] += coarse_param_groups[k]
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups
