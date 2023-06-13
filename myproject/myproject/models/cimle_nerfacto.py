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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional

import numpy as np
import torch
import torch.nn as nn 
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from collections import defaultdict
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from myproject.models.base_cimle_model import cIMLEModel, cIMLEModelConfig

from myproject.fields.cimle_nerfacto_field import cIMLENerfactoField

@dataclass
class cIMLENerfactoModelConfig(cIMLEModelConfig, NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: cIMLENerfactoModel)
    
    color_cimle: bool=True
    """whether apply cimle only to color channel"""


class cIMLENerfactoModel(cIMLEModel, NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: cIMLENerfactoModelConfig
    

    def populate_modules(self):
        """Set the fields and modules."""
        NerfactoModel.populate_modules(self)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = cIMLENerfactoField(
            self.scene_box.aabb,
            color_cimle=self.config.color_cimle,
            cimle_ch=self.config.cimle_ch,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        
    def get_outputs(self, ray_bundle: RayBundle):
        ray_bundle = cIMLEModel.get_outputs(self, ray_bundle)
        return NerfactoModel.get_outputs(self, ray_bundle)


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups.update(self.field.get_param_group())
        return param_groups
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = cIMLEModel.get_training_callbacks(self, training_callback_attributes)
        callbacks += NerfactoModel.get_training_callbacks(self, training_callback_attributes)

        return callbacks
    
    def get_cimle_loss(self, outputs, batch) -> torch.Tensor:
        """Obtain the cimle loss based on which caching is performed. 
            Inputs are the same as get loss dict. 
        Returns:
            cimle loss is returned
        """
        image = batch["image"].to(self.device)
        return self.rgb_loss(image, outputs["rgb"])
    
    
    def get_image_metrics_and_images(
        self, all_outputs: List[Dict[str, torch.Tensor]], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        return cIMLEModel.get_image_metrics_and_images_loop(self, NerfactoModel.get_image_metrics_and_images, all_outputs, batch)