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
from typing import Any, Dict, List, Literal, Mapping, Tuple, Type, Optional
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from myproject.models.base_cimle_model import cIMLEModel, cIMLEModelConfig
from myproject.fields.cimle_density_fields import cIMLEHashMLPDensityField
from myproject.fields.cimle_nerfacto_field import cIMLENerfactoField

@dataclass
class cIMLENerfactoModelConfig(cIMLEModelConfig, NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: cIMLENerfactoModel)
    
    color_cimle: bool=True
    """whether apply cimle only to color channel"""
    use_cimle_in_proposal_networks: bool = False 
    """Specifices whether to inject cimle to proposal networks"""
    


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
            num_layers_cimle=self.config.num_layers_cimle,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            implementation=self.config.implementation,
            cimle_injection_type=self.config.cimle_injection_type,
            cimle_pretrain=self.config.cimle_pretrain
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
            ) if not self.config.use_cimle_in_proposal_networks else \
            cIMLEHashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
                cimle_ch=self.config.cimle_ch,
                cimle_injection_type=self.config.cimle_injection_type,
                num_layers_cimle=self.config.num_layers_cimle,
                cimle_pretrain=self.config.cimle_pretrain
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
                ) if not self.config.use_cimle_in_proposal_networks else \
                cIMLEHashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                    cimle_ch=self.config.cimle_ch,
                    cimle_injection_type=self.config.cimle_injection_type,
                    num_layers_cimle=self.config.num_layers_cimle,
                    cimle_pretrain=self.config.cimle_pretrain
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        
    def get_outputs(self, ray_bundle: RayBundle, return_samples: bool = False, **kwargs):
        return NerfactoModel.get_outputs(self, ray_bundle, return_samples=return_samples)


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups: Dict[str, List[Parameter]] = {}
        prop_network_pgs: Dict[str, List[Parameter]] = defaultdict(list)
        field_pgs: Dict[str, List[Parameter]] = defaultdict(list)
        
        for name, param in self.proposal_networks.named_parameters():
            if "cimle" in name.split("."):
                prop_network_pgs["proposal_networks.cimle"].append(param)
            else:
                prop_network_pgs["proposal_networks"].append(param)
        for name, param in self.field.named_parameters():
            if "cimle" in name.split("."):
                field_pgs["fields.cimle"].append(param)
            else:
                field_pgs["fields"].append(param)
                
        prop_network_pgs.update(field_pgs)
        
        for k, v in prop_network_pgs.items():
            if "cimle" in k.split("."):
                if not self.config.cimle_pretrain:
                    param_groups[k] = v 
            else:
                param_groups[k] = v
                
        return param_groups
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        callbacks = NerfactoModel.get_training_callbacks(self, training_callback_attributes)
        callbacks += cIMLEModel.get_training_callbacks(self, training_callback_attributes)
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
        self, all_outputs: List[Dict[str, Any]], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        def _get_image_metrics_and_images(
            outputs: Dict[str, Any], _batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
            image: torch.Tensor = _batch["image"].to(self.device)
            rgb: torch.Tensor = outputs["rgb"]
            acc: torch.Tensor = outputs["accumulation"]
            depth: torch.Tensor = outputs["depth"]
            weights_list: Optional[Dict[str, torch.Tensor]] = outputs.get("weights_dict", None)
            ray_samples_list: Optional[Dict[str, torch.Tensor]] = outputs.get("ray_samples_dict", None)


            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

            psnr = self.psnr(image, rgb)
            ssim = self.ssim(image, rgb)
            lpips = self.lpips(image, rgb)

            # all of these metrics will be logged as scalars
            metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips":float(lpips)}  # type: ignore
            metrics_dict["max_depth"] = float(torch.max(depth))
            metrics_dict["min_depth"] = float(torch.min(depth))

            images_dict = {"img": rgb[0].moveaxis(0, -1), "accumulation": acc, "depth": depth}
            images_dict.update({f"prop_depth_{i}": outputs[f"prop_depth_{i}"] for i in range(self.config.num_proposal_iterations)})
            metrics_dict.update({f"max_prop_depth_{i}": float(torch.max(outputs[f"prop_depth_{i}"])) for i in range(self.config.num_proposal_iterations)})
            metrics_dict.update({f"min_prop_depth_{i}": float(torch.min(outputs[f"prop_depth_{i}"])) for i in range(self.config.num_proposal_iterations)})
            return metrics_dict, images_dict, {"img": image[0].moveaxis(0, -1)}, weights_list, ray_samples_list
        
        all_metrics_dict, all_images_dict = self.get_image_metrics_and_images_loop(_get_image_metrics_and_images, all_outputs, batch)
        
        original_tags = ['img']
        to_color_map_tags = ["accumulation"]
        to_depth_color_map_tags = ["depth"] + [f"prop_depth_{i}" for i in range(self.config.num_proposal_iterations)]
        clr_map_imgages_dict = {}
        for tag in original_tags:
            keys = [k for k in all_images_dict.keys() if tag in k]
            clr_map_imgages_dict.update({key: all_images_dict[key] if "variance" not in key else colormaps.apply_colormap(all_images_dict[key], colormap_options=colormaps.ColormapOptions(normalize=True)) for key in keys})
        for tag in to_color_map_tags:
            keys = [k for k in all_images_dict.keys() if tag in k]
            clr_map_imgages_dict.update({key: colormaps.apply_colormap(all_images_dict[key]) if "variance" not in key else colormaps.apply_colormap(all_images_dict[key], colormap_options=colormaps.ColormapOptions(normalize=True)) for key in keys})
        for tag in to_depth_color_map_tags:
            keys = [k for k in all_images_dict.keys() if tag in k]
            clr_map_imgages_dict.update({key: colormaps.apply_depth_colormap(
                all_images_dict[key], 
                accumulation=all_images_dict["accumulation/" + key.split("/")[-1]]
            ) if "variance" not in key else colormaps.apply_colormap(all_images_dict[key], colormap_options=colormaps.ColormapOptions(normalize=True)) for key in keys})
        return all_metrics_dict, clr_map_imgages_dict