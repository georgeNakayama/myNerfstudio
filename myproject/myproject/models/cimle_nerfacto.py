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

from myproject.fields.cimle_nerfacto_field import cIMLENerfactoField

@dataclass
class cIMLENerfactoModelConfig(cIMLEModelConfig, NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: cIMLENerfactoModel)
    
    color_cimle: bool=True
    """whether apply cimle only to color channel"""
    pretrained_path: Optional[Path]=None
    """Specifies the path to pretrained model."""
    cimle_pretrain: bool=False 
    """Specifies whether it is pretraining"""


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
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            implementation=self.config.implementation,
            cimle_type=self.config.cimle_type,
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
        return NerfactoModel.get_outputs(self, ray_bundle)


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups.update(self.field.get_param_group())
        return param_groups
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        

        def load_from_pretrain(step):
            if self.config.pretrained_path is None:
                CONSOLE.print("Pretrained model NOT loaded!~!")
                return 
            load_path = self.config.pretrained_path
            if not load_path.is_file():
                CONSOLE.print(f"Provided pretrained path {load_path} is invalid! Starting from scratch instead!~!")
                return 
            CONSOLE.print(f"Loading Nerfstudio pretrained model from {load_path}...")
            state_dict: Dict[str, torch.Tensor] = torch.load(load_path, map_location="cpu")["pipeline"]
            is_ddp_model_state = True
            model_state = {}
            for key, value in state_dict.items():
                if key.startswith("_model."):
                    # remove the "_model." prefix from key
                    model_state[key[len("_model.") :]] = value
                    # make sure that the "module." prefix comes from DDP,
                    # rather than an attribute of the model named "module"
                    if not key.startswith("_model.module."):
                        is_ddp_model_state = False
            # remove "module." prefix added by DDP
            if is_ddp_model_state:
                model_state = {key[len("module.") :]: value for key, value in model_state.items()}
            
            self.load_state_dict_from_pretrained(model_state)
            CONSOLE.print(f"Finished loading Nerfstudio pretrained model from {load_path}...")
        
        load_cb = TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN], load_from_pretrain)
        
        callbacks = NerfactoModel.get_training_callbacks(self, training_callback_attributes)
        if not self.config.cimle_pretrain:
            callbacks += cIMLEModel.get_training_callbacks(self, training_callback_attributes)
            callbacks += [load_cb]
            
        return callbacks
    
    def get_cimle_loss(self, outputs, batch) -> torch.Tensor:
        """Obtain the cimle loss based on which caching is performed. 
            Inputs are the same as get loss dict. 
        Returns:
            cimle loss is returned
        """
        image = batch["image"].to(self.device)
        return self.rgb_loss(image, outputs["rgb"])
    
    def load_state_dict_from_pretrained(self, state_dict: Mapping[str, torch.Tensor]):
        new_state_dict: Dict[str, Any] = {}
        model_state_dict: Dict[str, torch.Tensor] = self.state_dict()
        for k in state_dict.keys():
            if "cimle" in k:
                CONSOLE.print(f"Skip loading parameter: {k}")
                continue
            if k in model_state_dict.keys():
                if state_dict[k].shape != model_state_dict[k].shape:
                    CONSOLE.print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    continue
                new_state_dict[k] = state_dict[k]
            else:
                CONSOLE.print(f"Dropping parameter {k}")
        for k in model_state_dict.keys():
            if k not in state_dict.keys():
                CONSOLE.print(f"Layer {k} not loaded!")
        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False)
        for k in missing_keys:
            CONSOLE.print(f"parameter {k} is missing from pretrained model!")
        for k in unexpected_keys:
            CONSOLE.print(f"parameter {k} is unexpected from pretrained model!")
    
    
    
    
    def get_image_metrics_and_images(
        self, all_outputs: List[Dict[str, torch.Tensor]], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        def _get_image_metrics_and_images(
            outputs: Dict[str, torch.Tensor], _batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            image = _batch["image"].to(self.device)
            rgb = outputs["rgb"]
            acc = outputs["accumulation"]
            depth = outputs["depth"]


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
            return metrics_dict, images_dict, {"img": image[0].moveaxis(0, -1)}
        
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