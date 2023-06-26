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
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, Literal

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import ModelConfig, Model
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.data.datamanagers.base_datamanager import DataManager, VanillaDataManager
from nerfstudio.utils import colormaps


# Model related configs
@dataclass
class cIMLEModelConfig(ModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: cIMLEModel)
    """target class to instantiate"""
    cimle_sample_num: int = 10
    """specifies the number of cimle samples when caching"""
    cimle_cache_interval: int = 2000
    """specifies the frenquency of cimle caching"""
    cimle_ch: int = 32
    """specifies the number of cimle channel dimension"""
    num_layers_cimle: int = 1
    """specifies the number of layers in cimle linear"""
    cimle_activation: Literal["relu", "none"] = "relu"
    """specifies the activation used in cimle linear layers"""
    cimle_type: Literal["concat", "add"] = "concat"
    """specifies the method to integrate cimle latents"""
    cimle_num_rays_to_test: int = 200 * 200
    """speficies number of rays to test when caching cimle latents"""
    cimle_ensemble_num: int=5
    """specifies the number of cimle samples when calculating uncertainty"""


class cIMLEModel(Model):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: cIMLEModelConfig

    def __init__(
        self,
        config: cIMLEModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.cimle_sample_num=self.config.cimle_sample_num
        self.cimle_ch=self.config.cimle_ch
        self.cimle_cache_interval=self.config.cimle_cache_interval
        self.cimle_num_rays_to_test=self.config.cimle_num_rays_to_test
        self.cimle_ensemble_num=self.config.cimle_ensemble_num
        self.cimle_type=self.config.cimle_type
        self.cimle_latents: nn.Embedding = nn.Embedding.from_pretrained(torch.zeros([num_train_data, self.cimle_ch]), freeze=True)


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        
        def cache_latents(datamanager: VanillaDataManager, step):
            assert datamanager.fixed_indices_train_dataloader is not None, "must set up dataloader that loads training full images!"
            self.eval()
            num_images = len(datamanager.fixed_indices_train_dataloader)
            all_z = torch.normal(0.0, 1.0, size=(self.cimle_sample_num, num_images, self.cimle_ch), device=self.device)
            all_losses = torch.zeros((self.cimle_sample_num, num_images), device=self.device)
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                first_bundle, _ = next(t for t in datamanager.fixed_indices_train_dataloader)
                num_rays = min(self.cimle_num_rays_to_test, len(first_bundle)) if self.cimle_num_rays_to_test > 0 else len(first_bundle)
                task_outer = progress.add_task("[green]Caching cIMLE latent for all train images...", total=self.cimle_sample_num)
                task_inner = progress.add_task(f"[green] [0/{self.cimle_sample_num}] image loop, {num_rays} rays per image uses ...", total=num_images)
                for n in range(self.cimle_sample_num):
                    for i, (camera_ray_bundle, batch) in enumerate(datamanager.fixed_indices_train_dataloader):
                        # one latent per image for now. 
                        img_idx = batch['image_idx']
                        z = all_z[n, img_idx]
                        height, width = camera_ray_bundle.shape
                        perm = torch.randperm(height*width)[:num_rays]
                        # print(indices)
                        ray_bundle = camera_ray_bundle.flatten()[perm]
                        batch.pop("cimle_latent", None)
                        batch = {k: v.flatten(end_dim=1)[perm] if isinstance(v, Tensor) else v for k, v in batch.items()}
                        ray_bundle.metadata["cimle_latent"] = z.reshape(1, -1).expand(num_rays, -1)
                        model_outputs = self.get_outputs_for_ray_bundle_chunked(ray_bundle, sample_latent=False)
                        loss = self.get_cimle_loss(model_outputs, batch)
                        all_losses[n, img_idx] = loss
                        progress.update(task_id=task_inner, completed=i + 1)
                    progress.reset(task_inner, description=f"[green][{n + 1}/{self.cimle_sample_num}] image loop, {num_rays} rays per image...")

                    progress.update(task_id=task_outer, completed=n + 1)
                # get the min latent
            ### Get the best loss and select and z code
            idx_to_take = torch.argmin(all_losses, dim=0).reshape(1, -1, 1).expand(1, -1, self.cimle_ch) # [num_images]
            selected_z = torch.gather(all_z, 0, idx_to_take)[0] # [num_images, cimle_ch]
            self.set_cimle_latents(nn.Embedding.from_pretrained(selected_z, freeze=True).to(self.device))
            
            self.train()
            
        assert training_callback_attributes.pipeline is not None
        cimle_caching_callback = TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], cache_latents, update_every_num_iters=self.cimle_cache_interval, args=[training_callback_attributes.pipeline.datamanager])
        return callbacks + [cimle_caching_callback]

    @abstractmethod
    def get_cimle_loss(self, outputs, batch) -> torch.Tensor:
        """Obtain the cimle loss based on which caching is performed. 
            Inputs are the same as get loss dict. 
        Returns:
            cimle loss is returned
        """

    def set_cimle_latents(self, latents: nn.Embedding) -> None:
        self.cimle_latents = latents

    def forward(self, ray_bundle: RayBundle, sample_latent: bool=False, **kwargs) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        # prepare cimle
        if ray_bundle.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_bundle.camera_indices.squeeze()
        if sample_latent:
            max_cam_ind = int(camera_indices.max())
            samples = torch.randn([max_cam_ind + 1, self.cimle_ch]).to(self.device)
            cimle_latents = samples[camera_indices.flatten()]
            ray_bundle.metadata["cimle_latent"] = cimle_latents
        elif "cimle_latent" not in ray_bundle.metadata:
            assert self.cimle_latents is not None, "must initialize cimle latents"
            cimle_latents = self.cimle_latents(camera_indices)
            ray_bundle.metadata["cimle_latent"] = cimle_latents
            

        return super().forward(ray_bundle, sample_latent=sample_latent, **kwargs)


    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> List[Dict[str, torch.Tensor]]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        cimle_latents = torch.randn(self.cimle_ensemble_num, self.cimle_ch).to(self.device)
        all_outputs = []
        for n in range(self.cimle_ensemble_num):
            _z = cimle_latents[n]
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            camera_ray_bundle.metadata["cimle_latent"] = _z.reshape(1, 1, self.cimle_ch).expand(image_height, image_width, -1)
            outputs_dict: Dict[str, torch.Tensor] = {}
            outputs_lists = defaultdict(list)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle)
                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    outputs_lists[output_name].append(output)
            for output_name, outputs_list in outputs_lists.items():
                outputs_dict[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            
            all_outputs.append(outputs_dict)
        
        return all_outputs
    
    def get_image_metrics_and_images_loop(
        self, 
        eval_func: Callable[..., Tuple[Dict[str, float], Dict[str, Tensor], Dict[str, Tensor]]], 
        all_outputs: List[Dict[str, Tensor]], 
        batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        all_metrics_dict = {}
        all_images_dict = {}
        images_list_dict = defaultdict(list)
        metrics_list_dict = defaultdict(list)
        ground_truth_dict: Dict[str, Tensor] = {}
        for n, outputs in enumerate(all_outputs):
            metrics_dict, images_dict, ground_truth_dict = eval_func(outputs, batch)
            for k, v in metrics_dict.items():
                metrics_list_dict[k].append(v)
                all_metrics_dict[f"{k}/sample_{n}"] = v
            for k, v in images_dict.items():
                images_list_dict[k].append(v)

        avg_metrics_dict = {f"{k}": torch.mean(torch.tensor(v)) for k, v in metrics_list_dict.items()}
        var_metrics_dict = {f"{k}/variance": torch.var(torch.tensor(v)) for k, v in metrics_list_dict.items()}
        all_metrics_dict.update(avg_metrics_dict)
        all_metrics_dict.update(var_metrics_dict)
        
        concat_images_dict = {f"{k}/samples": torch.concat(v, dim=1) for k, v in images_list_dict.items()}
        avg_images_dict = {f"{k}/mean": torch.mean(torch.stack(v, dim=0), dim=0) for k, v in images_list_dict.items()}
        var_images_dict: Dict[str, Tensor] = {}
        for k, v in images_list_dict.items():
            var_images_dict[f"{k}/variance"] = torch.var(torch.stack(v, dim=0), dim=0)
            if len(var_images_dict[f"{k}/variance"].shape) == 3:
                var_images_dict[f"{k}/variance"] = var_images_dict[f"{k}/variance"].mean(-1, keepdim=True)
            all_metrics_dict[f"{k}/max_var"] = var_images_dict[f"{k}/variance"].max()
            all_metrics_dict[f"{k}/mean_var"] = var_images_dict[f"{k}/variance"].mean()
                
        all_images_dict.update(avg_images_dict)
        all_images_dict.update(var_images_dict)
        all_images_dict.update(concat_images_dict)
        all_images_dict.update({f"{k}/ground_truth": v for k, v in ground_truth_dict.items()})
                
        return all_metrics_dict, all_images_dict
    