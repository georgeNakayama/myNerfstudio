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
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, Literal, Mapping 
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor
from pathlib import Path
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import ModelConfig, Model
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.data.datamanagers.base_datamanager import  VanillaDataManager
from nerfstudio.utils import model_context_manager

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
    cimle_injection_type: Literal["concat", "add"] = "concat"
    """specifies the method to integrate cimle latents"""
    cimle_num_rays_to_test: int = -1
    """speficies number of rays to test when caching cimle latents"""
    cimle_ensemble_num: int=5
    """specifies the number of cimle samples when calculating uncertainty"""
    pretrained_path: Optional[Path]=None
    """Specifies the path to pretrained model."""
    cimle_pretrain: bool=False 
    """Specifies whether it is pretraining"""
    cimle_type: Literal["per_view", "per_scene"] = "per_view"
    """Specifies the type of cIMLE sampling to be used"""
    


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
        self.cimle_injection_type=self.config.cimle_injection_type
        self.cimle_latents: nn.Embedding = nn.Embedding.from_pretrained(torch.zeros([num_train_data, self.cimle_ch]), freeze=True)
        self.cimle_cached_samples: Optional[Tensor] = None

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        
        def cache_latents(datamanager: VanillaDataManager, step):
            assert datamanager.fixed_indices_train_dataloader is not None, "must set up dataloader that loads training full images!"
            self.eval()
            num_images = len(datamanager.fixed_indices_train_dataloader)
            all_z = self.sample_cimle_latent(self.cimle_sample_num, num_images)
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
            
            self.set_cimle_latents_from_loss(all_z, all_losses)
            
            self.train()
            
        assert training_callback_attributes.pipeline is not None
        cimle_caching_callback = TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], cache_latents, update_every_num_iters=self.cimle_cache_interval, args=[training_callback_attributes.pipeline.datamanager])
        
        
        
        
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
        
        if not self.config.cimle_pretrain:
            return callbacks + [cimle_caching_callback, load_cb]
        return callbacks

    def clear_eval_ctx(self):
        self.eval_ctx = False
        self.cimle_cached_samples = None

    def sample_cimle_latent(self, cimle_sample_num: int, image_num: int=1) -> Float[Tensor, "sample_num image_num cimle_ch"]:
        
        if self.config.cimle_type == "per_scene":
            z = torch.normal(0.0, 1.0, size=(cimle_sample_num, 1, self.cimle_ch), device=self.device).repeat_interleave(image_num, dim=1)
            if self.eval_ctx and self.cimle_cached_samples is None:
                self.cimle_cached_samples = z
            
            if self.cimle_cached_samples is not None and self.cimle_cached_samples.shape[:2] == (cimle_sample_num, image_num):
                return self.cimle_cached_samples
            
            return z
        if self.config.cimle_type == "per_view":
            return torch.normal(0.0, 1.0, size=(cimle_sample_num, image_num, self.cimle_ch), device=self.device)
        
        CONSOLE.print(f"cIMLE type {self.config.cimle_type} is not implemented!")
        raise NotImplementedError
            
    def sample_cimle_latent_single(self) -> Float[Tensor, "1 cimle_ch"]:
        if self.config.cimle_type == "per_scene":
            z = torch.normal(0.0, 1.0, size=(1, self.cimle_ch), device=self.device)
            
            if self.eval_ctx and self.cimle_cached_samples is None:
                self.cimle_cached_samples = z
            
            if self.cimle_cached_samples is not None:
                return self.cimle_cached_samples
            
        elif self.config.cimle_type == "per_view":
            z = torch.normal(0.0, 1.0, size=(1, self.cimle_ch), device=self.device)
        else:
            CONSOLE.print(f"cIMLE type {self.config.cimle_type} is not implemented!")
            raise NotImplementedError
        return z
    
    def set_cimle_latents_from_loss(self, all_z: Float[Tensor, "num_cimle_sample num_images cimle_ch"], all_losses: Float[Tensor, "num_cimle_sample num_images"]) -> None:
        num_images = all_z.shape[1]
        if self.config.cimle_type == "per_scene":
            all_losses = all_losses.mean(1)
            idx_to_take = torch.argmin(all_losses, dim=0).reshape(1, 1, 1).expand(1, num_images, self.cimle_ch) # [num_images]
        elif self.config.cimle_type == "per_view":
            idx_to_take = torch.argmin(all_losses, dim=0).reshape(1, -1, 1).expand(1, -1, self.cimle_ch) # [num_images]
        else:
            raise NotImplementedError
        selected_z = torch.gather(all_z, 0, idx_to_take)[0] # [num_images, cimle_ch]
        self.cimle_latents =  nn.Embedding.from_pretrained(selected_z, freeze=True).to(self.device)

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
    

    @abstractmethod
    def get_cimle_loss(self, outputs, batch) -> torch.Tensor:
        """Obtain the cimle loss based on which caching is performed. 
            Inputs are the same as get loss dict. 
        Returns:
            cimle loss is returned
        """


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
            samples = self.sample_cimle_latent(1, max_cam_ind + 1)[0]
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
        cimle_latents = self.sample_cimle_latent(self.cimle_ensemble_num).reshape(self.cimle_ensemble_num, self.cimle_ch)
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
    