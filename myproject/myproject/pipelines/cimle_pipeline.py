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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from torch import Tensor
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from jaxtyping import Shaped
from torch.cuda.amp.grad_scaler import GradScaler


from myproject.data.datamanagers.cimle_datamanager import cIMLEDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


def cache_latents(pipeline, step):
    assert hasattr(pipeline.datamanager, "fixed_indices_train_dataloader"), "must set up dataloader that loads training full images!"
    pipeline.eval()
    num_images = len(pipeline.datamanager.fixed_indices_train_dataloader)
    all_z = torch.normal(0.0, 1.0, size=(pipeline.cimle_sample_num, num_images, pipeline.cimle_ch), device=pipeline.device)
    all_losses = torch.zeros((pipeline.cimle_sample_num, num_images), device=pipeline.device)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        first_bundle, _ = next(t for t in pipeline.datamanager.fixed_indices_train_dataloader)
        num_rays = min(pipeline.config.cimle_num_rays_to_test, len(first_bundle)) if pipeline.config.cimle_num_rays_to_test > 0 else len(first_bundle)
        task_outer = progress.add_task("[green]Caching cIMLE latent for all train images...", total=pipeline.cimle_sample_num)
        task_inner = progress.add_task(f"[green] [0/{pipeline.cimle_sample_num}] image loop, {num_rays} rays per image uses ...", total=num_images)
        for n in range(pipeline.cimle_sample_num):
            for i, (camera_ray_bundle, batch) in enumerate(pipeline.datamanager.fixed_indices_train_dataloader):
                # one latent per image for now. 
                img_idx = batch['image_idx']
                z = all_z[n, img_idx]
                height, width = camera_ray_bundle.shape
                perm = torch.randperm(height*width)[:num_rays]
                # print(indices)
                ray_bundle = camera_ray_bundle.flatten()[perm]
                _ = batch.pop("cimle_latent", None)
                batch = {k: v.flatten(end_dim=1)[perm] if isinstance(v, Tensor) else v for k, v in batch.items()}
                ray_bundle.metadata["cimle_latent"] = z.reshape(1, -1).expand(num_rays, -1)
                model_outputs = pipeline.model.get_outputs_for_ray_bundle_chunked(ray_bundle)
                metrics_dict = pipeline.model.get_metrics_dict(model_outputs, batch)
                loss_dict = pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
                all_losses[n, img_idx] = loss_dict['rgb_loss']
                progress.update(task_id=task_inner, completed=i + 1)
            progress.reset(task_inner, description=f"[green][{n + 1}/{pipeline.cimle_sample_num}] image loop, {num_rays} rays per image...")

            progress.update(task_id=task_outer, completed=n + 1)
        # get the min latent
    metrics_dict = {}
    ### Get the best loss and select and z code
    idx_to_take = torch.argmin(all_losses, axis=0).reshape(1, -1, 1).expand(1, -1, pipeline.cimle_ch) # [num_images]
    selected_z = torch.gather(all_z, 0, idx_to_take)[0] # [num_images, cimle_ch]
    pipeline.datamanager.train_dataset.set_cimle_latent(selected_z)
    pipeline.datamanager.train_image_dataloader.recache()
    pipeline.train()

@dataclass
class cIMLEPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: cIMLEPipeline)
    """target class to instantiate"""
    cimle_sample_num: int = 10
    """specifies the number of cimle samples when caching"""
    cimle_cache_interval: int = 2000
    """specifies the frenquency of cimle caching"""
    cimle_ch: int = 32
    """specifies the number of cimle channel dimension"""
    cimle_num_rays_to_test: int = 200 * 200
    """speficies number of rays to test when caching cimle latents"""


class cIMLEPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """
    config: cIMLEPipelineConfig
    datamanager: cIMLEDataManager
    cimle_sample_num: int
    cimle_ch: int
    
    def __init__(
        self,
        config: cIMLEPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        assert isinstance(
            self.datamanager, cIMLEDataManager
        ), "cIMLEPipeline only works with MyVanillaDataManager."
        assert isinstance(
            self.config, cIMLEPipelineConfig
        ), "cIMLEPipeline only works with cIMLEPipelineConfig."
        
        self.cimle_sample_num=config.cimle_sample_num
        self.cimle_ch=config.cimle_ch
        self.cimle_cache_interval=config.cimle_cache_interval

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        super().load_pipeline(loaded_state, step)
        cache_latents(self, step)
    
    
    
    @profiler.time_function
    @torch.no_grad()
    def get_eval_loss_dict(self, step: int, z: Optional[Shaped[Tensor, "num_rays, cimle_ch"]] = None, num_samples: int=1) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        num_latents = torch.unique(batch['indices'][:, 0]).shape[0]
        if z is None:
            z = torch.normal(0.0, 1.0, size=(num_samples, num_latents, self.cimle_ch), device=self.device)
        scattered_z = torch.gather(z, 1, batch['indices'][:, :1].reshape(1, -1, 1).expand(num_samples, -1, self.cimle_ch).to(self.device))
        assert scattered_z.shape[1] == len(ray_bundle)
        assert scattered_z.shape[0] == num_samples
        all_loss_dict = []
        all_metrics_dict = []
        all_model_outputs = []
        for i in range(num_samples):
            ray_bundle.metadata["cimle_latent"] = scattered_z[i]
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
            all_loss_dict.append(loss_dict)
            all_metrics_dict.append(metrics_dict)
            all_model_outputs.append(model_outputs)
        self.train()
        return all_model_outputs, all_loss_dict, all_metrics_dict
    
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, 
                                          step: int, 
                                          z: Optional[Shaped[Tensor, "num_rays, cimle_ch"]] = None,
                                          num_samples: int=1
                                          ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
            z: optional cimle latent used for evaluation 
            num_samples: number of cimle samples to query. 
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        if z is None:
            z = torch.normal(0.0, 1.0, size=(num_samples, 1, 1, self.cimle_ch), device=self.device)
        height, width = camera_ray_bundle.shape
        all_metrics_dicts = []
        all_images_dicts = []
        for i in range(num_samples):
            camera_ray_bundle.metadata["cimle_latent"] = z[i].reshape(1, 1, -1).expand(height, width, -1)
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "image_idx" not in metrics_dict
            metrics_dict["image_idx"] = image_idx
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = len(camera_ray_bundle)
            all_metrics_dicts.append(metrics_dict)
            all_images_dicts.append(images_dict)
        self.train()
        return all_metrics_dicts, all_images_dicts
    
    @profiler.time_function
    def get_average_eval_image_metrics(self, 
                                       step: Optional[int] = None, 
                                       z: Optional[Shaped[Tensor, "num_rays, cimle_ch"]] = None,
                                       num_samples: int=1
                                       ) -> Dict[str, Any]:
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_lists = [[] for _ in range(num_samples)]
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            task_inner = progress.add_task(f"[green]Evaluating latent sample for image 0...", total=num_samples)
            for n, (camera_ray_bundle, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
                # time this the following line
                z = torch.normal(0.0, 1.0, size=(num_samples, 1, self.cimle_ch), device=self.device)
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                for i in range(num_samples):
                    camera_ray_bundle.metadata["cimle_latent"] = z[i:i+1].expand(height, width, -1)
                    inner_start = time()
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                    metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                    assert "num_rays_per_sec" not in metrics_dict
                    metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                    fps_str = "fps"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / num_rays
                    metrics_dict_lists[i].append(metrics_dict)
                    progress.update(task_inner, completed=i + 1)
                
                progress.reset(task_inner, description=f"[green]Evaluating latent sample for image {n + 1}...")
                    
                progress.update(task, completed=n + 1)
        # average the metrics list
        metrics_dicts = []
        for metrics_dict_list in metrics_dict_lists:
            metrics_dict = {}
            for key in metrics_dict_list[0].keys():
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
            metrics_dicts.append(metrics_dict)
            
        self.train()
        return metrics_dicts
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        
        
            
        # get cimle callback 
        cimle_caching_callback = TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], cache_latents, 
                                          update_every_num_iters=self.cimle_cache_interval, args=[self])
        
        callbacks = callbacks + [cimle_caching_callback]
        return callbacks
        
        
