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
from jaxtyping import Float, Shaped
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
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.models.base_model import ModelConfig, Model
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.data.datamanagers.base_datamanager import  VanillaDataManager
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.utils import misc
from myproject.model_components.distribution_metrics import ChamferPairWiseDistance
from myproject.utils.myutils import sync_data
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
    cimle_injection_type: Literal["concat", "add", "cat_linear", "add_linear", "cat_linear_res", "add_linear_res"] = "add"
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
    compute_distribution_diff: bool=False 
    """Specifies whether to compute distribution difference"""
    num_points_sample: int=256
    """Specifices the number of points to sample when computing distribution difference"""
    distribution_metric: Literal["chamfer"] = "chamfer"
    """Specifies what metric to use for distribution variance"""
    cimle_latent_std: float=1.0
    """The std from which to sample cIMLE latents"""
    


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
        self.config.compute_distribution_diff = self.config.compute_distribution_diff and (not self.config.cimle_pretrain) and False
        self.cimle_latents: nn.Embedding = nn.Embedding.from_pretrained(torch.zeros([num_train_data, self.cimle_ch]), freeze=True)
        self.cimle_cached_samples: Optional[Tensor] = None
        self.pdf_sampler = PDFSampler(num_samples=self.config.num_points_sample, train_stratified=False, add_end_bin=True)
        self.distribution_distance: Optional[nn.Module] = None
        if self.config.distribution_metric == "chamfer":
            self.distribution_distance = ChamferPairWiseDistance(reduction=["mean", "max"])


    @torch.no_grad()
    def cache_latents(self, datamanager: VanillaDataManager, step):
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
        
        
    def load_from_pretrain(self, step):
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

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        
        
            
        assert training_callback_attributes.pipeline is not None
        cimle_caching_callback = TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.cache_latents, update_every_num_iters=self.cimle_cache_interval, args=[training_callback_attributes.pipeline.datamanager])
        
        load_cb = TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN], self.load_from_pretrain)
        
        if not self.config.cimle_pretrain:
            return callbacks + [cimle_caching_callback, load_cb]
        return callbacks

    def clear_eval_ctx(self):
        self.eval_ctx = False
        self.cimle_cached_samples = None

    def sample_cimle_latent(self, cimle_sample_num: int, image_num: int=1) -> Float[Tensor, "sample_num image_num cimle_ch"]:
        
        if self.config.cimle_type == "per_scene":
            z = torch.normal(0.0, self.config.cimle_latent_std, size=(cimle_sample_num, 1, self.cimle_ch), device=self.device).repeat_interleave(image_num, dim=1)
            if self.eval_ctx and self.cimle_cached_samples is None:
                self.cimle_cached_samples = z
            
            if self.cimle_cached_samples is not None and self.cimle_cached_samples.shape[:2] == (cimle_sample_num, image_num):
                return self.cimle_cached_samples
            
            return z
        if self.config.cimle_type == "per_view":
            return torch.normal(0.0, self.config.cimle_latent_std, size=(cimle_sample_num, image_num, self.cimle_ch), device=self.device)
        
        CONSOLE.print(f"cIMLE type {self.config.cimle_type} is not implemented!")
        raise NotImplementedError
            
    def sample_cimle_latent_single(self) -> Float[Tensor, "1 cimle_ch"]:
        if self.config.cimle_type == "per_scene":
            z = torch.normal(0.0, self.config.cimle_latent_std, size=(1, self.cimle_ch), device=self.device)
            
            if self.eval_ctx and self.cimle_cached_samples is None:
                self.cimle_cached_samples = z
            
            if self.cimle_cached_samples is not None:
                return self.cimle_cached_samples
            
        elif self.config.cimle_type == "per_view":
            z = torch.normal(0.0, self.config.cimle_latent_std, size=(1, self.cimle_ch), device=self.device)
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
        new_model_state = misc.filter_model_state_dict(self.state_dict(), model_state)
        super().load_state_dict(new_model_state, strict=False)
    

    @abstractmethod
    def get_cimle_loss(self, outputs, batch) -> torch.Tensor:
        """Obtain the cimle loss based on which caching is performed. 
            Inputs are the same as get loss dict. 
        Returns:
            cimle loss is returned
        """


    def forward(self, ray_bundle: RayBundle, sample_latent: bool=False, return_samples:bool=False, **kwargs) -> Dict[str, Union[torch.Tensor, List]]:
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
            

        return super().forward(ray_bundle, return_samples=return_samples, **kwargs)


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
            outputs_dict: Dict[str, Any] = {}
            outputs_lists = defaultdict(list)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle, return_samples=self.config.compute_distribution_diff)
                for output_name, output in outputs.items():  # type: ignore
                    outputs_lists = sync_data(output_name, output, outputs_lists)
            
            for output_name, outputs_list in outputs_lists.items():
                if isinstance(outputs_list, dict):
                    outputs_dict[output_name] = {}
                    for k, v in outputs_list.items():
                        if isinstance(v, list) and all(torch.is_tensor(vv) for vv in v):
                            outputs_dict[output_name][k] = torch.cat(v).view(image_height, image_width, -1)
                        elif isinstance(v, list) and all(isinstance(vv, RaySamples) for vv in v):
                            # outputs_dict[output_name][k] = RaySamples.cat_samples(v).reshape((image_height, image_width, -1)) # OOM error
                            # CONSOLE.print("length of ray samples for an image is:", len(v), k, output_name, v[0].shape)
                            outputs_dict[output_name][k] = v
                elif isinstance(outputs_list, list):
                    if torch.is_tensor(outputs_list[0]):
                        outputs_dict[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
                    elif isinstance(outputs_list[0], list):
                        outputs_dict[output_name] =[torch.cat(out).view(image_height, image_width, -1) for out in outputs_list]  # type: ignore
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            if self.config.compute_distribution_diff:
                outputs_dict["ray_bundle"] = camera_ray_bundle.to("cpu")
                        
            all_outputs.append(outputs_dict)
        
        return all_outputs
    
    
    @torch.no_grad()
    def get_image_metrics_and_images_loop(
        self, 
        eval_func: Callable[..., Tuple[Dict[str, float], Dict[str, Tensor], Dict[str, Tensor], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, RaySamples]]]], 
        all_outputs: List[Dict[str, Any]], 
        batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        all_images_dict = {}
        images_list_dict: Dict[str, List[Tensor]] = defaultdict(list)
        metrics_list_dict: Dict[str, List[float]] = defaultdict(list)
        ground_truth_dict: Dict[str, Tensor] = {}
        weights_list_dict: Dict[str, List[Tensor]] = defaultdict(list)
        samples_list_dict: Dict[str, List[RaySamples]] = defaultdict(list)
        ray_bundle: Optional[RayBundle] = None
        for n, outputs in enumerate(all_outputs):
            metrics_dict, images_dict, ground_truth_dict, weights_dict, samples_dict = eval_func(outputs, batch)
            if self.config.compute_distribution_diff:
                ray_bundle = outputs["ray_bundle"]
            if weights_dict is not None:
                for k, v in weights_dict.items():
                    weights_list_dict[k].append(v)
                    
            if samples_dict is not None:
                for k, v in samples_dict.items():
                    samples_list_dict[k].append(v)
                
            for k, v in metrics_dict.items():
                metrics_list_dict[k].append(v)
            for k, v in images_dict.items():
                images_list_dict[k].append(v)
                
        if self.config.compute_distribution_diff:
            print(1111111)
            assert ray_bundle is not None and self.distribution_distance is not None
            distribution_diff_map: Dict[str, torch.Tensor] = {}
            for k in weights_list_dict.keys():
                assert len(weights_list_dict[k]) == len(samples_list_dict[k]) == len(all_outputs)
                samples_list = samples_list_dict[k]
                weights_list = weights_list_dict[k]
                num_rays_per_chunk = self.config.eval_num_rays_per_chunk
                image_height, image_width, n_samples = weights_list[0].shape[:3]
                num_rays = image_height * image_width
                all_pairwise_diff_list_mean = []
                all_pairwise_diff_list_max = []
                for num, idx in enumerate(range(0, num_rays, num_rays_per_chunk)):
                    new_samples_list = []
                    for i in range(len(samples_list)):
                        image_samples_list = samples_list[i]
                        image_weights = weights_list[i]
                        start_idx = idx
                        end_idx = idx + num_rays_per_chunk
                        chunk_samples: RaySamples = image_samples_list[num].to(self.device)
                        chunk_weights: Tensor = image_weights.view(-1, n_samples, 1)[start_idx: end_idx].to(self.device)
                        chunk_ray_bundle: RayBundle = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx).to(self.device)
                        assert chunk_samples.shape[0] == chunk_weights.shape[0] == chunk_ray_bundle.shape[0]
                        new_samples = self.pdf_sampler(chunk_ray_bundle, chunk_samples, chunk_weights)
                        new_samples_list.append(new_samples)
                    pairwise_diff_mean, pairwise_diff_max = self.distribution_distance(new_samples_list)
                    all_pairwise_diff_list_mean.append(pairwise_diff_mean.detach().cpu())
                    all_pairwise_diff_list_max.append(pairwise_diff_max.detach().cpu())
                distribution_diff_map[f"{k}.dist_var_mean"] = torch.cat(all_pairwise_diff_list_mean).reshape(image_height, image_width, -1)
                distribution_diff_map[f"{k}.dist_var_max"] = torch.cat(all_pairwise_diff_list_max).reshape(image_height, image_width, -1)
            all_images_dict.update(distribution_diff_map)
            

        avg_metrics_dict = {f"{k}": torch.mean(torch.tensor(v)) for k, v in metrics_list_dict.items()}
        var_metrics_dict = {f"{k}.var": torch.var(torch.tensor(v)) for k, v in metrics_list_dict.items()}
        out_metrics_dict = {**avg_metrics_dict, **var_metrics_dict}
        
        concat_images_dict = {f"{k}.samples": torch.concat(v, dim=1) for k, v in images_list_dict.items()}
        avg_images_dict = {f"{k}.samples_mean": torch.mean(torch.stack(v, dim=0), dim=0) for k, v in images_list_dict.items()}
        var_images_dict: Dict[str, Tensor] = {}
        for k, v in images_list_dict.items():
            var_images_dict[f"{k}.var"] = torch.var(torch.stack(v, dim=0), dim=0)
            if len(var_images_dict[f"{k}.var"].shape) == 3:
                var_images_dict[f"{k}.var"] = var_images_dict[f"{k}.var"].mean(-1, keepdim=True)
            out_metrics_dict[f"{k}.max_var"] = var_images_dict[f"{k}.var"].max()
            out_metrics_dict[f"{k}.mean_var"] = var_images_dict[f"{k}.var"].mean()
                
        all_images_dict.update(avg_images_dict)
        all_images_dict.update(var_images_dict)
        all_images_dict.update(concat_images_dict)
        all_images_dict.update({f"{k}.GT": v for k, v in ground_truth_dict.items()})
                
        return out_metrics_dict, all_images_dict
    