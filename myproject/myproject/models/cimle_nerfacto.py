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
from typing import Any, Dict, List, Literal, Mapping, Tuple, Type, Optional, Union
from pathlib import Path
from collections import defaultdict
from jaxtyping import Float
import numpy as np
import torch
from torch.nn import Parameter
from torch import Tensor
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.models.base_model import Model
from myproject.models.base_cimle_model import cIMLEModel, cIMLEModelConfig
from myproject.fields.cimle_density_fields import cIMLEHashMLPDensityField
from myproject.fields.cimle_nerfacto_field import cIMLENerfactoField
from myproject.utils.myutils import sync_data

@dataclass
class cIMLENerfactoModelConfig(cIMLEModelConfig, NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: cIMLENerfactoModel)
    
    color_cimle: bool=False
    """whether apply cimle only to color channel"""
    use_cimle_in_proposal_networks: bool = True 
    """Specifices whether to inject cimle to proposal networks"""
    finetune_mlps: bool = False
    """Whether to finetune the mlp layers when training with cIMLE"""
    finetune_mlp_type: Literal[1, 2] = 1
    """Choose type of mlp type for finetuning"""
    use_cimle_grid: bool = False 
    """Specifies whether to use cIMLE hash grid"""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "uniform"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    


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
            cimle_pretrain=self.config.cimle_pretrain,
            use_cimle_grid=self.config.use_cimle_grid
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
                cimle_pretrain=self.config.cimle_pretrain,
                use_cimle_grid=self.config.use_cimle_grid
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
                    cimle_pretrain=self.config.cimle_pretrain,
                    use_cimle_grid=self.config.use_cimle_grid
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

    
    def sample_cimle_latent(self, cimle_sample_num: int, image_num: int = 1) -> List[Tensor]:
        if not self.config.use_cimle_grid:
            return super().sample_cimle_latent(cimle_sample_num, image_num)
        
        fine_nerf_zs = torch.randn([cimle_sample_num] + list(self.field.cimle.cimle_grid.get_hash_table_size())) * self.config.cimle_latent_std
        all_zs = [fine_nerf_zs]
        if self.config.use_cimle_in_proposal_networks:
            if self.config.use_same_proposal_network:
                all_zs.append(torch.randn([cimle_sample_num] + list(self.proposal_networks[0].cimle.cimle_grid.get_hash_table_size())) * self.config.cimle_latent_std)
            else:
                for i in range(self.config.num_proposal_iterations):
                    all_zs.append(torch.randn([cimle_sample_num] + list(self.proposal_networks[i].cimle.cimle_grid.get_hash_table_size())) * self.config.cimle_latent_std)
        
        if self.eval_ctx and self.cimle_cached_samples is None:
                self.cimle_cached_samples = all_zs
            
        if self.cimle_cached_samples is not None and len(self.cimle_cached_samples) == len(all_zs):
            f = True 
            for z, cached_sample in zip(all_zs, self.cimle_cached_samples):
                f = f and z.shape == cached_sample.shape
            if f:
                return self.cimle_cached_samples
        
        return all_zs
    
    def set_cimle_latents_from_loss(self, all_z: List[Tensor], all_losses: Tensor) -> None:
        if not self.config.use_cimle_grid:
            return super().set_cimle_latents_from_loss(all_z, all_losses)
        
        all_losses = all_losses.mean(1)
        idx_to_take = torch.argmin(all_losses, dim=0)
        fine_nerf_z = all_z[0][idx_to_take]
        self.field.cimle.cimle_grid.load_hash_table_weights(fine_nerf_z)
        if self.config.use_cimle_in_proposal_networks:
            if self.config.use_same_proposal_network:
                prop_net_z = all_z[1][idx_to_take]
                assert prop_net_z.shape == self.proposal_networks[0].cimle.cimle_grid.get_hash_table_size()
                self.proposal_networks[0].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
            else:
                for k in range(self.config.num_proposal_iterations):
                    prop_net_z = all_z[k + 1][idx_to_take]
                    assert prop_net_z.shape == self.proposal_networks[k].cimle.cimle_grid.get_hash_table_size()
                    self.proposal_networks[k].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
    
    @torch.no_grad()
    def cache_latents(self, datamanager: VanillaDataManager, step):
        if not self.config.use_cimle_grid:
            return cIMLEModel.cache_latents(self, datamanager, step)
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
                
                # load cIMLE latent into the hash tables
                fine_nerf_z = all_z[0][n]
                assert fine_nerf_z.shape == self.field.cimle.cimle_grid.get_hash_table_size()
                self.field.cimle.cimle_grid.load_hash_table_weights(fine_nerf_z)
                if self.config.use_cimle_in_proposal_networks:
                    if self.config.use_same_proposal_network:
                        prop_net_z = all_z[1][n]
                        assert prop_net_z.shape == self.proposal_networks[0].cimle.cimle_grid.get_hash_table_size()
                        self.proposal_networks[0].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
                    else:
                        for k in range(self.config.num_proposal_iterations):
                            prop_net_z = all_z[k + 1][n]
                            assert prop_net_z.shape == self.proposal_networks[k].cimle.cimle_grid.get_hash_table_size()
                            self.proposal_networks[k].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
                
                for i, (camera_ray_bundle, batch) in enumerate(datamanager.fixed_indices_train_dataloader):
                    # one latent per image for now. 
                    img_idx = batch['image_idx']
                    height, width = camera_ray_bundle.shape
                    perm = torch.randperm(height*width)[:num_rays]
                    # print(indices)
                    ray_bundle = camera_ray_bundle.flatten()[perm]
                    batch = {k: v.flatten(end_dim=1)[perm] if isinstance(v, Tensor) else v for k, v in batch.items()}
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
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = NerfactoModel.get_metrics_dict(self, outputs, batch)
        for k in outputs.keys():
            if "latent_diff" in k:
                metrics_dict[k] = outputs[k].mean()
        return metrics_dict
    
    def get_outputs(self, ray_bundle: RayBundle, return_samples: bool = False, **kwargs):
        return NerfactoModel.get_outputs(self, ray_bundle, return_samples=return_samples)

    def forward(self, ray_bundle: RayBundle, sample_latent: bool=False, return_samples:bool=False, **kwargs) -> Dict[str, Union[Tensor, List]]:
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
            samples = self.sample_cimle_latent(1, max_cam_ind + 1)
            # load cIMLE latent into the hash tables
            fine_nerf_z = samples[0][0]
            assert fine_nerf_z.shape == self.field.cimle.cimle_grid.get_hash_table_size()
            self.field.cimle.cimle_grid.load_hash_table_weights(fine_nerf_z)
            if self.config.use_cimle_in_proposal_networks:
                if self.config.use_same_proposal_network:
                    prop_net_z = samples[1][0]
                    assert prop_net_z.shape == self.proposal_networks[0].cimle.cimle_grid.get_hash_table_size()
                    self.proposal_networks[0].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
                else:
                    for k in range(self.config.num_proposal_iterations):
                        prop_net_z = samples[k + 1][0]
                        assert prop_net_z.shape == self.proposal_networks[k].cimle.cimle_grid.get_hash_table_size()
                        self.proposal_networks[k].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
            

        return Model.forward(self, ray_bundle, return_samples=return_samples, **kwargs)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups: Dict[str, List[Parameter]] = {}
        prop_network_pgs: Dict[str, List[Parameter]] = defaultdict(list)
        field_pgs: Dict[str, List[Parameter]] = defaultdict(list)
        
        for name, param in self.proposal_networks.named_parameters():
            if "cimle" in name.split("."):
                prop_network_pgs["proposal_networks.cimle"].append(param)
                CONSOLE.print(f"Parameter [red]{name}[/red] is added to cIMLE group!")
            else:
                if self.config.finetune_mlps:
                    if self.config.finetune_mlp_type == 1:
                        if "mlp_base_mlp" in name.split("."):
                            prop_network_pgs["proposal_networks.cimle"].append(param)
                            CONSOLE.print(f"Parameter [red]{name}[/red] is added to cIMLE group!")
                        else:
                            prop_network_pgs["proposal_networks"].append(param)
                    elif self.config.finetune_mlp_type == 2:
                        if "mlp_base_grid" in name.split("."):
                            prop_network_pgs["proposal_networks"].append(param)
                        else:
                            prop_network_pgs["proposal_networks.cimle"].append(param)
                            CONSOLE.print(f"Parameter [red]{name}[/red] is added to cIMLE group!")
                else:
                    prop_network_pgs["proposal_networks"].append(param)
                    
        for name, param in self.field.named_parameters():
            if "cimle" in name.split("."):
                prop_network_pgs["fields.cimle"].append(param)
                CONSOLE.print(f"Parameter [red]{name}[/red] is added to cIMLE group!")
            else:
                if self.config.finetune_mlps:
                    if self.config.finetune_mlp_type == 1:
                        if "mlp_base_mlp" in name.split("."):
                            prop_network_pgs["fields.cimle"].append(param)
                            CONSOLE.print(f"Parameter [red]{name}[/red] is added to cIMLE group!")
                        else:
                            prop_network_pgs["fields"].append(param)
                    elif self.config.finetune_mlp_type == 2:
                        if "mlp_base_grid" in name.split("."):
                            prop_network_pgs["fields"].append(param)
                        else:
                            prop_network_pgs["fields.cimle"].append(param)
                            CONSOLE.print(f"Parameter [red]{name}[/red] is added to cIMLE group!")
                else:
                    prop_network_pgs["fields"].append(param)
                
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
    
    
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> List[Dict[str, torch.Tensor]]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        if not self.config.use_cimle_grid:
            return super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        
        cimle_latents = self.sample_cimle_latent(self.cimle_ensemble_num)
        all_outputs = []
        for n in range(self.cimle_ensemble_num):
                
            # load cIMLE latent into the hash tables
            fine_nerf_z = cimle_latents[0][n]
            assert fine_nerf_z.shape == self.field.cimle.cimle_grid.get_hash_table_size()
            self.field.cimle.cimle_grid.load_hash_table_weights(fine_nerf_z)
            if self.config.use_cimle_in_proposal_networks:
                if self.config.use_same_proposal_network:
                    prop_net_z = cimle_latents[1][n]
                    assert prop_net_z.shape == self.proposal_networks[0].cimle.cimle_grid.get_hash_table_size()
                    self.proposal_networks[0].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
                else:
                    for k in range(self.config.num_proposal_iterations):
                        prop_net_z = cimle_latents[k + 1][n]
                        assert prop_net_z.shape == self.proposal_networks[k].cimle.cimle_grid.get_hash_table_size()
                        self.proposal_networks[k].cimle.cimle_grid.load_hash_table_weights(prop_net_z)
                
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
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
    
    
    def get_image_metrics_and_images(
        self, all_outputs: List[Dict[str, Any]], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        def _get_image_metrics_and_images(
            outputs: Dict[str, Any], _batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
            image: torch.Tensor = _batch["image"].to(self.device)
            rgb: torch.Tensor = outputs["rgb"].to(self.device)
            acc: torch.Tensor = outputs["accumulation"]
            depth: torch.Tensor = outputs["depth"]
            field_ld: torch.Tensor = outputs["field.latent_diff"]
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
            
            metrics_dict["max_field.latent_diff"] = float(torch.max(outputs["field.latent_diff"]))
            metrics_dict["min_field.latent_diff"] = float(torch.min(outputs["field.latent_diff"]))
            metrics_dict["mean_field.latent_diff"] = float(torch.mean(outputs["field.latent_diff"]))
            
            images_dict = {"img": rgb[0].moveaxis(0, -1), "accumulation": acc, "depth": depth, "field.latent_diff": field_ld}
            if self.config.use_cimle_in_proposal_networks:
                images_dict.update({f"prop_{i}.latent_diff": outputs[f"prop_{i}.latent_diff"] for i in range(self.config.num_proposal_iterations)})
                metrics_dict.update({f"max_prop_{i}.latent_diff": float(torch.max(outputs[f"prop_{i}.latent_diff"])) for i in range(self.config.num_proposal_iterations)})
                metrics_dict.update({f"min_prop_{i}.latent_diff": float(torch.min(outputs[f"prop_{i}.latent_diff"])) for i in range(self.config.num_proposal_iterations)})
                metrics_dict.update({f"mean_prop_{i}.latent_diff": float(torch.mean(outputs[f"prop_{i}.latent_diff"])) for i in range(self.config.num_proposal_iterations)})

            images_dict.update({f"prop_depth_{i}": outputs[f"prop_depth_{i}"] for i in range(self.config.num_proposal_iterations)})
            metrics_dict.update({f"max_prop_depth_{i}": float(torch.max(outputs[f"prop_depth_{i}"])) for i in range(self.config.num_proposal_iterations)})
            metrics_dict.update({f"min_prop_depth_{i}": float(torch.min(outputs[f"prop_depth_{i}"])) for i in range(self.config.num_proposal_iterations)})
            return metrics_dict, images_dict, {"img": image[0].moveaxis(0, -1)}, weights_list, ray_samples_list
        
        all_metrics_dict, all_images_dict = self.get_image_metrics_and_images_loop(_get_image_metrics_and_images, all_outputs, batch)
        original_tags = ['img']
        to_color_map_tags = ["accumulation", "field.latent_diff"]
        if self.config.use_cimle_in_proposal_networks:
            to_color_map_tags += [f"prop_{i}.latent_diff" for i in range(self.config.num_proposal_iterations)]
        to_depth_color_map_tags = ["depth"] + [f"prop_depth_{i}" for i in range(self.config.num_proposal_iterations)]
        clr_map_imgages_dict = {}
        for tag in original_tags:
            keys = [k for k in all_images_dict.keys() if tag == ".".join(k.split(".")[:-1])]
            clr_map_imgages_dict.update({key: all_images_dict[key] if "var" not in key.split(".")[-1] else colormaps.apply_colormap(all_images_dict[key], colormap_options=colormaps.ColormapOptions(normalize=True)) for key in keys})
        for tag in to_color_map_tags:
            keys = [k for k in all_images_dict.keys() if tag == ".".join(k.split(".")[:-1])]
            clr_map_imgages_dict.update({key: colormaps.apply_colormap(all_images_dict[key]) if "var" not in key.split(".")[-1] else colormaps.apply_colormap(all_images_dict[key], colormap_options=colormaps.ColormapOptions(normalize=True)) for key in keys})
        for tag in to_depth_color_map_tags:
            keys = [k for k in all_images_dict.keys() if tag == ".".join(k.split(".")[:-1])]
            clr_map_imgages_dict.update({key: colormaps.apply_depth_colormap(
                all_images_dict[key], 
                accumulation=all_images_dict["accumulation." + key.split(".")[-1]]
            ) if "var" not in key.split(".")[-1] else colormaps.apply_colormap(all_images_dict[key], colormap_options=colormaps.ColormapOptions(normalize=True)) for key in keys})
        return all_metrics_dict, clr_map_imgages_dict