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
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
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
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.model_components.losses import depth_loss, DepthLossType
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig, DepthNerfactoModel
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from myproject.model_components.losses import neg_nll_loss, gaussian_entropy, log_normal_entropy, logistic_normal_entropy
from myproject.fields.stochastic_nerfacto_field import StochasticNerfactoField

@dataclass
class StochasticNerfactoModelConfig(DepthNerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: StochasticNerfactoModel)
    
    K_samples: int = 32
    """The number of stochastic samples used to compute the expectation"""
    eval_log_image_num: int = 5
    """Specifies the number of images to log when evaluating test views"""
    num_nerf_samples_per_ray: int = 256
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    entropy_mult: float = 0.001
    """multiplicative factor to the entropy loss term"""
    num_proposal_iterations: int = 0
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "uniform"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = False
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    depth_method: Literal["expected", "median"] = "expected"
    """Which depth map rendering method to use."""
    appearance_embed_dim: int = 0
    """Dimension of the appearance embedding."""
    add_end_bin: bool = True 
    """Specifies whether to add an ending bin to each ray's samples."""
    use_aabb_collider: bool = True
    """Specifies whether to use aabb collider instead of near and far collider"""


class StochasticNerfactoModel(DepthNerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: StochasticNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = StochasticNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            stochastic_samples=self.config.K_samples,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )


        self.sampler = UniformSampler(single_jitter=self.config.use_single_jitter, num_samples=self.config.num_nerf_samples_per_ray)

        # Collider
        if self.config.use_aabb_collider:
            self.collider = AABBBoxCollider(near_plane=self.config.near_plane, scene_box=self.scene_box)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
            


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.config.num_proposal_iterations > 0:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle, return_samples:bool=False, **kwargs):
        ray_samples: RaySamples
        ray_samples = self.sampler(ray_bundle)
        metric_dict_list: Optional[Dict[str, List[Tensor]]] = None
        
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        bundle_shape = ray_bundle.shape
        ray_bundle = ray_bundle.reshape(list(bundle_shape) + [1]).broadcast_to(list(bundle_shape) + [self.config.K_samples])
        new_ray_samples = ray_bundle.get_ray_samples(
            bin_starts=ray_samples.frustums.starts.unsqueeze(-3).repeat_interleave(self.config.K_samples, dim=-3),
            bin_ends=ray_samples.frustums.ends.unsqueeze(-3).repeat_interleave(self.config.K_samples, dim=-3),
            spacing_starts=ray_samples.spacing_starts.unsqueeze(-3).repeat_interleave(self.config.K_samples, dim=-3),
            spacing_ends=ray_samples.spacing_ends.unsqueeze(-3).repeat_interleave(self.config.K_samples, dim=-3),
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )
        
        weights = new_ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=new_ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        
            
        
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "rgb_std": field_outputs["rgb_std"].mean(-2, keepdim=True).repeat_interleave(self.config.K_samples, dim=-2),
            "rgb_mean": field_outputs["rgb_mean"].mean(-2, keepdim=True).repeat_interleave(self.config.K_samples, dim=-2),
            "density_std": field_outputs["density_std"].mean(-2, keepdim=True).repeat_interleave(self.config.K_samples, dim=-2),
            "density_mean": field_outputs["density_mean"].mean(-2, keepdim=True).repeat_interleave(self.config.K_samples, dim=-2),
        }
        
        if self.training:
            outputs["weights"] = weights
            outputs["ray_samples"] = new_ray_samples
            
        outputs["rgb_entropy"] = field_outputs["rgb_entropy"].mean(-2, keepdim=True).repeat_interleave(self.config.K_samples, dim=-2)
        outputs["density_entropy"] = field_outputs["density_entropy"].mean(-2, keepdim=True).repeat_interleave(self.config.K_samples, dim=-2)
        
        if "field.latent_diff" in field_outputs.keys():
            outputs["field.latent_diff"] = field_outputs["field.latent_diff"].mean(-1)
        if metric_dict_list is not None:
            if "latent_diff" in metric_dict_list.keys():
                for i, diff in enumerate(metric_dict_list["latent_diff"]):
                    outputs[f"prop_{i}.latent_diff"] = diff.mean(-1)
        
        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )
            
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
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
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        
        outputs = {k: v.reshape(image_height, image_width, self.config.K_samples, -1) for k, v in outputs.items()}
        rgb = outputs["rgb"]
        rgb_sample_std = outputs["rgb_std"].mean(dim=(-1, -2))[..., None]
        density_sample_std = outputs["density_std"].mean(-2)
        rgb_variance = rgb.var(dim=(-1, -2))[..., None]
        rgb_mean = rgb.mean(dim=-2)
        depth_variance = outputs["depth"].var(dim=-2)
        depth_mean = outputs["depth"].mean(dim=-2)
        acc_variance = outputs["accumulation"].var(dim=-2)
        acc_mean = outputs["accumulation"].mean(dim=-2)
        outputs = {
            "rgb":rgb_mean,
            "rgb_sample_std":rgb_sample_std,
            "density_sample_std":density_sample_std,
            "rgb_var":rgb_variance,
            "depth":depth_mean,
            "depth_var":depth_variance,
            "acc":acc_mean,
            "acc_var":acc_variance,
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        rgb_mean = outputs["rgb"].mean(-2)
        metrics_dict["psnr"] = self.psnr(rgb_mean, image)
        metrics_dict["rgb_loss"] = self.rgb_loss(image, rgb_mean)
        metrics_dict["rgb_std"] = outputs["rgb_std"].mean()
        metrics_dict["rgb_mean"] = outputs["rgb_mean"].mean()
        metrics_dict["density_std"] = outputs["density_std"].mean()
        metrics_dict["density_mean"] = outputs["density_mean"].mean()
        
        if self.training:
            metrics_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            termination_depth = batch["depth_image"].to(self.device)
            metrics_dict["depth_loss"] += depth_loss(
                weights=outputs["weights"],
                ray_samples=outputs["ray_samples"],
                termination_depth=termination_depth.unsqueeze(-2).repeat_interleave(self.config.K_samples, dim=-2),
                predicted_depth=outputs["depth"],
                sigma=sigma,
                directions_norm=outputs["directions_norm"],
                is_euclidean=self.config.is_euclidean_depth,
                depth_loss_type=self.config.depth_loss_type,
            )
                
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        
        if self.training:
            loss_dict["neg_nll_loss"] = neg_nll_loss(outputs["rgb"], image, self.config.K_samples)
            loss_dict["rgb_entropy_loss"] = self.config.entropy_mult * outputs["rgb_entropy"].mean()
            loss_dict["density_entropy_loss"] = self.config.entropy_mult * outputs["density_entropy"].mean()
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            assert metrics_dict is not None and "depth_loss" in metrics_dict
            loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        ground_truth_depth = batch["depth_image"].to(self.device)
        if not self.config.is_euclidean_depth:
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]
            
        h, w = image.shape[:2]
        rgb_mean = outputs["rgb"]
        
        
        
        rendered_depth_mean = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["acc"],
        )
        combined_rgb_mean = torch.cat([image, outputs["rgb"]], dim=1)
        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        combined_depth_mean = torch.cat([ground_truth_depth_colormap, rendered_depth_mean], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_mean = torch.moveaxis(rgb_mean, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb_mean)
        ssim = self.ssim(image, rgb_mean)
        lpips = self.lpips(image, rgb_mean)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        metrics_dict["max_depth"] = float(torch.max(outputs["depth"]))
        metrics_dict["min_depth"] = float(torch.min(outputs["depth"]))
        depth_mask = ground_truth_depth > 0
        metrics_dict["depth_mse"] = float(
            torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
        )
        
        images_dict = {
            "img.mean": combined_rgb_mean, 
            "color_samples.std": colormaps.apply_colormap(outputs["rgb_sample_std"], colormap_options=ColormapOptions(normalize=True)),
            "color_samples.std_unorm": colormaps.apply_colormap(outputs["rgb_sample_std"]),
            "density_samples.std": colormaps.apply_colormap(outputs["density_sample_std"], colormap_options=ColormapOptions(normalize=True)),
            "density_samples.std_unorm": colormaps.apply_colormap(outputs["density_sample_std"]),
            "img.var": colormaps.apply_colormap(outputs["rgb_var"], colormap_options=ColormapOptions(normalize=True)), 
            "img.var_unnorm": colormaps.apply_colormap(outputs["rgb_var"]), 
            "accumulation.mean": colormaps.apply_colormap(outputs["acc"]), 
            "depth.mean": combined_depth_mean,
            "depth.var": colormaps.apply_colormap(outputs["depth_var"], colormap_options=ColormapOptions(normalize=True)),
            "depth.var_unnorm": colormaps.apply_colormap(outputs["depth_var"])
            }
        
        # images_dict["img.samples"] = rgb[:, :, :self.config.eval_log_image_num].permute(0, 2, 1, 3).reshape(h, -1, 3)
        
        # rendered_depth_samples = []
        # rendered_acc_samples = []
        # for i in range(self.config.eval_log_image_num):
        #     acc_render = colormaps.apply_colormap(outputs["accumulation"][..., i, :])
        #     depth_render = colormaps.apply_depth_colormap(
        #         outputs["depth"][..., i, :],
        #         accumulation=outputs["accumulation"][..., i, :],
        #     )
        #     rendered_acc_samples.append(acc_render)
        #     rendered_depth_samples.append(depth_render)
        
        # images_dict["depth.samples"] = torch.cat(rendered_depth_samples, dim=1)
        # images_dict["accumulation.samples"] = torch.cat(rendered_acc_samples, dim=1)

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i
            metrics_dict[f"max_{key}"] = float(torch.max(outputs[key]))
            metrics_dict[f"min_{key}"] = float(torch.min(outputs[key]))
            
        return metrics_dict, images_dict
