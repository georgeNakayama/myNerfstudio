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
import torch.nn.functional as F
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
from nerfstudio.data.scene_box import SceneBox
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
    global_entropy: bool = False
    """Whether to calculate entropy by sampling points in the entire bounding box"""
    add_end_bin: bool = True 
    """Specifies whether to add an ending bin to each ray's samples."""
    use_aabb_collider: bool = False
    """Specifies whether to use aabb collider instead of near and far collider"""
    use_gaussian_entropy: bool = False
    """Whether to use Gaussian entropy to compute entropy loss"""
    compute_kl: bool = False
    """Whether to compute the KL divergence"""


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
            use_gaussian_ent=self.config.use_gaussian_entropy,
            compute_kl=self.config.compute_kl
        )


        self.sampler = UniformSampler(single_jitter=self.config.use_single_jitter, num_samples=self.config.num_nerf_samples_per_ray)

        # Collider
        if self.config.use_aabb_collider:
            self.collider = AABBBoxCollider(near_plane=self.config.near_plane, scene_box=self.scene_box)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        
        self.entropy_renderer = RGBRenderer(background_color="last_sample")


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.config.num_proposal_iterations > 0:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle, return_samples:bool=False, **kwargs):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        metric_dict_list: Optional[Dict[str, List[Tensor]]] = None
        
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals, compute_global_entropy=self.config.global_entropy and self.training)
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
        mean_weights = weights.mean(-3)
        weights_list.append(mean_weights)
        ray_samples_list.append(ray_samples)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=new_ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        
            
        # [batch_ray, num_samples, dim] -> [batch_ray, K_samples, dim]
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "rgb_field_std": field_outputs["rgb_std"],
            "rgb_field_mean": field_outputs["rgb_mean"],
            "density_field_std": field_outputs["density_std"],
            "density_field_mean": field_outputs["density_mean"],
            "field_mean_weights": mean_weights,
            "field_sample_positions":SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), aabb=self.scene_box.aabb.to(weights.device))
        }
        
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            
            
        
        outputs["rgb_field_entropy"] = field_outputs["rgb_entropy"]
        outputs["density_field_entropy"] = field_outputs["density_entropy"]
        if self.config.compute_kl:
            outputs["rgb_field_kl"] = field_outputs["rgb_kl"]
            outputs["density_field_kl"] = field_outputs["density_kl"]
        
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
            
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

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
        outputs: Dict[str, Tensor] = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list)
            sh = outputs[output_name].shape
            outputs[output_name] = outputs[output_name].reshape(image_height, image_width, *sh[1:])
                
        rgb = outputs["rgb"]
        rgb_field_std = outputs["rgb_field_std"]
        rgb_field_mean = outputs["rgb_field_mean"]
        rgb_field_entropy = outputs["rgb_field_entropy"]
        ray_field_pos = outputs["field_sample_positions"]
        density_field_std = outputs["density_field_std"]
        density_field_entropy = outputs["density_field_entropy"] # [h, w, N, dim]
        density_field_mean = outputs["density_field_mean"]
        field_mean_weight = outputs["field_mean_weights"]
        rgb_variance = rgb.var(dim=(-1, -2))[..., None]
        rgb_mean = rgb.mean(dim=-2)
        depth_variance = outputs["depth"].var(dim=-2)
        depth_mean = outputs["depth"].mean(dim=-2)
        acc_variance = outputs["accumulation"].var(dim=-2)
        acc_mean = outputs["accumulation"].mean(dim=-2)
        rendered_rgb_entropy = self.entropy_renderer(rgb_field_entropy, weights=field_mean_weight)
        
            
        rendered_density_entropy = self.entropy_renderer(density_field_entropy, weights=field_mean_weight)
        rendered_rgb_std = self.entropy_renderer(rgb_field_std, weights=field_mean_weight)
        rendered_density_std = self.entropy_renderer(density_field_std, weights=field_mean_weight)
        new_outputs = {
            "rgb":rgb_mean,
            "rgb_field_std":rgb_field_std,
            "rgb_field_entropy":rgb_field_entropy,
            "rendered_rgb_std":rendered_rgb_std,
            "rendered_rgb_entropy":rendered_rgb_entropy,
            "rgb_field_mean":rgb_field_mean,
            "density_field_std":density_field_std,
            "density_field_entropy":density_field_entropy,
            "rendered_density_std":rendered_density_std,
            "rendered_density_entropy":rendered_density_entropy,
            "density_field_mean":density_field_mean,
            "rgb_var":rgb_variance,
            "depth":depth_mean,
            "depth_var":depth_variance,
            "acc":acc_mean,
            "acc_var":acc_variance,
            "field_sample_positions":ray_field_pos,
            "field_mean_weights":field_mean_weight,
            "origins": camera_ray_bundle.origins,
            "directions": camera_ray_bundle.directions,
        }
        if self.config.compute_kl:
            new_outputs["rgb_kl"] = outputs["rgb_field_kl"]
            new_outputs["density_kl"] = outputs["density_field_kl"]
            rendered_rgb_kl = self.entropy_renderer(outputs["rgb_field_kl"], weights=field_mean_weight)
            rendered_density_kl = self.entropy_renderer(outputs["density_field_kl"], weights=field_mean_weight)
            new_outputs["rendered_rgb_kl"] = rendered_rgb_kl
            new_outputs["rendered_density_kl"] = rendered_density_kl
        for i in range(self.config.num_proposal_iterations):
            new_outputs[f"prop_depth_{i}"] = outputs[f"prop_depth_{i}"]
        return new_outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        rgb_mean = outputs["rgb"].mean(-2)
        metrics_dict["psnr"] = self.psnr(rgb_mean, image)
        metrics_dict["rgb_loss"] = self.rgb_loss(image, rgb_mean)
        metrics_dict["rgb_std"] = outputs["rgb_field_std"].mean()
        metrics_dict["density_std"] = outputs["density_field_std"].mean()
        metrics_dict["global_rgb_mean"] = self.field.global_rgb_mean.mean()
        metrics_dict["global_density_mean"] = self.field.global_density_mean.mean()
        
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            termination_depth = batch["depth_image"].to(self.device)
            metrics_dict["depth_loss"] += depth_loss(
                weights=outputs["weights_list"][-1],
                ray_samples=outputs["ray_samples_list"][-1],
                termination_depth=termination_depth,
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
            if self.config.compute_kl:
                loss_dict["rgb_kl_loss"] = self.config.entropy_mult * outputs["rgb_field_kl"].mean()
                loss_dict["density_kl_loss"] = self.config.entropy_mult * outputs["density_field_kl"].mean()
            else:
                loss_dict["neg_rgb_entropy_loss"] = -1 * self.config.entropy_mult * outputs["rgb_field_entropy"].mean()
                loss_dict["neg_density_entropy_loss"] = -1 * self.config.entropy_mult * outputs["density_field_entropy"].mean()
            
            # interlevel loss
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            
            # distortion loss
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            
            # normal losses
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
        metrics_dict["density_entropy"] = float(torch.mean(outputs["density_field_entropy"]))
        metrics_dict["rgb_entropy"] = float(torch.mean(outputs["rgb_field_entropy"]))
        depth_mask = ground_truth_depth > 0
        metrics_dict["depth_mse"] = float(
            torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
        )
        
        avged_rgb_field_std = outputs["rgb_field_std"].mean(dim=(-1, -2))[..., None]
        avged_rgb_field_ent = outputs["rgb_field_entropy"].mean(-2)
        avged_density_field_std = outputs["density_field_std"].mean(-2)
        avged_density_field_ent = outputs["density_field_entropy"].mean(-2)
        
        projected_depth = outputs["origins"] + outputs["directions"] * outputs["depth"]
        projected_depth = SceneBox.get_normalized_positions(projected_depth, aabb=self.scene_box.aabb.to(self.device))
        projected_depth_im = torch.cat([projected_depth, outputs["rgb"]], dim=-1).reshape(-1, 6)
        perm = torch.randperm(projected_depth_im.shape[0])[:projected_depth_im.shape[0] // 8]
        projected_depth_im = projected_depth_im[perm]
        projected_gt_depth = outputs["origins"] + outputs["directions"] * ground_truth_depth
        projected_gt_depth = projected_gt_depth.reshape(-1, 3)[depth_mask.flatten()]
        projected_gt_depth = SceneBox.get_normalized_positions(projected_gt_depth, aabb=self.scene_box.aabb.to(self.device))
        projected_gt_depth_im = torch.cat([projected_gt_depth, image.reshape(3, -1).transpose(0,1)[depth_mask.flatten()]], dim=-1).reshape(-1, 6)
        perm = torch.randperm(projected_gt_depth_im.shape[0])[:projected_gt_depth_im.shape[0] // 8]
        projected_gt_depth_im = projected_gt_depth_im[perm]
        pos = outputs["field_sample_positions"] # [h, w, N, 3]
        device = pos.device
        DOWNSAMPLE_RES = 16
        if DOWNSAMPLE_RES > 0:
            x, y, z = torch.linspace(-1, 1, steps=DOWNSAMPLE_RES), torch.linspace(-1, 1, steps=DOWNSAMPLE_RES), torch.linspace(-1, 1, steps=DOWNSAMPLE_RES)
            y, x, z = torch.meshgrid(y, x, z, indexing="ij")
            indices = torch.stack([y, x, z], dim=-1)[None].to(device)
            pos = F.grid_sample(
                pos.reshape(1, -1, 3).transpose(1,2).reshape(1, 3, h, w, self.config.num_nerf_samples_per_ray),
                indices, mode="nearest", align_corners=True).reshape(3, -1).transpose(0,1)
            colored_rgb_field_mean = F.grid_sample(
                outputs["rgb_field_mean"].reshape(1, -1, 3).transpose(1,2).reshape(1, 3, h, w, self.config.num_nerf_samples_per_ray), 
                indices, mode="nearest", align_corners=True).reshape(3, -1).transpose(0,1)
            colored_rgb_field_std = F.grid_sample(outputs["rgb_field_std"].to(indices).mean(-1).unsqueeze(0).unsqueeze(0), indices,
                                                   mode="nearest", align_corners=True).reshape(-1, 1)
            colored_rgb_field_ent = F.grid_sample(outputs["rgb_field_entropy"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                   mode="nearest", align_corners=True).reshape(-1, 1)
            colored_density_field_mean = F.grid_sample(outputs["density_field_mean"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                        mode="nearest", align_corners=True).reshape(-1, 1)
            colored_weight_field_mean = F.grid_sample(outputs["field_mean_weights"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                       mode="nearest", align_corners=True).reshape(-1, 1)
            colored_density_field_std = F.grid_sample(outputs["density_field_std"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                       mode="nearest", align_corners=True).reshape(-1, 1)
            colored_density_field_ent = F.grid_sample(outputs["density_field_entropy"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                       mode="nearest", align_corners=True).reshape(-1, 1)
            if self.config.compute_kl:
                colored_rgb_field_kl = F.grid_sample(outputs["rgb_kl"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                   mode="nearest", align_corners=True).reshape(-1, 1)
                colored_density_field_kl = F.grid_sample(outputs["density_kl"].to(indices)[..., 0].unsqueeze(0).unsqueeze(0), indices,
                                                   mode="nearest", align_corners=True).reshape(-1, 1)
        
        colored_rgb_field_std = colormaps.apply_colormap(colored_rgb_field_std, colormap_options=ColormapOptions(colormap="default", normalize=True))
        colored_rgb_field_ent = colormaps.apply_colormap(colored_rgb_field_ent, colormap_options=ColormapOptions(colormap="default", normalize=True))
        colored_weight_field_mean = colormaps.apply_colormap(colored_weight_field_mean, colormap_options=ColormapOptions(colormap="default"))
        colored_density_field_mean = colormaps.apply_colormap(colored_density_field_mean, colormap_options=ColormapOptions(colormap="default", normalize=True))
        colored_density_field_std = colormaps.apply_colormap(colored_density_field_std, colormap_options=ColormapOptions(colormap="default", normalize=True))
        colored_density_field_ent = colormaps.apply_colormap(colored_density_field_ent, colormap_options=ColormapOptions(colormap="default", normalize=True))
        if self.config.compute_kl:
                colored_rgb_field_kl = colormaps.apply_colormap(colored_rgb_field_kl, colormap_options=ColormapOptions(colormap="default", normalize=True))
                colored_density_field_kl = colormaps.apply_colormap(colored_density_field_kl, colormap_options=ColormapOptions(colormap="default", normalize=True))
        rgb_field_3d_frustum_mean = torch.cat([pos, colored_rgb_field_mean], dim=-1).reshape(-1, 3+3)
        rgb_field_3d_frustum_std = torch.cat([pos, colored_rgb_field_std], dim=-1).reshape(-1, 3+3)
        rgb_field_3d_frustum_ent = torch.cat([pos, colored_rgb_field_ent], dim=-1).reshape(-1, 3+3)
        weight_field_3d_frustum_mean = torch.cat([pos, colored_weight_field_mean], dim=-1).reshape(-1, 3+3)
        density_field_3d_frustum_mean = torch.cat([pos, colored_density_field_mean], dim=-1).reshape(-1, 3+3)
        density_field_3d_frustum_std = torch.cat([pos, colored_density_field_std], dim=-1).reshape(-1, 3+3)
        density_field_3d_frustum_ent = torch.cat([pos, colored_density_field_ent], dim=-1).reshape(-1, 3+3)
        if self.config.compute_kl:
            density_field_3d_frustum_kl = torch.cat([pos, colored_density_field_kl], dim=-1).reshape(-1, 3+3)
            rgb_field_3d_frustum_kl = torch.cat([pos, colored_rgb_field_kl], dim=-1).reshape(-1, 3+3)
        
        images_dict = {
            "sample_img.mean": combined_rgb_mean, 
            "sample_img.var": colormaps.apply_colormap(outputs["rgb_var"], colormap_options=ColormapOptions(normalize=True)), 
            "rendered_rgb_std": colormaps.apply_colormap(outputs["rendered_rgb_std"], colormap_options=ColormapOptions(normalize=True)),
            "rendered_rgb_ent": colormaps.apply_colormap(outputs["rendered_rgb_entropy"], colormap_options=ColormapOptions(normalize=True)),
            "color_field_avg_std": colormaps.apply_colormap(avged_rgb_field_std, colormap_options=ColormapOptions(normalize=True)),
            "color_field_avg_ent": colormaps.apply_colormap(avged_rgb_field_ent, colormap_options=ColormapOptions(normalize=True)),
            "rgb_field_3d_mean": rgb_field_3d_frustum_mean,
            "color_field_3d_std": rgb_field_3d_frustum_std,
            "color_field_3d_ent": rgb_field_3d_frustum_ent,
            "sample_depth_mean": combined_depth_mean,
            "sample_depth_var": colormaps.apply_colormap(outputs["depth_var"], colormap_options=ColormapOptions(normalize=True)),
            "rendered_density_std": colormaps.apply_colormap(outputs["rendered_density_std"], colormap_options=ColormapOptions(normalize=True)),
            "rendered_density_ent": colormaps.apply_colormap(outputs["rendered_density_entropy"], colormap_options=ColormapOptions(normalize=True)),
            "density_field_avg_std": colormaps.apply_colormap(avged_density_field_std, colormap_options=ColormapOptions(normalize=True)),
            "density_field_avg_ent": colormaps.apply_colormap(avged_density_field_ent, colormap_options=ColormapOptions(normalize=True)),
            "weight_field_3d_mean": weight_field_3d_frustum_mean,
            "density_field_3d_mean": density_field_3d_frustum_mean,
            "density_field_3d_std": density_field_3d_frustum_std,
            "density_field_3d_ent": density_field_3d_frustum_ent,
            "sample_accumulation_mean": colormaps.apply_colormap(outputs["acc"]), 
            "projected_depth": projected_depth_im,
            "projected_GT_depth": projected_gt_depth_im,
            }
        
        if self.config.compute_kl:
            images_dict["rgb_3d_kl"] = rgb_field_3d_frustum_kl
            images_dict["density_3d_kl"] = density_field_3d_frustum_kl
            images_dict["rgb_rendered_kl"] = colormaps.apply_colormap(outputs["rendered_rgb_kl"], colormap_options=ColormapOptions(normalize=True))
            images_dict["density_rendered_kl"] = colormaps.apply_colormap(outputs["rendered_density_kl"], colormap_options=ColormapOptions(normalize=True))
            images_dict["rgb_avg_kl"] = colormaps.apply_colormap(outputs["rgb_kl"].mean(-2), colormap_options=ColormapOptions(normalize=True))
            images_dict["density_avg_kl"] = colormaps.apply_colormap(outputs["density_kl"].mean(-2), colormap_options=ColormapOptions(normalize=True))
        
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
                accumulation=outputs["acc"],
            )
            images_dict[key] = prop_depth_i
            metrics_dict[f"max_{key}"] = float(torch.max(outputs[key]))
            metrics_dict[f"min_{key}"] = float(torch.min(outputs[key]))
            
        return metrics_dict, images_dict
