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
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Literal,
    Tuple,
    Type,
    Union,
)

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import (
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig, TDataset
from myproject.data.datasets.cimle_dataset import cIMLEDataset
from myproject.data.cimle_pixel_samplers import cIMLEPixelSampler
from nerfstudio.data.pixel_samplers import PixelSampler, PatchPixelSampler
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class cIMLEDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: cIMLEDataManager)
    """target class to instantiate"""
    

class cIMLEDataManager(VanillaDataManager):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: cIMLEDataManagerConfig
    
    def __init__(
        self,
        config: cIMLEDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):

        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs, _dataset_type=cIMLEDataset)


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        ray_bundle, batch = super().next_train(step)
        if "cimle_latent" in batch.keys():
            ray_bundle.metadata['cimle_latent'] = batch['cimle_latent'].clone()
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        ray_bundle, batch = super().next_eval(step)
        if "cimle_latent" in batch.keys():
            ray_bundle.metadata['cimle_latent'] = batch['cimle_latent'].clone()
        return ray_bundle, batch    
    
    def _get_pixel_sampler(self, dataset: TDataset, *args: Any, **kwargs: Any) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return cIMLEPixelSampler(*args, **kwargs)

    def next_train_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_idx, camera_ray_bundle, batch = super().next_train_image(step)
        height, width = camera_ray_bundle.shape
        if "cimle_latent" in batch.keys():
            camera_ray_bundle.metadata['cimle_latent'] = batch['cimle_latent'].clone().reshape(1, 1, -1).expand(height, width, -1)
        return image_idx, camera_ray_bundle, batch
    
