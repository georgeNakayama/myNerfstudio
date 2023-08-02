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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Dict, List, Tuple, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import imageio
import numpy as np
import torch
from collections import defaultdict

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class SparseScannetDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: SparseScannet)
    """target class to instantiate"""
    data: Path = Path("data/scannet/0710")
    """Directory specifying location of data."""
    train_json_name: str = "transforms_train.json"
    """Directory specifying location of data."""
    val_json_name: str = "transforms_val.json"
    """Directory specifying location of data."""
    test_json_name: str = "transforms_test.json"
    """Directory specifying location of data."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    use_mask: bool = False
    """whether to use mask for training"""


@dataclass
class SparseScannet(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    scene_box: Optional[SceneBox] = None
    all_poses: Optional[Dict[str, torch.Tensor]] = None 
    all_image_filenames: Optional[Dict[str, List[Path]]] = None
    all_depth_filenames: Optional[Dict[str, List[Path]]] = None
    all_mask_filenames: Optional[Dict[str, List[Path]]] = None
    depth_unit_scale_factor: float = 1e-3
    transform_matrix: Optional[torch.Tensor] = None
    hwcxcyfxfy: Optional[Tuple[int, int, float, float, float, float]] = None
    config: SparseScannetDataParserConfig

    def __init__(self, config: SparseScannetDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.train_json_name: str = config.train_json_name
        self.val_json_name: str = config.val_json_name
        self.test_json_name: str = config.test_json_name
        self.scale_factor: float = config.scale_factor
        self.scene_scale: float = config.scene_scale
        self.use_mask: bool=config.use_mask
        self.setup()
        
    def setup(self):
        json_dict = {'train':self.train_json_name, 'val':self.val_json_name, 'test':self.test_json_name}
        all_poses = defaultdict(list)
        all_image_filenames = defaultdict(list)
        all_depth_filenames = defaultdict(list)
        all_mask_filenames = defaultdict(list)
        CONSOLE.print(f"[green]Loading metadata from {str((self.data))} for all splits...")
        for s, name in json_dict.items():
            meta = load_from_json(self.data / Path(name))
            self.depth_unit_scale_factor = 1.0 / meta["depth_scaling_factor"]
            for frame in meta["frames"]:
                fname = self.data / Path(frame["file_path"].replace("./", ""))
                depth_fname = self.data / Path(frame["depth_file_path"].replace("./", ""))
                mask_fname = self.data / Path(frame["mask_file_path"].replace("./", ""))
                all_image_filenames[s].append(fname)
                all_depth_filenames[s].append(depth_fname)
                all_mask_filenames[s].append(mask_fname)
                all_poses[s].append(np.array(frame["transform_matrix"]))
        all_poses = {k: np.array(poses).astype(np.float32) for k, poses in all_poses.items()}
        all_poses = {k: torch.from_numpy(poses) for k, poses in all_poses.items()}
        self.all_poses = all_poses
        all_poses = torch.cat(list(all_poses.values()), dim=0)
        img_0 = imageio.v2.imread(all_image_filenames['train'][0])
        image_height, image_width = img_0.shape[:2]
        frame_0 = meta["frames"][0]
        near, far = meta["near"], meta["far"]
        bds = torch.tensor([near, far]).unsqueeze(0).float().repeat(all_poses.shape[0], 1)

        
        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(all_poses[:, :3, 3])))
        self.scale_factor *= scale_factor

        all_poses[:, :3, 3] *= self.scale_factor
        
        all_poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            all_poses,
            method="none",
            center_method=self.config.center_method,
        )
        
        all_cameras = Cameras(
            camera_to_worlds=all_poses[:, :3, :4],
            fx=frame_0['fx'],
            fy=frame_0['fy'],
            cx=frame_0['cx'],
            cy=frame_0['cy'],
            height=image_height,
            width=image_width,
            camera_type=CameraType.PERSPECTIVE,
        )
        
        coords_grid = all_cameras.get_image_coords()
        bd_coords = torch.stack([coords_grid[0, 0], coords_grid[0, -1], coords_grid[-1, 0], coords_grid[-1, -1]], dim=0).repeat(all_poses.shape[0], 1)
        bd_rays = all_cameras.generate_rays(torch.arange(all_poses.shape[0]).unsqueeze(1).repeat_interleave(4, dim=0).reshape(-1, 1), bd_coords)
        aabb_scale = self.scene_scale
        boundaries = bd_rays.origins.repeat(2, 1) + bd_rays.directions.repeat(2, 1) * bds.repeat_interleave(4, dim=0).transpose(0, 1).reshape(-1, 1)
        tight_bd_min, tight_bd_max = boundaries.min(0)[0], boundaries.max(0)[0]
        tight_bd_center = (tight_bd_min + tight_bd_max) / 2.0
        bd_min = tight_bd_min - (tight_bd_center - tight_bd_min) * 0.1
        bd_max = tight_bd_max + (tight_bd_max - tight_bd_center) * 0.1
        CONSOLE.print(f"Bounds for the scene is {bd_min} and {bd_max}")
        self.scene_box = SceneBox(aabb=torch.stack([bd_min , bd_max], dim=0) * aabb_scale)
        self.transform_matrix = transform_matrix
        self.all_image_filenames = all_image_filenames
        self.all_depth_filenames = all_depth_filenames
        self.all_mask_filenames = all_mask_filenames
        self.hwcxcyfxfy = (image_height, image_width, frame_0['cx'], frame_0['cy'], frame_0['fx'], frame_0['fy'])
        
    def _generate_dataparser_outputs(self, split="train"):
        assert self.scene_box is not None
        assert self.transform_matrix is not None
        assert self.all_image_filenames is not None
        assert self.all_poses is not None
        assert self.hwcxcyfxfy is not None
        split_poses = self.all_poses[split]
        split_image_filenames = self.all_image_filenames[split]
        split_depth_filenames = self.all_depth_filenames[split]
        split_mask_filenames = self.all_mask_filenames[split]
        h, w, cx, cy, fx, fy = self.hwcxcyfxfy
        cameras = Cameras(
            camera_to_worlds=split_poses[:, :3, :4],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=h,
            width=w,
            camera_type=CameraType.PERSPECTIVE,
        )
        
        dataparser_outputs = DataparserOutputs(
            image_filenames=split_image_filenames,
            mask_filenames=split_mask_filenames if self.use_mask else None,
            cameras=cameras,
            dataparser_transform=self.transform_matrix,
            scene_box=self.scene_box,
            dataparser_scale=self.scale_factor,
            metadata={
                "depth_filenames": split_depth_filenames if len(split_depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.depth_unit_scale_factor,
            },
        )

        return dataparser_outputs
