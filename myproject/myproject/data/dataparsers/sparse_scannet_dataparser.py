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
from typing import Type

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import imageio
import numpy as np
import torch

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
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""


@dataclass
class SparseScannet(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: SparseScannetDataParserConfig

    def __init__(self, config: SparseScannetDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.train_json_name: str = config.train_json_name
        self.val_json_name: str = config.val_json_name
        self.test_json_name: str = config.test_json_name
        self.scale_factor: float = config.scale_factor
        self.scene_scale: float = config.scene_scale

    def _generate_dataparser_outputs(self, split="train"):
        json_name = {'train':self.train_json_name, 'val':self.val_json_name, 'test':self.test_json_name}[split]
        meta = load_from_json(self.data / Path(json_name))
        CONSOLE.print(f"[green]Loading metadata from {str((self.data / Path(json_name)))} for split {split}...")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", ""))
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        poses = torch.from_numpy(poses)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        frame_0 = meta["frames"][0]

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="none",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.scale_factor

        poses[:, :3, 3] *= scale_factor
        
        # in x,y,z order
        aabb_scale = self.scene_scale
        scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))
        CONSOLE.print(frame_0['fx'], frame_0['fy'], frame_0['cx'], frame_0['cy'], image_height,image_width)
        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=frame_0['fx'],
            fy=frame_0['fy'],
            cx=frame_0['cx'],
            cy=frame_0['cy'],
            height=image_height,
            width=image_width,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            dataparser_transform=transform_matrix,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
        )

        return dataparser_outputs
