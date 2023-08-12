# # Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Data parser for blender dataset"""
# from __future__ import annotations

# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Type, Optional
# from nerfstudio.utils.rich_utils import CONSOLE
# from collections import defaultdict
# import imageio
# import numpy as np
# import torch

# from nerfstudio.cameras.cameras import Cameras, CameraType
# from nerfstudio.data.dataparsers.base_dataparser import (
#     DataParser,
#     DataParserConfig,
#     DataparserOutputs,
# )
# from nerfstudio.data.scene_box import SceneBox
# from nerfstudio.utils.colors import get_color
# from nerfstudio.utils.io import load_from_json


# @dataclass
# class BlenderDataParserConfig(DataParserConfig):
#     """Blender dataset parser config"""

#     _target: Type = field(default_factory=lambda: Blender)
#     """target class to instantiate"""
#     data: Path = Path("data/blender/lego")
#     """Directory specifying location of data."""
#     scale_factor: float = 1.0
#     """How much to scale the camera origins by."""
#     alpha_color: str = "white"
#     """alpha color of background"""


# @dataclass
# class Blender(DataParser):
#     """Blender Dataset
#     Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
#     """

#     config: BlenderDataParserConfig
#     scene_box: Optional[SceneBox] = None

#     def __init__(self, config: BlenderDataParserConfig):
#         super().__init__(config=config)
#         self.data: Path = config.data
#         self.scale_factor: float = config.scale_factor
#         self.alpha_color = config.alpha_color
#         self.setup()

#     def setup(self):
#         json_dict = ["train", "val", "test"]
#         all_poses = defaultdict(list)
#         all_image_filenames = defaultdict(list)
#         all_depth_filenames = defaultdict(list)
#         CONSOLE.print(f"[green]Loading metadata from {str((self.data))} for all splits...")
#         for split in json_dict:
#             meta = load_from_json(self.data / f"transforms_{split}.json")
#             for frame in meta["frames"]:
#                 fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
#                 all_image_filenames[split].append(fname)
#                 all_poses[split].append(np.array(frame["transform_matrix"]))
#         all_poses = {k: np.array(poses).astype(np.float32) for k, poses in all_poses.items()}
#         all_poses = {k: torch.from_numpy(poses) for k, poses in all_poses.items()}
#         self.all_poses = all_poses
#         all_poses = torch.cat(list(all_poses.values()), dim=0)
#         img_0 = imageio.v2.imread(all_image_filenames['train'][0])
#         image_height, image_width = img_0.shape[:2]
#         near, far = 2.0, 6.0
#         bds = torch.tensor([near, far]).unsqueeze(0).float().repeat(all_poses.shape[0], 1)

        
#         # Scale poses
#         self.scale_factor = 1.0
        
#         img_0 = imageio.v2.imread(all_image_filenames["train"][0])
#         image_height, image_width = img_0.shape[:2]
#         camera_angle_x = float(meta["camera_angle_x"])
#         focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

#         cx = image_width / 2.0
#         cy = image_height / 2.0
#         camera_to_world = all_poses[:, :3]  # camera to world transform

        
#         all_cameras = Cameras(
#             camera_to_worlds=camera_to_world,
#             fx=focal_length,
#             fy=focal_length,
#             cx=cx,
#             cy=cy,
#             height=image_height,
#             width=image_width,
#             camera_type=CameraType.PERSPECTIVE,
#         )
        
#         coords_grid = all_cameras.get_image_coords()
#         bd_coords = torch.stack([coords_grid[0, 0],coords_grid[0, -1],coords_grid[-1, 0], coords_grid[-1, -1]], dim=0).repeat(all_poses.shape[0], 1)
#         bd_rays = all_cameras.generate_rays(torch.arange(all_poses.shape[0]).unsqueeze(1).repeat_interleave(4, dim=0).reshape(-1, 1), bd_coords)
#         aabb_scale = 1.5
#         boundaries = bd_rays.origins.repeat(2, 1) + bd_rays.directions.repeat(2, 1) * bds.repeat_interleave(4, dim=0).transpose(0, 1).reshape(-1, 1)
#         tight_bd_min, tight_bd_max = boundaries.min(0)[0], boundaries.max(0)[0]
#         tight_bd_center = (tight_bd_min + tight_bd_max) / 2.0
#         bd_min = tight_bd_min - (tight_bd_center - tight_bd_min) * 0.1
#         bd_max = tight_bd_max + (tight_bd_center - tight_bd_max) * 0.1
#         CONSOLE.print(f"Bounds for the scene is {bd_min} and {bd_max}")
#         self.scene_box = SceneBox(aabb=torch.stack([bd_min , bd_max], dim=0) * aabb_scale)

#     def _generate_dataparser_outputs(self, split="train"):
#         if self.alpha_color is not None:
#             alpha_color_tensor = get_color(self.alpha_color)
#         else:
#             alpha_color_tensor = None

#         meta = load_from_json(self.data / f"transforms_{split}.json")
#         image_filenames = []
#         poses = []
#         for frame in meta["frames"]:
#             fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
#             image_filenames.append(fname)
#             poses.append(np.array(frame["transform_matrix"]))
#         poses = np.array(poses).astype(np.float32)

#         img_0 = imageio.v2.imread(image_filenames[0])
#         image_height, image_width = img_0.shape[:2]
#         camera_angle_x = float(meta["camera_angle_x"])
#         focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

#         cx = image_width / 2.0
#         cy = image_height / 2.0
#         camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

#         # in x,y,z order
#         camera_to_world[..., 3] *= self.scale_factor

#         cameras = Cameras(
#             camera_to_worlds=camera_to_world,
#             fx=focal_length,
#             fy=focal_length,
#             cx=cx,
#             cy=cy,
#             camera_type=CameraType.PERSPECTIVE,
#         )

#         dataparser_outputs = DataparserOutputs(
#             image_filenames=image_filenames,
#             cameras=cameras,
#             alpha_color=alpha_color_tensor,
#             scene_box=self.scene_box,
#             dataparser_scale=self.scale_factor,
#         )

#         return dataparser_outputs


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

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    use_object_mask: bool = False
    """Whether to use object mask """


@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.use_object_mask = config.use_object_mask

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        mask_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            mask_fname = self.data / Path(frame["mask_file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            mask_filenames.append(mask_fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames if self.use_object_mask else None,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
