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
from typing import Type, Tuple, List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import imageio
import numpy as np
import torch
from torch import Tensor
import os
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
from subprocess import check_output





########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir: Path, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = basedir / 'images_{}'.format(r)
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = basedir / 'images_{}x{}'.format(r[1], r[0])
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
    




@dataclass
class LLFFDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: LLFF)
    """target class to instantiate"""
    data: Path = Path("data/scannet/0710")
    """Directory specifying location of data."""
    factor: int = 8
    """specify the factor to downsample images by"""
    mask_ratio: Literal[-1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = -1
    """specify the mask ratio to use for optimization"""
    spherify: bool =False
    """Whether to spherify the poses"""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scale_factor: float = 0.75
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    select_seed: int = 123
    """specify seed used to generate train and test split"""
    train_ratio: float = 0.2
    """specify from 0 to 1 the portion of images used for training set"""



@dataclass
class LLFF(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: LLFFDataParserConfig

    def __init__(self, config: LLFFDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.factor: int = config.factor
        self.scale_factor: float = config.scale_factor
        self.scene_scale: float = config.scene_scale
        self.select_seed: int = config.select_seed
        self.train_ratio: float = config.train_ratio
        self.mask_ratio: int = config.mask_ratio
        self.sfx = ''
        if self.factor is not None:
            self.sfx = '_{}'.format(self.factor)
            _minify(self.data, factors=[self.factor])
        
           
    def _load_data(self) -> Tuple[np.ndarray, List[float], np.ndarray, List[Path], Optional[List[Path]]]:
        
        poses_arr = np.load(self.data / "poses_bounds.npy")
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        bds = poses_arr[:, -2:].transpose([1,0])
        
        img0 = [self.data / Path('images') / Path(f) for f in sorted(os.listdir(self.data / Path('images'))) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape
        
        
        imgdir = self.data  / Path('images' + self.sfx)
        if not os.path.exists(imgdir):
            CONSOLE.print( imgdir, 'does not exist, existing' )
            assert False
        
        imgfiles = [imgdir / Path(f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
        
            
        if poses.shape[-1] != len(imgfiles):
            print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
            assert False
        
        sh = imageio.imread(imgfiles[0]).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1./ self.factor
        
        # get mask file based on mask_ratio
        maskfiles = None
        if self.mask_ratio > 0:
            maskfile = self.data  / Path('mask' + self.sfx) / Path(f"mask_lower_{self.mask_ratio}.png")
            if not maskfile.is_file():
                mask_save_path = self.data  / Path('mask' + self.sfx)
                CONSOLE.print(f' [red]{maskfile}[/red] does not exist, creating mask_lower_{self.mask_ratio}.png in {mask_save_path}...')
                h, w = sh[:2] 
                mask_im = np.ones([h, w])
                lower_mask_bd = int(h * (self.mask_ratio - 1) / self.mask_ratio)
                mask_im[lower_mask_bd:] = 0.
                if not mask_save_path.is_dir():
                    os.makedirs(mask_save_path, exist_ok=False)
                imageio.imsave(maskfile, (mask_im * 255).astype(np.uint8))
            maskfiles = [maskfile for _ in range(len(imgfiles))]
        
        return poses[:, :-1], poses[:, -1, 0].tolist(), bds, imgfiles, maskfiles

    def _generate_dataparser_outputs(self, split="train"):
        poses, hwf, bds, imgfiles, maskfiles = self._load_data()
            
        CONSOLE.print(f'Loaded {self.data} bd max: {bds.max()} bd min: {bds.min()}')
        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        poses = torch.from_numpy(poses)


        sc = 1. if self.scale_factor is None else 1./(bds.min() * self.scale_factor)
        bds *= sc

        poses[:, :3, 3] *= sc
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            torch.cat([poses, torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand(len(imgfiles), 1, 4)], dim=1),
            method="none",
            center_method=self.config.center_method,
        )

        
        # in x,y,z order
        aabb_scale = self.scene_scale
        scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))
        num_train = int(len(imgfiles) * self.train_ratio)
        i_perm = np.random.RandomState(self.select_seed).permutation(len(imgfiles))
        ids = i_perm[:num_train] if split == "train" else i_perm[num_train:]
        
        if maskfiles is not None and split == "train":
            maskfiles = [maskfiles[i] for i in ids]
        else:
            maskfiles = None
        print(maskfiles)
            
        H, W, focal = hwf
        cameras = Cameras(
            camera_to_worlds=poses[ids, :3, :4],
            fx=focal,
            fy=focal,
            cx=W / 2.0,
            cy=H / 2.0,
            height=int(H),
            width=int(W),
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=[imgfiles[i] for i in ids],
            mask_filenames=maskfiles,
            cameras=cameras,
            dataparser_transform=transform_matrix,
            scene_box=scene_box,
            dataparser_scale=sc,
        )

        return dataparser_outputs