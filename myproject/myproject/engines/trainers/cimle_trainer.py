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
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type, cast, List

import torch
from torch.nn import Parameter
from rich import box, style
from rich.panel import Panel
from rich.table import Table

from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from myproject.pipelines.cimle_pipeline import cIMLEPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server.viewer_state import ViewerState
from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState
from nerfstudio.engine.trainer import Trainer, TrainerConfig

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str


@dataclass
class cIMLETrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: cIMLETrainer)
    """target class to instantiate"""
    
    eval_cimle_sample_num: int = 5
    """cimle_sample num for evaluation"""
    
    eval_cimle_sample_num_per_eval_image: int = 5
    """cimle_sample num for image evaluation"""
    
    eval_cimle_sample_num_per_eval_all_images: int = 2
    """cimle_sample num for all image evaluation"""


class cIMLETrainer(Trainer):
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """
    config: cIMLETrainerConfig
    pipeline: cIMLEPipeline



    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            
            
            _, all_eval_loss_dict, all_eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step, num_samples=self.config.eval_cimle_sample_num)
            eval_losses = [functools.reduce(torch.add, eval_loss_dict.values()) for eval_loss_dict in all_eval_loss_dict]
            total_eval_loss = sum(eval_losses)
            min_eval_loss = min(eval_losses)
            max_eval_loss = max(eval_losses)
            all_eval_metrics_dict = {k: sum([d[k] for d in all_eval_metrics_dict]) / self.config.eval_cimle_sample_num for k in all_eval_metrics_dict[0]}
            all_eval_loss_dict = {k: sum([d[k] for d in all_eval_loss_dict]) / self.config.eval_cimle_sample_num for k in all_eval_loss_dict[0]}
            
            writer.put_scalar(name="Avg Eval Loss", scalar=total_eval_loss / self.config.eval_cimle_sample_num, step=step)
            writer.put_scalar(name="Max Eval Loss", scalar=max_eval_loss, step=step)
            writer.put_scalar(name="Min Eval Loss", scalar=min_eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=all_eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=all_eval_metrics_dict, step=step)


        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            group = "Eval Images"
            for i in range(self.config.eval_cimle_sample_num_per_eval_image):
                with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                    all_metrics_dicts, all_images_dicts = self.pipeline.get_eval_image_metrics_and_images(step=step, num_samples=self.config.eval_cimle_sample_num_per_eval_image)
                
                all_metrics_dicts = {k: sum([d[k] for d in all_metrics_dicts]) for k in all_metrics_dicts[0]}
                all_images_dicts = {k: [d[k] for d in all_images_dicts] for k in all_images_dicts[0]}
                total_duration = all_metrics_dicts["num_rays"] / test_t.duration
                    
                for image_name, images in all_images_dicts.items():
                    for i, image in enumerate(images):
                        writer.put_image(name=group + f"/{image_name}_sample_{i}", image=image, step=step)
                    
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=total_duration,
                step=step,
                avg_over_steps=True,
            )
            
            all_metrics_dicts = {k: v / self.config.eval_cimle_sample_num_per_eval_image for k, v in all_metrics_dicts.items()}
            writer.put_dict(name="Eval Images Metrics", scalar_dict=all_metrics_dicts, step=step)
            

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            for i in range(self.config.eval_cimle_sample_num_per_eval_all_images):
                metrics_dicts = self.pipeline.get_average_eval_image_metrics(step=step, num_samples=self.config.eval_cimle_sample_num_per_eval_all_images)
                
                metrics_dicts = {k: sum([d[k] for d in metrics_dicts]) / self.config.steps_per_eval_all_images for k in metrics_dicts[0]}
                    
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dicts, step=step)
            
            