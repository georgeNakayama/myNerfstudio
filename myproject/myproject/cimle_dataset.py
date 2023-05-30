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
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

from jaxtyping import Float
from torch import Tensor
from nerfstudio.data.datasets.base_dataset import InputDataset

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs


class cIMLEDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    cached_latents: Optional[Float[Tensor, "num_images num_channels"]] = None

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)
   
    def set_cimle_latent(self, selected_zs: Float[Tensor, "num_images num_channels"]) -> None:
        """sets cimle latents to each image

        Args:
            selected_zs: selected latents per image [num_images, cimle_ch].
        """
        assert selected_zs.shape[0] == self.__len__()
        self.cached_latents = selected_zs.detach().clone()

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        metadata = {}
        if self.cached_latents is not None:
            metadata['cimle_latent'] = self.cached_latents[data['image_idx']].clone()
        del data
        return metadata
