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
Some ray datastructures.
"""
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload, List

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: Float[Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    pixel_area: Float[Tensor, "*bs 1"]
    """Projected area of pixel a distance 1 away from origin."""
    offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""
    spacing_offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """spacing_offsets for each sample position"""
    sample_method: Literal["constant", "piecewise_linear"] = "constant"

    def get_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        
        if self.sample_method == "constant":
            pos = self.origins + self.directions * (self.starts + self.ends) / 2
        elif self.sample_method == "piecewise_linear":
            pos = torch.cat([self.origins + self.directions * self.starts, self.origins[..., -1:, :] + self.directions[..., -1:, :] * self.ends[..., -1:, :]], dim=-2)
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def get_start_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "start" position of frustum.

        Returns:
            xyz positions.
        """
        return self.origins + self.directions * self.starts

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets
        

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Returns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device: Optional[TORCH_DEVICE] = "cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )
    
    @classmethod
    def cat_frustums(cls, frustums_list: List["Frustums"]) -> "Frustums":
        first_sample = frustums_list[0]
        combined_frustums = Frustums(
            origins=torch.cat([vv.origins for vv in frustums_list]),
            ends=torch.cat([vv.ends for vv in frustums_list]),
            directions=torch.cat([vv.directions for vv in frustums_list]),
            starts=torch.cat([vv.starts for vv in frustums_list]),
            pixel_area=torch.cat([vv.pixel_area for vv in frustums_list]),
            offsets=torch.cat([vv.offsets for vv in frustums_list]) if first_sample.offsets is not None else None
        )
        return combined_frustums


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[Int[Tensor, "*bs 1 1"]] = None
    """Camera index."""
    nears: Optional[Int[Tensor, "*bs 1 1"]] = None
    """near planes for each ray"""
    fars: Optional[Int[Tensor, "*bs 1 1"]] = None
    """far planes for each ray"""
    deltas: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """"width" of each sample."""
    spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
    """additional information relevant to generating ray samples"""

    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    
    def get_weights_linear(self, densities: Float[Tensor, "*batch num_samples 1"], concat_walls: bool=True) -> Float[Tensor, "*batch num_samples 1"]:
        if concat_walls:
            densities = torch.cat([torch.ones((densities.shape[0], 1, 1), device=densities.device)*1e-10, densities, torch.ones((densities.shape[0], 1, 1), device=densities.device)*1e10], 1) # N + 3
            densities = F.relu(densities)
            deltas = torch.cat([self.frustums.starts[..., :1, :] - self.nears[..., :1, :], self.deltas, self.fars[..., -1:, :] - self.frustums.ends[..., -1:, :]], dim=-2)
            deltas = F.relu(deltas)
        else:
            deltas = self.deltas
        interval_ave_densities = 0.5 * (densities[..., 1:, :] + densities[..., :-1, :]) # N + 2 (N)
        delta_density = deltas * interval_ave_densities # N + 2 (N)
        alphas = 1 - torch.exp(-delta_density)
        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2) # N + 1 (N - 1)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        ) # N + 2 (N)
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
        # transmittance = torch.cat([transmittance, torch.zeros([transmittance.shape[0], 1, 1], device=densities.device)], dim=1)
        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights, densities, transmittance

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:-2], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)
        return weights

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[True]
    ) -> Float[Tensor, "*batch num_samples 1"]:
        ...

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[False] = False
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]]:
        ...

    @staticmethod
    def get_weights_and_transmittance_from_alphas(
        alphas: Float[Tensor, "*batch num_samples 1"], weights_only: bool = False
    ) -> Union[
        Float[Tensor, "*batch num_samples 1"],
        Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]],
    ]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )

        weights = alphas * transmittance[:, :-1, :]
        if weights_only:
            return weights
        return weights, transmittance
    
    @classmethod
    def cat_samples(cls, ray_samples_list:List["RaySamples"]) -> "RaySamples":
        first_sample = ray_samples_list[0]
        combined_frustums = Frustums.cat_frustums([ray_samples.frustums for ray_samples in ray_samples_list])
        combined_metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
        if first_sample.metadata is not None: 
            combined_metadata = {}
            for k in ray_samples_list[0].metadata.keys():
                combined_metadata[k] = torch.cat([vv.metadata[k] for vv in ray_samples_list])
        combined_sample = RaySamples(
            frustums=combined_frustums,
            metadata=combined_metadata,
            camera_indices = torch.cat([vv.camera_indices for vv in ray_samples_list]) if first_sample.camera_indices is not None else None,
            deltas = torch.cat([vv.deltas for vv in ray_samples_list]) if first_sample.deltas is not None else None,
            spacing_starts = torch.cat([vv.spacing_starts for vv in ray_samples_list]) if first_sample.spacing_starts is not None else None,
            spacing_ends = torch.cat([vv.spacing_ends for vv in ray_samples_list]) if first_sample.spacing_ends is not None else None,
            times = torch.cat([vv.times for vv in ray_samples_list]) if first_sample.times is not None else None,
            spacing_to_euclidean_fn=first_sample.spacing_to_euclidean_fn,
        )
        return combined_sample
    
        

@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    # TODO(ethan): make sure the sizes with ... are correct
    origins: Float[Tensor, "*batch 3"]
    """Ray origins (XYZ)"""
    directions: Float[Tensor, "*batch 3"]
    """Unit ray direction vector"""
    pixel_area: Float[Tensor, "*batch 1"]
    """Projected area of pixel a distance 1 away from origin"""
    camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    """Camera indices"""
    nears: Optional[Float[Tensor, "*batch 1"]] = None
    """Distance along ray to start sampling"""
    fars: Optional[Float[Tensor, "*batch 1"]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(default_factory=dict)
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self) -> int:
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self.flatten())
        indices = random.sample(range(len(self.flatten())), k=num_rays)
        return self.flatten()[tuple(indices)]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
        self,
        bin_starts: Float[Tensor, "*bs num_samples 1"],
        bin_ends: Float[Tensor, "*bs num_samples 1"],
        spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
        sample_method: Literal["constant", "piecewise_linear"] = "constant"
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        
        deltas = bin_ends - bin_starts
        spacing_offsets = None
                   
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]

        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [..., 1, 3]
            directions=shaped_raybundle_fields.directions,  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]
            ends=bin_ends,  # [..., num_samples, 1]
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1],
            spacing_offsets=spacing_offsets,
            sample_method=sample_method
        )

        ray_samples = RaySamples(
            nears = self.nears[..., None],
            fars = self.fars[..., None],
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        return ray_samples
