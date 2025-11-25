from typing import Sequence

import torch
from torch import Tensor

from .utils import angular_dist_torch

DEFAULT_DOA_RESOLUTION: int = 360
DEFAULT_SIGMA: float = 5 * torch.pi / 180


def get_ref_doas(doa_resolution: int) -> Tensor:
    """
    Generate reference DOA angles spanning the full circle.

    Args:
        doa_resolution: Number of DOA angles to generate

    Returns:
        Tensor of DOA angles from -π to π
    """
    ref_doas: Tensor = torch.linspace(
        start=-torch.pi,
        end=torch.pi,
        steps=doa_resolution,
    )
    return ref_doas


def _get_distance_matrix(
    gt_doas: Tensor,
    doa_resolution: int,
) -> Tensor:
    """
    Compute angular distance matrix between ground truth DOAs and reference DOAs.

    Args:
        gt_doas: Ground truth DOA angles
        doa_resolution: Number of reference DOA angles

    Returns:
        Distance matrix of shape [N_sources, DOA_resolution]
    """
    ref_doas: Tensor = get_ref_doas(doa_resolution=doa_resolution)

    # Shape: [N_sources, DOA_res]
    dist_matrix: Tensor = torch.stack(
        [
            angular_dist_torch(
                theta_1=src_doa,
                theta_2=ref_doas,
            )
            for src_doa in gt_doas
        ]
    )

    assert dist_matrix.shape == (
        len(gt_doas),
        doa_resolution,
    )

    return dist_matrix


def encode_sources(
    sources_doas: Tensor | Sequence[float],
    sigma: float = DEFAULT_SIGMA,
    doa_resolution: int = DEFAULT_DOA_RESOLUTION,
    heats: Tensor | list[float] | float = 1.0,
) -> Tensor:
    """
    Encode a list of DoA angles (in radians) into a DoA spectrum representation.

    See the following paper for more information on this format:
        Neural Network Adaptation and Data Augmentation for Multi-Speaker Direction-of-Arrival Estimation
        Weipeng He; Petr Motlicek; Jean-Marc Odobez
        https://ieeexplore.ieee.org/document/9357962

    Args:
        sources_doas: A list of DoA angles in radians
        sigma: Standard deviation of the Gaussian encoding
        doa_resolution: Number of DOA angles in the output spectrum
        heats: Heat values for each source (can be scalar, list, or tensor)

    Returns:
        DOA spectrum encoding as a tensor
    """
    doa_vec: Tensor = torch.zeros(
        size=(doa_resolution,),
    )

    if not torch.is_tensor(sources_doas):
        sources_doas = torch.tensor(sources_doas)
    assert isinstance(sources_doas, Tensor)

    # heats is a scalar
    if isinstance(heats, float) or (isinstance(heats, Tensor) and heats.size == (1,)):
        heats = heats * torch.ones((len(sources_doas)))
    # list of scalars
    if isinstance(heats, list):
        assert len(heats) == len(sources_doas)
    # 1-d tensor
    else:
        assert heats.shape == (len(sources_doas),)

    if len(sources_doas) == 0:
        return doa_vec

    dist_matrix = _get_distance_matrix(
        gt_doas=sources_doas,
        doa_resolution=doa_resolution,
    )

    for source_idx, dist in enumerate(dist_matrix):
        # maximum of the Gaussian curves
        doa_vec = torch.maximum(
            input=doa_vec,
            other=heats[source_idx] * torch.exp(-(dist**2) / sigma**2),
        )

    return doa_vec
