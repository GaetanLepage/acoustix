import numpy as np
from torch import Tensor

from acoustix.doa_encoding import encode_sources
from acoustix.egocentric_map import (
    EgocentricMap,
    PolarRelativePosition,
)


def test_gt_encoding() -> None:
    em: EgocentricMap = EgocentricMap(
        size=6,
        size_pixel=128,
        doa_res=360,
    )
    sources: list[PolarRelativePosition] = [
        PolarRelativePosition(dist=1, angle=0),
        PolarRelativePosition(dist=2, angle=np.pi / 2),
        PolarRelativePosition(dist=3, angle=-3 * np.pi / 4),
    ]
    em.sources_positions = sources
    em.compute_gt()

    np.testing.assert_allclose(
        actual=[
            em.get_pixel_value(
                position=position,
                use_gt_map=True,
            )
            for position in sources
        ],
        desired=1.0,
        rtol=1e-2,
    )
    em.plot_gt()


def test_egocentric_map() -> None:
    em: EgocentricMap = EgocentricMap(
        size=6,
        size_pixel=128,
        doa_res=360,
    )

    doas: list[float] = [
        -np.pi / 2,
        np.pi / 4,
        np.pi / 5,
    ]
    encoded_doas: Tensor = encode_sources(sources_doas=doas)
    # doas = encode_sources(sources_doas=[0, np.pi])
    em.apply_doa(doas=encoded_doas.numpy())
    em.sources_positions = [
        PolarRelativePosition(
            dist=0.4 * em.size,
            angle=angle,
        )
        for angle in doas
    ]
    em.plot()
    em.move(
        angle=0.1,
        dist=0.5,
    )
    em.plot()
