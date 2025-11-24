from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..doa_encoding import DEFAULT_DOA_RESOLUTION
from ..utils import angle_between_two_vectors, rotate_2d_vector
from .utils import PolarRelativePosition, move_map, plot_map

# in meters
DEFAULT_EGOCENTRIC_MAP_SIGMA: float = 0.5


class EgocentricMap:
    def __init__(
        self,
        size: float,
        size_pixel: int,
        doa_res: int = DEFAULT_DOA_RESOLUTION,
    ) -> None:
        self.map: np.ndarray

        self.size: float = size
        self.size_pixel: int = size_pixel

        self.shape: tuple[int, int] = (self.size_pixel, self.size_pixel)

        self.doa_res: int = doa_res
        angle_indices: np.ndarray = np.zeros(
            shape=self.shape,
            dtype=np.uint16,
        )
        max_index: int = self.doa_res - 1
        for i in range(self.size_pixel):
            for j in range(self.size_pixel):
                theta: float = angle_between_two_vectors(
                    vec_1=np.array([0, 1]),  # Agent orientation
                    vec_2=self.cartesian_coords(i=i, j=j),
                )
                angle_indices[i, j] = int(
                    (theta + np.pi) * max_index / (2 * np.pi),
                )

        self._flat_angle_indices = angle_indices.flatten()
        assert self._flat_angle_indices.min() == 0
        assert self.doa_res - 2 <= self._flat_angle_indices.max() <= self.doa_res - 1

        self.sources_positions: Optional[list[PolarRelativePosition]] = None

        self.past_maps: list[np.ndarray] = []
        self.horizon: int = 1

    def cartesian_coords(self, i: int, j: int) -> np.ndarray:
        max_pixel_index: int = self.size_pixel - 1
        x: float = 0.5 - (j / max_pixel_index)
        y: float = 0.5 - (i / max_pixel_index)

        return self.size * np.array([x, y])

    def pixel_coords(self, position: np.ndarray) -> tuple[int, int]:
        assert position.shape == (2,)
        max_pixel_index: int = self.size_pixel - 1
        i: int = int(max_pixel_index * (0.5 - (position[1] / self.size)))  # i = f(y)
        j: int = int(max_pixel_index * (0.5 - (position[0] / self.size)))  # j = f(x)

        return i, j

    def to_tensor(self) -> Tensor:
        return torch.tensor(self.map)

    def get_pixel_value(
        self,
        position: PolarRelativePosition,
        use_gt_map: bool = False,
    ) -> float:
        return self.get_pixel_value_from_cartesian(
            position=position.to_cartesian(),
            use_gt_map=use_gt_map,
        )

    def get_pixel_value_from_cartesian(
        self,
        position: np.ndarray,
        use_gt_map: bool = False,
    ) -> float:
        i, j = self.pixel_coords(
            position=position,
        )

        if use_gt_map:
            return self.gt_map[i, j]

        return self.map[i, j]

    def normalize(self) -> None:
        # Ensure that values are positive
        self.map = np.maximum(self.map, 0)

        max: float = self.map.max()
        if max > 0:
            self.map /= max

    def clip(self, threshold: float) -> None:
        self.map[self.map < threshold] = 0.0

    def _single_source_gaussian(
        self,
        source_pos: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        source_x: float = source_pos[0]
        source_y: float = source_pos[1]

        def _dist(x: np.ndarray, y: np.ndarray):
            return np.sqrt(
                (x - source_x) ** 2 + (y - source_y) ** 2,
            )

        dists: np.ndarray = _dist(x_axis, y_axis)

        # dist: np.ndarray =
        return np.exp(-(dists**2) / (sigma**2)).T

    @property
    def source_relative_cartesian_positions(self) -> np.ndarray:
        if self.sources_positions is not None:
            return np.array([source_pos.to_cartesian() for source_pos in self.sources_positions])
        else:
            raise ValueError("This map has no sources")

    def compute_gt(
        self,
        source_positions: Optional[list[PolarRelativePosition]] = None,
        sigma: float = DEFAULT_EGOCENTRIC_MAP_SIGMA,
    ) -> None:
        if source_positions is None:
            assert self.sources_positions is not None
            source_positions = self.sources_positions

        axis: np.ndarray = np.linspace(
            self.size / 2,
            -self.size / 2,
            num=self.size_pixel,
        )
        x_axis: np.ndarray = axis[:, None]
        y_axis: np.ndarray = axis[None, :]

        gt_map: np.ndarray = np.zeros(
            shape=self.shape,
            dtype=np.float32,
        )

        for source_pos in source_positions:
            gaussian: np.ndarray = self._single_source_gaussian(
                source_pos=source_pos.to_cartesian(),
                x_axis=x_axis,
                y_axis=y_axis,
                sigma=sigma,
            )

            gt_map = np.maximum(
                gt_map,
                gaussian,
            )

        self.gt_map = gt_map

    def mse(self) -> float:
        return np.mean(
            (self.map - self.gt_map) ** 2,
        )

    def _plot(
        self,
        map: np.ndarray,
        detections: np.ndarray | None,
    ) -> None:
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        fig, ax = plt.subplots()
        fig.set_size_inches((10, 10))

        sources_positions: None | np.ndarray
        if self.sources_positions is not None:
            sources_positions = np.array(
                [source_pos.to_cartesian() for source_pos in self.sources_positions],
            )
        else:
            sources_positions = None

        plot_map(
            map=map,
            ax=ax,
            size_m=self.size,
            sources_positions=sources_positions,
            detections=detections,
        )
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot(
        self,
        detections: np.ndarray | None = None,
    ) -> None:
        self._plot(
            map=self.map,
            detections=detections,
        )

    def plot_gt(
        self,
        detections: np.ndarray | None = None,
    ) -> None:
        self._plot(
            map=self.gt_map,
            detections=detections,
        )

    def get_doa_map(self, doas: np.ndarray) -> np.ndarray:
        doas = np.array(doas)

        assert doas.shape == (self.doa_res,), f"{doas.shape} != ({self.doa_res},)"
        flat_values = doas[self._flat_angle_indices]

        doa_map: np.ndarray = flat_values.reshape(
            (
                self.size_pixel,
                self.size_pixel,
            )
        )

        return doa_map

    def apply_doa(
        self,
        doas: np.ndarray,
        alpha: float = 1.0,
    ) -> None:
        assert 0.0 < alpha <= 1.0

        doa_map: np.ndarray = self.get_doa_map(doas=doas)

        if alpha < 1.0:
            doa_map = alpha * doa_map + (1 - alpha) * self.map

        self.map = doa_map

    def extract_non_zero_points(
        self,
        threshold: float = 0.0,
    ) -> np.ndarray:
        points: list[np.ndarray] = []
        for i in range(self.size_pixel):
            for j in range(self.size_pixel):
                if self.map[i, j] > threshold:
                    points.append(
                        self.cartesian_coords(i=i, j=j),
                    )

        return np.array(points)

    def move(
        self,
        angle: float,
        dist: float,
    ) -> None:
        new_relative_position: np.ndarray = dist * rotate_2d_vector(
            vec=np.array([0, 1]),
            angle=angle,
        )
        self.map = move_map(
            angle=angle,
            new_relative_position=new_relative_position,
            map=self.map,
            map_size_m=self.size,
        )

        # Resetting sources as the positions are not valid anymore
        # Consider moving them automatically, though it might be prone to accumulating numerical errors
        self.sources_positions = None
