from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib.axes import Axes


def polar_to_cartesian(angle: float, dist: float) -> np.ndarray:
    return dist * np.array(
        [
            -np.sin(angle),
            np.cos(angle),
        ],
    )


@dataclass
class PolarRelativePosition:
    angle: float
    dist: float

    def to_cartesian(self) -> np.ndarray:
        return polar_to_cartesian(
            angle=self.angle,
            dist=self.dist,
        )


def plot_map(
    map: np.ndarray,
    ax: Axes,
    sources_positions: np.ndarray | None,
    size_m: float,
    detections: np.ndarray | None,
    title: bool = True,
    show_axes: bool = True,
) -> None:
    half_size: float = size_m / 2
    _title: str = f"FOV={half_size}m"

    if sources_positions is not None:
        assert sources_positions.ndim == 2
        assert sources_positions.shape[1] == 2
        _title += f" | {len(sources_positions)} sources"

    if title:
        ax.set_title(_title)
    ax.imshow(
        map,
        # origin="lower",
        extent=(
            half_size,
            -half_size,
            -half_size,
            half_size,
        ),
        vmin=0.0,
        vmax=1.0,
        cmap="grey",
    )
    ax.set_xlim(left=half_size, right=-half_size)
    ax.set_ylim(bottom=-half_size, top=half_size)

    # Show agent frame/axes
    frame_color: str = "cyan"
    ax.arrow(0, 0, dx=1, dy=0, color=frame_color, head_width=0.1)
    ax.arrow(0, 0, dx=0, dy=1, color=frame_color, head_width=0.1)

    if not show_axes:
        ax.set_axis_off()

    marker_size: int = 140

    if sources_positions is not None:
        for source_pos in sources_positions:
            ax.scatter(
                source_pos[0],
                source_pos[1],
                s=marker_size,
                marker="+",
                color="green",
            )
    if detections is not None:
        assert detections.ndim == 2
        for center in detections:
            ax.scatter(
                center[0],
                center[1],
                s=marker_size,
                marker="x",
                color="red",
            )


def _transform_position_to_map_coordinate(
    position,
    map_area,
    map_resolution,
) -> tuple[int, int]:
    r"""Transform a position (x, y) in the simulation to a coordinate (x, y) of a map of the simulation.

    Args:
        position (x,y): x,y - Position that should be transformed.
        map_area (tuple): Area ((x_min, x_max), (y_min, y_max)) of the simulation that the map is depicting.
        map_resolution (float): Resolution of the map in m per cells.

    Returns: Map coordinate as a tuple (x, y).
    """

    x_min: float = map_area[0][0]
    y_min: float = map_area[1][0]
    x_pixel: int = int((position[0] - x_min) * map_resolution)
    y_pixel: int = int((position[1] - y_min) * map_resolution)
    return x_pixel, y_pixel


def move_map(
    map: np.ndarray,
    angle: float,
    new_relative_position: np.ndarray,
    map_size_m: float,
) -> np.ndarray:
    assert map.ndim == 2
    assert map.shape[0] == map.shape[1]

    dim: int = map.shape[0]

    area: np.ndarray = np.array(
        [
            [-map_size_m / 2, map_size_m / 2],
            [-map_size_m / 2, map_size_m / 2],
        ]
    )

    pos_in_map: tuple[int, int] = _transform_position_to_map_coordinate(
        position=-np.flip(new_relative_position),
        map_area=area,
        map_resolution=dim / map_size_m,
    )

    # rotation in for cv operator is in opposite direction
    cos: float = np.cos(angle)
    sin: float = np.sin(angle)
    hvect: np.ndarray = np.array([cos, -sin])
    vvect: np.ndarray = np.array([sin, cos])

    # source points in pixel space
    rect_ll: np.ndarray = pos_in_map + (-hvect - vvect) * dim / 2
    rect_ul: np.ndarray = pos_in_map + (-hvect + vvect) * dim / 2
    rect_ur: np.ndarray = pos_in_map + (+hvect + vvect) * dim / 2
    rect_lr: np.ndarray = pos_in_map + (+hvect - vvect) * dim / 2
    src_pts: np.ndarray = np.vstack(
        [
            rect_ll,
            rect_ul,
            rect_ur,
            rect_lr,
        ]
    ).astype(np.float32)

    # destination points in pixel space
    dst_pts: np.ndarray = np.array(
        [
            [0, 0],
            [0, dim],
            [dim, dim],
            [dim, 0],
        ],
        dtype=np.float32,
    )

    # use transpose operations to bring image into coordinate frame of cv and then back to our coordinate system
    shifted_map: np.ndarray = cv2.warpPerspective(
        src=map.T,
        M=cv2.getPerspectiveTransform(src_pts, dst_pts),
        dsize=(dim, dim),
    ).T

    assert map.shape == shifted_map.shape

    return shifted_map
