from abc import ABC
from typing import Optional

import matplotlib.axes
import numpy as np

from ...utils import normalize_vector

DEFAULT_SAMPLING_FREQUENCY: int = 16_000


class AudioObject(ABC):
    """
    An AudioObject is either a Microphone or a Source.
    This class gathers common aspects of those.

    Attributes:
        name (str):                         A string to define the AudioObject.
        location (np.ndarray):              The 3d location of the object. shape: (3,)
        orientation (np.ndarray):           A 3d vector encoding the orientation of the
                                                AudioObject. shape: (3,)
        room_dims (np.ndarray):             The dimensions of the room (size_x, size_y, height).
                                                shape: (3,)
        sampling_frequency (np.ndarray):    The sampling frequency associated with this object.
    """

    def __init__(
        self,
        name: str,
        *,
        default_plot_color: str,
        pattern: str,
        position: np.ndarray = np.ndarray([0, 0, 0]),
        orientation: Optional[np.ndarray] = None,
        sampling_frequency: int = DEFAULT_SAMPLING_FREQUENCY,
    ) -> None:
        """
        Init method.

        Args:
            name (str):                         A string to define the AudioObject.
            location (np.ndarray):              The 3d location of the object. shape: (3,)
            orientation (np.ndarray):           A 3d vector encoding the orientation of the
                                                    AudioObject. shape: (3,)
            sampling_frequency (np.ndarray):    The sampling frequency associated with this object.
        """
        self.name: str = name

        self.position: np.ndarray
        self.set_position(pos=position)

        self.index: int = -1

        self.orientation: Optional[np.ndarray] = None

        assert pattern in (
            "omni",
            "homni",
            "card",
            "hypcard",
            "subcard",
            "bidir",
        )
        if pattern != "omni":
            assert orientation is not None

        self.pattern: str = pattern

        if orientation is not None:
            self.set_orientation(orientation=orientation)
        else:
            self.orientation = np.zeros(shape=(3,))

        self.sampling_frequency: int = sampling_frequency

        self._default_plot_color: str = default_plot_color

        self.is_active: bool = True

    def set_position(self, pos: np.ndarray) -> None:
        """
        Set the audio object position.

        Args:
            pos (np.ndarray):   A 3d position for the AudioObject.
                                    shape: (3,)
        """
        # Case where a 2D (x, y) vector is provided
        if pos.shape == (2,):
            assert self.position is not None, (
                "`set_position` was called with a 2D position but no previous location was set."
                "In other words, the height is not set."
            )
            self.position[:2] = pos
            return

        # Case where a 3D (x, y, z) vector is provided
        assert pos.shape == (3,)
        self.position = pos.astype(np.float32)

    @property
    def height(self) -> float:
        return self.position[2]

    def set_orientation(
        self,
        orientation: np.ndarray,
    ) -> None:
        assert orientation.ndim == 1

        if len(orientation) == 2:
            orientation = np.append(orientation, 0)

        assert orientation.shape == (3,)

        # If the orientation vector is 0, do not normalize it
        if orientation.any():
            orientation = normalize_vector(vec=orientation)

        self.orientation = orientation

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        color: Optional[str] = None,
        label: str = "",
        show_label: bool = True,
    ) -> None:
        if not color:
            color = self._default_plot_color if self.is_active else "grey"

        x_coord, y_coord = self.position[:2]
        ax.scatter(
            x_coord,
            y_coord,
            color=color,
            s=100,
            marker="o" if self.is_active else "x",
        )
        if show_label:
            if not label:
                label = self.name
            ax.annotate(
                label,
                xy=(x_coord, y_coord),
                xytext=(4, 4),
                textcoords="offset points",
            )

        if self.pattern != "omni" and self.orientation is not None:
            delta: np.ndarray = self.orientation[:2] / 8

            ax.arrow(
                self.position[0],
                self.position[1],
                *delta,
                width=0.01,
                color=color,
            )
