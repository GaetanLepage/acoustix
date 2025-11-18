import matplotlib
import numpy as np

from ..room.audio_objects import Microphone
from .array import MicArray


class UniformLinearArray(MicArray):
    def __init__(
        self,
        position: np.ndarray,
        n_mics,
        inter_mic_dist_cm: float = 2,
        orientation: np.ndarray = None,
        mic_pattern: str = "card",
    ) -> None:
        """
        Args:
            mic_dist (float):   The distance between the two microphones in CENTIMETERS.
        """

        self._mic_dist: float = inter_mic_dist_cm / 100

        assert n_mics >= 2, "A UniformLinearArray should have at least two microphones"

        super().__init__(
            n_mics=n_mics,
            position=position,
            orientation=orientation,
            mic_pattern=mic_pattern,
        )

        if inter_mic_dist_cm < 1:
            self._logger.warning(
                "The inter-microphone distance is really short (%f). It should be provided in cm.",
                inter_mic_dist_cm,
            )

    @property
    def radius(self) -> float:
        diameter: float = (self.n_mics - 1) * self._mic_dist
        return diameter / 2

    def _init_microphones(self) -> list[Microphone]:
        mic_positions: np.ndarray
        mic_orientations: np.ndarray
        mic_positions, mic_orientations = self.get_mic_coordinates()

        return [
            Microphone(
                name=f"mic_{i}",
                position=mic_positions[i],
                orientation=mic_orientations[i],
                pattern=self.mic_pattern,
            )
            for i in range(self.n_mics)
        ]

    def _get_mic_coordinates(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert orientation[2] == 0, "No 3D support yet"

        n_x: float = orientation[0]
        n_y: float = orientation[1]
        pos_2d: np.ndarray = position[0:2]
        height: float = position[2]
        orthog_vector: np.ndarray = np.array([-n_y, n_x])

        mic_positions: np.ndarray = np.zeros(shape=(self.n_mics, 3))
        for i in range(self.n_mics):
            mic_positions[i][0:2] = (
                pos_2d + self._mic_dist * (i - (self.n_mics - 1) / 2) * orthog_vector
            )
            mic_positions[i][2] = height

        print(orientation.shape)
        # All microphones share the array's orientation
        mic_orientations: np.ndarray = np.broadcast_to(
            orientation,
            (self.n_mics, 3),
        )
        print(mic_orientations.shape)

        return mic_positions, mic_orientations

    def plot(
        self,
        ax: matplotlib.axes.Axes,
    ) -> None:
        color: str = "purple"
        mic_coordinates: np.ndarray = self.get_mic_coordinates()[0]
        ax.add_line(
            matplotlib.lines.Line2D(
                xdata=[
                    mic_coordinates[0][0],  # first mic X
                    mic_coordinates[-1][0],  # last mic X
                ],
                ydata=[
                    mic_coordinates[0][1],  # first mic Y
                    mic_coordinates[-1][1],  # last mic Y
                ],
                color=color,
                linestyle="--",
            )
        )

        # Position
        x_coord, y_coord = self.position[:2]
        ax.scatter(
            x_coord,
            y_coord,
            s=100,
            marker="D",
            color=color,
        )

        # Plot microphones
        for mic in self.microphones:
            mic.plot(
                ax=ax,
                color=color,
            )
