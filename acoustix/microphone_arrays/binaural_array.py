import matplotlib
import numpy as np

from ..room.audio_objects import Microphone
from .array import MicArray


class BinauralArray(MicArray):
    """
    Two microphone linear array.

    self.mic_positions = [
        [left_mic_x, left_mic_y],
        [right_mic_x, right_mic_y]
    ]
    """

    def __init__(
        self,
        position: np.ndarray,
        orientation: np.ndarray | None = None,
        mic_dist: float = 2,
        mic_relative_orientation: float = 45,
        mic_pattern: str = "card",
    ) -> None:
        """
        Initialize a binaural microphone array.

        Args:
            position: 3D position of the array center
            orientation: 3D orientation vector (optional)
            mic_dist: Distance between two microphones in CENTIMETERS
            mic_relative_orientation: Relative orientation of microphones in degrees
            mic_pattern: Microphone pattern (defaults to "card")
        """

        self._mic_dist: float = mic_dist / 100

        assert 0 <= mic_relative_orientation <= 360
        self.mic_relative_orientation: float = mic_relative_orientation * np.pi / 180

        super().__init__(
            n_mics=2,
            position=position,
            orientation=orientation,
            mic_pattern=mic_pattern,
        )

        self._logger.info(
            "Instanciating a binaural array (pattern='%s', dist=%fcm)",
            mic_pattern,
            mic_dist,
        )

        if mic_dist < 1:
            self._logger.warning(
                "The inter-microphone distance is really short (%f). It should be provided in cm.",
                mic_dist,
            )

    @property
    def radius(self) -> float:
        """
        Get the radius of the binaural array.

        Returns:
            Radius equal to half the inter-microphone distance
        """
        return self._mic_dist

    def _init_microphones(self) -> list[Microphone]:
        mic_positions: np.ndarray
        mic_orientations: np.ndarray
        mic_positions, mic_orientations = self.get_mic_coordinates()
        left_mic: Microphone = Microphone(
            name="binaural_agent_left_mic",
            position=mic_positions[0],
            orientation=mic_orientations[0],
            pattern=self.mic_pattern,
        )
        right_mic: Microphone = Microphone(
            name="binaural_agent_right_mic",
            position=mic_positions[1],
            orientation=mic_orientations[1],
            pattern=self.mic_pattern,
        )

        return [left_mic, right_mic]

    def _get_mic_coordinates(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the new left and right microphone positions.

        Args:
            position (np.ndarray):      The new location of the array.
            orientation (Orientation):  The new orientation of the array.

        Returns:
            mic_positions (np.ndarray):     The microphones new positions.
            mic_orientations (np.ndarray):  The microphones new orientations.
        """

        n_x: float = orientation[0]
        n_y: float = orientation[1]
        # n_z: float = orientation[2]  # TODO not handling 3d yet
        delta: np.ndarray = np.array(
            [
                [n_y, -n_x, 0],  # left
                [-n_y, n_x, 0],  # right
            ]
        )

        # `position` is a 3D point and `delta` is a 2x3 matrix.
        # The summation is giving the right result as it automatically broadcasts `position` to a
        # 2x2 matrix.
        mic_positions: np.ndarray = position + (self._mic_dist / 2) * delta

        cos_phi: float = np.cos(self.mic_relative_orientation)
        sin_phi: float = np.sin(self.mic_relative_orientation)
        left_mic_normal_vec: np.ndarray = np.array(
            [
                cos_phi * n_x + sin_phi * n_y,
                -sin_phi * n_x + cos_phi * n_y,
                0,
            ]
        )
        right_mic_normal_vec: np.ndarray = np.array(
            [
                cos_phi * n_x - sin_phi * n_y,
                sin_phi * n_x + cos_phi * n_y,
                0,
            ]
        )
        mic_orientations: np.ndarray = np.vstack(
            (
                left_mic_normal_vec,
                right_mic_normal_vec,
            ),
        )

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
                    mic_coordinates[0][0],  # left_mic_x
                    mic_coordinates[1][0],  # right_mic_x
                ],
                ydata=[
                    mic_coordinates[0][1],  # left_mic_y
                    mic_coordinates[1][1],  # right_mic_y
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

        # Orientation
        delta: np.ndarray = self.orientation[:2] / 4
        ax.arrow(
            self.position[0],
            self.position[1],
            *delta,
            width=0.01,
            color=color,
        )

        # Plot microphones
        self.microphones[0].plot(
            ax=ax,
            color=color,
            label="L",
        )
        self.microphones[1].plot(
            ax=ax,
            color=color,
            label="R",
        )
