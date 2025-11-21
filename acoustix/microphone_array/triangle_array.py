import matplotlib
import numpy as np

from ..room.audio_objects import Microphone
from .array import MicArray


class TriangleArray(MicArray):
    """
    self.mic_positions = [
        [front_mic_x, front_mic_y],
        [left_mic_x, left_mic_y],
        [right_mic_x, right_mic_y]
    ]
    """

    def __init__(
        self,
        position: np.ndarray,
        lr_mic_dist_cm: float = 2,
        front_mic_dist_cm: float = 2,
        orientation: np.ndarray = None,
        lr_mic_relative_orientation: float = 90,
        mic_pattern: str = "card",
    ) -> None:
        """
        Args:
            mic_dist (float):   The distance between the two microphones in CENTIMETERS.
        """

        self._lr_mic_dist: float = lr_mic_dist_cm / 100
        self._front_mic_dist: float = front_mic_dist_cm / 100

        assert 0 <= lr_mic_relative_orientation <= 360
        self.lr_mic_relative_orientation: float = lr_mic_relative_orientation * np.pi / 180

        super().__init__(
            n_mics=3,
            position=position,
            orientation=orientation,
            mic_pattern=mic_pattern,
        )

        if lr_mic_dist_cm < 1:
            self._logger.warning(
                "The inter-microphone distance is really short (%f). It should be provided in cm.",
                lr_mic_dist_cm,
            )

    @property
    def radius(self) -> float:
        return max(
            self._front_mic_dist,
            self._lr_mic_dist / 2,
        )

    def _init_microphones(self) -> list[Microphone]:
        mic_positions: np.ndarray
        mic_orientations: np.ndarray
        mic_positions, mic_orientations = self.get_mic_coordinates()
        front_mic: Microphone = Microphone(
            name="agent_front_mic",
            position=mic_positions[0],
            orientation=mic_orientations[0],
            pattern=self.mic_pattern,
        )
        left_mic: Microphone = Microphone(
            name="agent_left_mic",
            position=mic_positions[1],
            orientation=mic_orientations[1],
            pattern=self.mic_pattern,
        )
        right_mic: Microphone = Microphone(
            name="agent_right_mic",
            position=mic_positions[2],
            orientation=mic_orientations[2],
            pattern=self.mic_pattern,
        )

        return [
            front_mic,
            left_mic,
            right_mic,
        ]

    def _get_mic_coordinates(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the new front, left and right microphone positions.

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

        # `position` is a 3D point and `delta` is a 2x3 matrix.
        # The summation is giving the right result as it automatically broadcasts `position` to a
        # 2x2 matrix.
        mic_positions: np.ndarray = position + np.vstack(
            [
                (self._front_mic_dist) * orientation,  # front
                (self._lr_mic_dist / 2) * np.array([n_y, -n_x, 0]),  # left
                (self._lr_mic_dist / 2) * np.array([-n_y, n_x, 0]),  # right
            ]
        )

        cos_phi: float = np.cos(self.lr_mic_relative_orientation)
        sin_phi: float = np.sin(self.lr_mic_relative_orientation)
        front_mic_normal_vec: np.ndarray = orientation.copy()
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
                front_mic_normal_vec,
                left_mic_normal_vec,
                right_mic_normal_vec,
            )
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
                    mic_coordinates[0][0],  # front_mic_x
                    mic_coordinates[1][0],  # left_mic_x
                    mic_coordinates[2][0],  # right_mic_x
                    mic_coordinates[0][0],  # front_mic_x (close polygon)
                ],
                ydata=[
                    mic_coordinates[0][1],  # front_mic_y
                    mic_coordinates[1][1],  # left_mic_y
                    mic_coordinates[2][1],  # right_mic_y
                    mic_coordinates[0][1],  # front_mic_y (close polygon)
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
        # delta: np.ndarray = self.orientation[:2] / 4
        # ax.arrow(
        #     self.position[0], self.position[1],
        #     *delta,
        #     width=0.01,
        #     color=color
        # )

        # Plot microphones
        for mic, label in zip(self.microphones, ("F", "L", "R")):
            mic.plot(
                ax=ax,
                color=color,
                label=label,
            )
