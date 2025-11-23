import matplotlib
import numpy as np

from ..room.audio_objects import Microphone
from ..utils import rotate_3d_vector
from .array import MicArray


class SquareArray(MicArray):
    """
    self.mic_positions = [
        [front_left_mic_x, front_left_mic_y],
        [front_right_mic_x, front_right_mic_y],
        [back_left_mic_x, back_left_mic_y],
        [back_right_mic_x, back_right_mic_y],
    ]
    """

    def __init__(
        self,
        position: np.ndarray,
        center_to_mic_dist: float = 2,
        orientation: np.ndarray | None = None,
        mic_pattern: str = "omni",
    ) -> None:
        """
        Args:
            mic_dist (float):   The distance between the two microphones in CENTIMETERS.
        """

        # Convert to meters
        self._center_to_mic_dist: float = center_to_mic_dist / 100

        assert mic_pattern == "omni", (
            "The square mic array only support omnidirectional microphones for now"
        )

        super().__init__(
            n_mics=4,
            position=position,
            orientation=orientation,
            mic_pattern=mic_pattern,
        )

        if center_to_mic_dist < 1:
            self._logger.warning(
                "The inter-microphone distance is really short (%f). It should be provided in cm.",
                center_to_mic_dist,
            )

    @property
    def radius(self) -> float:
        return 2 * self._center_to_mic_dist

    def _init_microphones(self) -> list[Microphone]:
        mic_positions: np.ndarray
        mic_orientations: np.ndarray
        mic_positions, mic_orientations = self.get_mic_coordinates()
        front_right_mic: Microphone = Microphone(
            name="agent_front_right_mic",
            position=mic_positions[0],
            orientation=mic_orientations[0],
        )
        front_left_mic: Microphone = Microphone(
            name="agent_front_left_mic",
            position=mic_positions[1],
            orientation=mic_orientations[1],
        )
        back_right_mic: Microphone = Microphone(
            name="agent_back_right_mic",
            position=mic_positions[2],
            orientation=mic_orientations[2],
        )
        back_left_mic: Microphone = Microphone(
            name="agent_back_left_mic",
            position=mic_positions[3],
            orientation=mic_orientations[3],
        )

        return [
            front_right_mic,
            front_left_mic,
            back_right_mic,
            back_left_mic,
        ]

    def _get_mic_coordinates(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the new microphone positions.

        Args:
            position (np.ndarray):      The new location of the array.
            orientation (Orientation):  The new orientation of the array.

        Returns:
            mic_positions (np.ndarray):     The microphones new positions.
            mic_orientations (np.ndarray):  The microphones new orientations.
        """

        # WARNING: this is for the simulator, down-looking basis:
        # o-→
        # ↓
        front_left_dir: np.ndarray = rotate_3d_vector(
            vec=orientation,
            angle_xy=-np.pi / 4,
        )
        front_right_dir: np.ndarray = rotate_3d_vector(
            vec=orientation,
            angle_xy=np.pi / 4,
        )
        back_left_dir: np.ndarray = rotate_3d_vector(
            vec=orientation,
            angle_xy=-3 * np.pi / 4,
        )
        back_right_dir: np.ndarray = rotate_3d_vector(
            vec=orientation,
            angle_xy=3 * np.pi / 4,
        )

        mic_positions: np.ndarray = position + self._center_to_mic_dist * np.vstack(
            [
                front_right_dir,
                front_left_dir,
                back_right_dir,
                back_left_dir,
            ]
        )

        # All zeros
        mic_orientations: np.ndarray = np.zeros(
            shape=(self.n_mics, 3),
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
                    mic_coordinates[0][0],  # front_right_mic_x
                    mic_coordinates[2][0],  # back_right_mic_x
                    mic_coordinates[3][0],  # back_left_mic_x
                    mic_coordinates[1][0],  # front_left_mic_x
                    mic_coordinates[0][0],  # front_right_mic_x (close polygon)
                ],
                ydata=[
                    mic_coordinates[0][1],  # front_right_mic_y
                    mic_coordinates[2][1],  # back_right_mic_y
                    mic_coordinates[3][1],  # back_left_mic_y
                    mic_coordinates[1][1],  # front_left_mic_y
                    mic_coordinates[0][1],  # front_right_mic_y (close polygon)
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
        for mic, label in zip(self.microphones, ("FR", "FL", "BR", "BL")):
            mic.plot(
                ax=ax,
                color=color,
                label=label,
            )
