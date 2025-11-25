import numpy as np

from ..room.audio_objects import Microphone
from .array import MicArray


class MonoArray(MicArray):
    """
    Single microphone array implementation.
    """

    def __init__(
        self,
        position: np.ndarray,
        orientation: np.ndarray = None,
        mic_pattern: str = "omni",
    ) -> None:
        """
        Initialize a mono microphone array.

        Args:
            position: 3D position of the microphone
            orientation: 3D orientation vector (optional, defaults to [0, -1, 0])
            mic_pattern: Microphone pattern (defaults to "omni")
        """
        if orientation is None:
            assert mic_pattern == "omni"
            orientation = np.array(
                [0, -1, 0],
                dtype=np.float32,
            )

        super().__init__(
            n_mics=1,
            position=position,
            orientation=orientation,
            mic_pattern=mic_pattern,
        )

    def _init_microphones(self) -> list[Microphone]:
        """
        Initialize the single microphone for this array.

        Returns:
            List containing the single microphone
        """
        mic: Microphone = Microphone(
            name="mono_agent_mic",
            position=self.position,
            orientation=self.orientation,
            pattern=self.mic_pattern,
        )

        return [mic]

    def _get_mic_coordinates(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get microphone coordinates for the mono array.

        Args:
            position: Array position
            orientation: Array orientation

        Returns:
            Tuple of (mic_position, mic_orientation)
        """
        return position, orientation
