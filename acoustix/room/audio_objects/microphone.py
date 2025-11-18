from typing import Optional

import numpy as np

from .audio_object import AudioObject


class Microphone(AudioObject):
    """
    A microphone.

    Attributes:
        name (str):                     A name for this source to easily identify it.
        position (np.ndarray):          Location array (x, y, z) coordinates.
        pattern (str):                  The microphone pattern
        listened_signal (np.ndarray):   TODO.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        pattern: str = "omni",
    ) -> None:
        """
        Init method.

        Args:
            name (str):             A name for this source to easily identify it.
            position (np.ndarray):  Location array (x, y, z) coordinates.
            pattern (str):          The microphone pattern
        """
        super().__init__(
            name=name,
            position=position,
            pattern=pattern,
            orientation=orientation,
            default_plot_color="purple",
        )

        self.listened_signal: np.ndarray
