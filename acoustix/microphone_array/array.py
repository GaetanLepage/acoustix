import logging
from abc import ABC, abstractmethod
from typing import Optional

import exputils as eu
import matplotlib
import numpy as np

from ..room.audio_objects import Microphone
from ..utils import random_orientation


def compute_ild_ipd_from_stft(
    audio_stft: np.ndarray,
    ild: bool = True,
    ipd: bool = True,
) -> np.ndarray:
    """
    Compute the interaural representations (ILD, IPD) of a C-channel audio signal given its STFT.

    The number of interaural features is TODO

    Args:
        audio_stft (np.ndarray):    STFT representation of the input signal.
        ild (bool):                 Whether to compute the ILD of the signal.
        ipd (bool):                 Whether to compute the IPD of the signal.

    Returns:
        features (np.ndarray):      The interaural representation of the provided signal.
    """
    # Shape should be [C, F, T]
    assert audio_stft.ndim == 3

    num_mics: int = audio_stft.shape[0]

    num_features: int
    if num_mics < 2:
        raise ValueError("Cannot compute ILD/IPD with less than 2 microphones")

    elif num_mics == 2:
        num_features = 1

    else:
        num_features = num_mics

    assert np.iscomplexobj(audio_stft)

    _ild: np.ndarray = np.zeros(
        shape=(
            num_features,
            audio_stft.shape[1],
            audio_stft.shape[2],
        ),
    )
    _ipd: np.ndarray = np.zeros(
        shape=(
            num_features,
            audio_stft.shape[1],
            audio_stft.shape[2],
        ),
    )

    for feature_idx in range(num_features):
        spectro_1: np.ndarray = audio_stft[feature_idx]
        spectro_2: np.ndarray = audio_stft[(feature_idx + 1) % num_mics]

        # H = s_i / s_{i+1}
        assert np.abs(spectro_2).min() > 0
        ratio: np.ndarray = spectro_1 / spectro_2
        # plot_stft(spectro_1)
        # plot_stft(spectro_2)

        # ILD = 20xlog10(|H|)
        if ild:
            _ild[feature_idx] = 20.0 * np.log10(np.abs(ratio))

        # IPD = phase(H)
        if ipd:
            _ipd[feature_idx] = np.angle(ratio)

    features: np.ndarray = np.zeros(
        shape=(
            (int(ild) + int(ipd)) * num_features,
            audio_stft.shape[1],
            audio_stft.shape[2],
        ),
    )

    if ild and ipd:
        for feature_idx in range(num_features):
            features[2 * feature_idx] = _ild[feature_idx]
            features[2 * feature_idx + 1] = _ipd[feature_idx]
    elif ild:
        features = _ild
    elif ipd:
        features = _ipd

    assert features.ndim == 3
    assert features.shape[0] == num_features * (int(ild) + int(ipd))

    return features


class MicArray(ABC):
    default_config: eu.AttrDict = eu.AttrDict()

    def __init__(
        self,
        n_mics: int,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        mic_pattern: str = "omni",
    ) -> None:
        """
        Constructor for an abstract MicArray.

        Args:
            n_mics (int):               Number of microphones in this array
            position (np.ndarray):      Array position (3D vector).
            orientation (np.ndarray):   Array orientation (3D vector) (optional).
            mic_pattern (str):          The microphones pattern (optional).
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.n_mics: int = n_mics

        self.position: np.ndarray = position

        # unit vector pointing the same way as the agent
        self.orientation: np.ndarray
        if orientation is None:
            if mic_pattern != "omni":
                self._logger.warning("no orientation was provided: using a random orientation.")
            orientation = random_orientation()

        self.set_orientation(orientation=orientation)

        self.mic_pattern: str = mic_pattern

        self.microphones: list[Microphone] = self._init_microphones()
        assert len(self.microphones) == self.n_mics

    @classmethod
    def from_config(
        cls,
        config: eu.AttrDict = None,
    ) -> "MicArray":
        config = eu.combine_dicts(
            config,
            cls.default_config,
        )

        return cls(
            n_mics=config.n_mics,
            position=config.position,
            orientation=config.orientation,
            mic_pattern=config.mic_pattern,
        )

    @property
    def height(self) -> float:
        return self.position[2]

    @property
    def radius(self) -> float:
        """
        Compute the radius of the array circular footprint.

        Returns:
            radius (float): The array's radius in m.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_microphones(self) -> list[Microphone]:
        raise NotImplementedError

    def set_orientation(
        self,
        orientation: np.ndarray,
    ) -> None:
        """
        Set the orientation of the agent (and thus of all its microphones).

        Args:
            orientation (np.ndarray):   A 3d vector encoding the orientation of the AudioObject.
                                            shape: (3,)
        """

        assert orientation.ndim == 1

        if len(orientation) == 2:
            orientation = np.append(orientation, 0)
        elif len(orientation) != 3:
            raise ValueError(
                "The orientation vector should be 2d or 3d (actual length: %i)",
                len(orientation),
            )

        # Normalize the orientation if needed.
        norm: float = float(np.linalg.norm(orientation))
        if norm != 1:
            orientation /= norm

        self.orientation = orientation.copy()

    def get_mic_coordinates(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute each microphone's orientation and position for the provided array
        position/orientation.
        If no position or orientation are provided, use the array's current ones.

        Args:
            position (np.ndarray):      The new location of the array (optional).
            orientation (Orientation):  The new orientation of the array (optional).

        Returns:
            mic_positions (np.ndarray):     The microphones new positions.
            mic_orientations (np.ndarray):  The microphones new orientations.
        """
        if position is None:
            position = self.position

        if orientation is None:
            orientation = self.orientation

        mic_positions: np.ndarray
        mic_orientations: np.ndarray

        mic_positions, mic_orientations = self._get_mic_coordinates(
            position=position,
            orientation=orientation,
        )
        assert mic_positions.shape == (self.n_mics, 3)
        assert mic_orientations.shape == (self.n_mics, 3)

        return mic_positions, mic_orientations

    @abstractmethod
    def _get_mic_coordinates(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute each microphone's orientation and position for the provided array
        position/orientation.

        Args:
            position (np.ndarray):      The array position.
            orientation (Orientation):  The array orientation.

        Returns:
            mic_positions (np.ndarray):     The microphones positions.
            mic_orientations (np.ndarray):  The microphones orientations.
        """
        raise NotImplementedError

    def plot(
        self,
        ax: matplotlib.axes.Axes,
    ) -> None:
        """
        Plot the microphone array in the room's canva.

        Args:
            ax (matplotlib.axes.Axes): The room's Axes handle.
        """
        color: str = "purple"

        # Orientation
        delta: np.ndarray = self.orientation[:2] / 4
        ax.arrow(
            self.position[0],
            self.position[1],
            *delta,
            width=0.01,
            color=color,
        )

        for mic in self.microphones:
            mic.plot(
                ax=ax,
                color=color,
            )
