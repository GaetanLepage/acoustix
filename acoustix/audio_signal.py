import logging
from os.path import basename
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from scipy.io import wavfile

from . import utils

AudioSignalNumpy = npt.NDArray[Union[np.float32, np.int16]]
AudioSignalNumpyInt16 = npt.NDArray[np.int16]
AudioSignalNumpyFloat32 = npt.NDArray[np.float32]
AudioSignalTensor = torch.Tensor
AudioSignal = Union[AudioSignalNumpy, AudioSignalTensor]


class AudioSignalObject:
    """
    Abstraction on top of mono-channel sound file stored on disk.
    This class allows to load, play and plot any sound signals.
    """

    def __init__(
        self,
        file_path: str,
        name: str = "",
        load_audio_array: bool = False,
        load_audio_tensor: bool = False,
    ) -> None:
        """
        Initialize an AudioSignalObject.

        Args:
            file_path: Path to the audio file on disk
            name: Optional name for the audio signal
            load_audio_array: Whether to load the audio as numpy array on initialization
            load_audio_tensor: Whether to load the audio as torch tensor on initialization
        """
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.name: str = name
        self.file_path: str = file_path
        self.filename: str = basename(file_path)

        self._raw_signal: AudioSignalNumpy
        self.signal: AudioSignalNumpy
        self.sample_rate: int
        self._signal_tensor: AudioSignalTensor
        self.signal_shape: tuple
        self.signal_dim: int

        if load_audio_array:
            self.load_audio()

        if load_audio_tensor:
            self.load_audio_tensor()

    def load_audio(
        self,
        n_samples: int = -1,
        random_start_index: bool = False,
    ) -> None:
        """
        Load a (single-channel) audio signal from disk, optionnaly trim it to the required length
        and store it in the `signal` attribute.

        Warning: Only loads the first channel of multi-channel audio signals.
        """
        # Load raw signal from file if necessary
        if (not hasattr(self, "_raw_signal")) or self._raw_signal is None:
            self.sample_rate, self._raw_signal = wavfile.read(
                filename=self.file_path,
            )
            # Multi-channel
            if self._raw_signal.ndim == 2:
                self._raw_signal = self._raw_signal.T
                # TODO we take only 1st channel
                self._raw_signal = self._raw_signal[0]

        self.signal = self._raw_signal.copy()

        signal_length: int = self.signal.shape[-1]
        if 0 < n_samples < signal_length:
            start_index: int = 0
            trimmed_signal: np.ndarray = np.zeros(n_samples)

            if random_start_index:
                # Some signal portions of audio signals can be all zero.
                # In this case, we need to pick another part of the recording. Otherwise, this can
                # crash the policy network by leading to 'nan'.
                while np.all(trimmed_signal == 0):
                    start_index = np.random.randint(
                        low=0,
                        high=signal_length - n_samples,
                    )
                    trimmed_signal = self.signal[start_index : (start_index + n_samples)]
            else:
                trimmed_signal = self.signal[start_index : (start_index + n_samples)]

            assert trimmed_signal.shape == (n_samples,), (
                f"trimmed_signal.shape ({trimmed_signal.shape}) != (n_samples,) = ({n_samples},)"
            )

            self.signal = trimmed_signal

        self.signal_shape = self.signal.shape
        self.signal_dim = self.signal.ndim

    def load_audio_tensor(self) -> None:
        """
        Load the audio file as a torch tensor.

        Directly exits if the signal is already loaded.
        """
        # Directly exit if the signal is already loaded
        if hasattr(self, "_signal_tensor") and self.signal_tensor is not None:
            return

        self._signal_tensor, self.sample_rate = torchaudio.load(uri=self.file_path)
        self._signal_tensor = self._signal_tensor.squeeze()
        self.signal_shape = self._signal_tensor.shape
        self.signal_dim = self.signal_tensor.dim()

    def unload_audio(self) -> None:
        """
        Unload the audio data to free memory
        """
        # Regular audio
        if hasattr(self, "_raw_signal"):
            del self._raw_signal

        if hasattr(self, "signal"):
            del self.signal

        if hasattr(self, "_signal_tensor"):
            del self.signal_tensor

        if hasattr(self, "signal_shape"):
            del self.signal_shape

        if hasattr(self, "signal_ndim"):
            del self.signal_ndim

    @property
    def signal_tensor(self) -> torch.Tensor:
        if self._signal_tensor is None:
            self.load_audio_tensor()

        return self._signal_tensor

    @property
    def normalized_signal(self) -> AudioSignalNumpyFloat32:
        """
        Get the audio signal as float32 normalized array.

        Returns:
            The audio signal converted to float32 format
        """
        return utils.to_float32(signal=self.signal)

    @property
    def n_channels(self) -> int:
        """
        Get the number of channels in the audio signal.

        Returns:
            Number of audio channels (1 for mono, 2 for stereo, etc.)
        """
        if self.signal_dim > 1:
            return self.signal_shape[0]

        # Mono signal
        return 1

    @property
    def n_samples(self) -> int:
        """
        Get the number of samples in the audio signal.

        Returns:
            Number of audio samples

        Raises:
            AttributeError: If the signal has not been loaded yet
        """
        if not hasattr(self, "signal_shape") or self.signal_shape is None:
            raise AttributeError(
                "You have to load the signal before asking for the number of samples."
            )
        return self.signal_shape[-1]

    @property
    def duration(self) -> float:
        """
        Get the duration of the audio signal in seconds.

        Returns:
            Duration in seconds
        """
        return self.n_samples / self.sample_rate

    def is_mono(self) -> bool:
        """
        Check if the audio signal is mono.

        Returns:
            True if the signal has exactly one channel
        """
        return self.n_channels == 1

    def is_stereo(self) -> bool:
        """
        Check if the audio signal is stereo.

        Returns:
            True if the signal has exactly two channels
        """
        return self.n_channels == 2

    def play(self) -> None:
        """
        Play the audio signal through the system audio output.
        """
        utils.play_audio(
            audio_signal=self.signal,
            sample_rate=self.sample_rate,
            num_channels=self.n_channels,
            bytes_per_sample=2,
        )

    def save(self, path: str) -> None:
        """
        Save the audio signal to a WAV file.

        Args:
            path: Path where to save the audio file
        """
        wavfile.write(
            filename=path,
            rate=self.sample_rate,
            data=self.signal,
        )

    @staticmethod
    def plot_signals(signals: list["AudioSignalObject"]) -> None:
        """
        Plot multiple audio signals on the same graph.

        Args:
            signals: List of AudioSignalObject instances to plot
        """
        for signal in signals:
            if signal.signal is None:
                signal.load_audio()

            times: np.ndarray = np.arange(
                start=0,
                stop=signal.duration,
            )

            plt.plot(
                times,
                signal.signal,
                label=signal.name,
            )

        plt.legend()
        plt.show()


class SpeechSignal(AudioSignalObject):
    """
    Audio signal consisting in a speech sentence.
    """

    def __init__(
        self,
        file_path: str,
        transcript: str,
        load_audio_array: bool = False,
        load_audio_tensor: bool = False,
    ) -> None:
        """
        Initialize a SpeechSignal object.

        Args:
            file_path: Path to the audio file on disk
            transcript: Text transcript of the speech content
            load_audio_array: Whether to load the audio as numpy array on initialization
            load_audio_tensor: Whether to load the audio as torch tensor on initialization
        """
        super().__init__(
            file_path=file_path,
            load_audio_array=load_audio_array,
            load_audio_tensor=load_audio_tensor,
        )

        self.transcript: str = transcript
        self.num_words: int = len(transcript.split(" "))
