import logging
from math import ceil, floor
from typing import Any, Generic, TypeVar

import exputils as eu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
import torch
from torch import Tensor

LOGGER: logging.Logger = logging.getLogger(__name__)


StftNumpy = npt.NDArray[np.float32]
StftTensor = torch.Tensor
Stft = StftNumpy | StftTensor


StftType = TypeVar("StftType")


class StftModule(Generic[StftType]):
    default_config: eu.AttrDict = eu.AttrDict(
        freq=16_000,
        window_length=512,
        min_freq=-1,
        max_freq=-1,
        padded=True,
        boundary="zeros",
        log_stft=False,
        plot=False,
        save_dir="",
    )

    def __init__(
        self,
        config: eu.AttrDict | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.config: eu.AttrDict = eu.combine_dicts(
            kwargs,
            config,
            self.default_config,
        )

        self.window_length: int = self.config.window_length
        self.freq: int = self.config.freq

        freqs_array: np.ndarray = np.fft.rfftfreq(
            n=self.window_length,
            d=1.0 / self.freq,
        )
        max_freq: int = freqs_array[-1]
        self._min_freq: float = self.config.min_freq
        self._min_freq_index: int = 0
        if self._min_freq > 0:
            assert self._min_freq < max_freq
            self._min_freq_index = int(
                np.argmax(freqs_array >= self._min_freq),
            )

        self._max_freq: float = self.config.max_freq
        self._max_freq_index: int | None = None
        if 0 < self._max_freq < max_freq:
            assert self._min_freq < self._max_freq <= max_freq
            self._max_freq_index = int(np.argmin(freqs_array <= self._max_freq))
            assert self._min_freq_index < self._max_freq_index <= len(freqs_array)

    def get_output_dim(
        self,
        num_signal_samples: int,
    ) -> tuple[int, int]:
        num_freq_bins: int = (self.window_length // 2) + 1

        if self._max_freq_index is not None:
            # Only take max_index-min_index frequency bins
            num_freq_bins = self._max_freq_index - self._min_freq_index
        else:
            # We remove the _min_freq_index first frequency bins
            num_freq_bins -= self._min_freq_index

        noverlap: int = self.window_length // 2
        hopsize: int = self.window_length - noverlap

        padded_signal_len: int = num_signal_samples
        # Extends the signal at both ends to make sure that the first windowed segment is centered
        # on the first input point.
        # Example for `window_length = 3`
        # [1, 2, 3, 4] -> [0, 1, 2, 3, 4, 0]
        if self.config.boundary is not None:
            padded_signal_len += 2 * noverlap

        # Extends the signal at both ends to make sure that the first windowed segment is centered
        # on the first input point.
        if self.config.padded:
            padded_signal_len += (
                -(padded_signal_len - self.window_length) % hopsize
            ) % self.window_length

        # num_time_frames: int = ceil(num_signal_samples / hopsize) + 1
        num_time_frames: int = (
            floor(
                (padded_signal_len - self.window_length) / hopsize,
            )
            + 1
        )

        return num_freq_bins, num_time_frames

    def __call__(
        self,
        audio_signal: np.ndarray,
    ) -> tuple[
        np.ndarray,  # freqs
        np.ndarray,  # times
        np.ndarray,  # stft
    ]:
        """
        Returns (freqs, times, STFT)
        """
        assert audio_signal.dtype == np.float32

        num_channels: int = 1
        assert audio_signal.ndim in (1, 2)

        if audio_signal.ndim == 2:
            num_channels = audio_signal.shape[0]

        num_signal_samples: int = audio_signal.shape[-1]

        # Compute STFT
        freqs_array: np.ndarray
        times_array: np.ndarray
        stft: np.ndarray
        freqs_array, times_array, stft = scipy.signal.stft(
            x=audio_signal,
            fs=self.config.freq,
            nperseg=self.window_length,
            padded=self.config.padded,
            boundary=self.config.boundary,
        )

        # Bandpass filtering
        match audio_signal.ndim:
            case 1:
                stft = stft[self._min_freq_index : self._max_freq_index]
            case 2:
                stft = stft[:, self._min_freq_index : self._max_freq_index]
            case _:
                assert False
        freqs_array = freqs_array[self._min_freq_index : self._max_freq_index]

        if self.config.log_stft:
            stft = np.absolute(stft)
            assert stft.dtype == np.float32
            stft = 20.0 * np.log10(stft)

        # Expected shape
        num_time_frames: int
        num_freq_bins: int
        num_freq_bins, num_time_frames = self.get_output_dim(
            num_signal_samples=num_signal_samples,
        )

        expected_shape: tuple[int, ...] = ()
        if audio_signal.ndim == 1:
            expected_shape = (
                num_freq_bins,
                num_time_frames,
            )

        else:
            expected_shape = (
                num_channels,
                num_freq_bins,
                num_time_frames,
            )

        assert stft.shape == expected_shape, (
            f"Actual shape: {stft.shape} | Expected shape: {expected_shape}"
        )

        return freqs_array, times_array, stft


def compute_stft(
    audio_signal: np.ndarray | Tensor,
    freq: float = 16_000,
    nperseg: int = 512,
    log_stft: bool = True,
    plot: bool = False,
    save_dir: str = "",
) -> Stft:
    assert audio_signal.dtype == np.float32

    num_channels: int = 1
    assert audio_signal.ndim in (1, 2)

    if audio_signal.ndim == 2:
        num_channels = audio_signal.shape[0]

    num_time_samples: int = audio_signal.shape[-1]

    complex_stft: Stft
    stft: Stft

    # Numpy array
    if isinstance(audio_signal, np.ndarray):
        complex_stft = scipy.signal.stft(
            x=audio_signal,
            fs=freq,
            nperseg=nperseg,
        )[-1]

        if log_stft:
            stft = np.absolute(complex_stft)
            assert stft.dtype == np.float32
            stft = 20.0 * np.log10(stft)
        else:
            stft = complex_stft

    # Torch tensor
    else:
        complex_stft = torch.stft(
            input=audio_signal,
            n_fft=nperseg,
            window=torch.hann_window(window_length=nperseg).cuda(),
        ).abs()

        if log_stft:
            stft = 20.0 * torch.log10(complex_stft)
        else:
            stft = complex_stft

    if plot:
        plot_stft(complex_stft[0], log=log_stft)
        plot_stft(complex_stft[1], log=log_stft)

    # Expected shape
    noverlap: int = nperseg // 2
    hopsize: int = nperseg - noverlap
    len_stft: int = ceil(num_time_samples / hopsize) + 1
    num_freq_bins: int = (nperseg // 2) + 1

    if audio_signal.ndim == 1:
        assert stft.shape == (
            num_freq_bins,
            len_stft,
        )

    else:
        assert stft.shape == (
            num_channels,
            num_freq_bins,
            len_stft,
        )

    return stft


def tensor_stft_from_signal(
    audio_signal: np.ndarray | Tensor,
    log_stft: bool = True,
) -> StftTensor:
    """
    TODO: Maybe to a backend-agnostic version (have a `backend` str argument)
    """

    audio_signal_tensor: Tensor
    if isinstance(audio_signal, np.ndarray):
        audio_signal_tensor = torch.tensor(audio_signal)
    else:
        audio_signal_tensor = audio_signal

    n_fft: int = 512
    stft: Tensor = torch.stft(
        audio_signal_tensor,
        n_fft=n_fft,
        return_complex=True,
        window=torch.hann_window(window_length=n_fft).cuda(),
    ).abs()

    if log_stft:
        stft = 20 * torch.log10(stft)

    return stft


def plot_stft(
    stft: Stft,
    times: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
    log: bool | None = True,
    vmin: float | None = None,
    vmax: float | None = None,
    close: bool = True,
) -> None:
    fig: matplotlib.figure.Figure
    axes: matplotlib.axes.Axes
    fontsize: int = 18
    fig, axes = plt.subplots(figsize=(10, 8))

    np_stft: StftNumpy
    if isinstance(stft, StftTensor):
        np_stft = stft.cpu().numpy()
    else:
        np_stft = stft

    if np_stft.ndim == 3:
        assert np_stft.shape[0] == 1
        np_stft = np_stft.squeeze()

    assert np_stft.ndim == 2, np_stft.shape
    assert np.iscomplexobj(np_stft)

    spectrogram: np.ndarray = np.abs(np_stft)

    if log:
        spectrogram = 20 * np.log10(spectrogram + 1e-6)

    LOGGER.debug(
        "min, max, avg: %f, %f, %f",
        spectrogram.min(),
        spectrogram.max(),
        spectrogram.mean(),
    )

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if times is not None and freqs is not None:
        axes.pcolormesh(
            times,
            freqs,
            spectrogram,
        )
        axes.set_xlabel("time (s)", fontsize=fontsize)
        axes.set_ylabel("frequency (Hz)", fontsize=fontsize)
    else:
        axes.pcolormesh(
            spectrogram,
            vmin=vmin,
            vmax=vmax,
        )

    # fig.tight_layout()
    plt.show()
    if close:
        plt.close()
