import numpy as np
import scipy
import torch
from torch import Tensor

from acoustix.datasets.librispeech import get_speech_data_set
from acoustix.stft import StftTensor, plot_stft


def test_stft() -> None:
    for speech_signal in get_speech_data_set(
        random=False,
        n_samples=2,
        load_audio_array=True,
        load_audio_tensor=True,
    ):
        print("########################################")
        print("Scipy STFT:")
        times: np.ndarray
        freqs: np.ndarray
        stft_scipy: np.ndarray
        freq: int = 16_000
        freqs, times, stft_scipy = scipy.signal.stft(
            x=speech_signal.signal[: 2 * freq],
            fs=freq,
        )
        # stft_scipy = np.absolute(stft_scipy)

        print(f"shape: {stft_scipy.shape}")
        print(f"stft dtype: {stft_scipy.dtype}")

        plot_stft(
            times=times,
            freqs=freqs,
            stft=stft_scipy,
            log=False,
        )
        log_stft_scipy: np.ndarray = 20 * np.log10(stft_scipy)
        print(log_stft_scipy.dtype)
        plot_stft(
            times=times,
            freqs=freqs,
            stft=stft_scipy,
            log=True,
        )

        print("########################################")
        print("Torch STFT:")
        signal_tensor: Tensor = speech_signal.signal_tensor
        signal_tensor = signal_tensor.cuda()

        n_stft: int = 512
        stft_torch: StftTensor = torch.stft(
            signal_tensor[:32000],
            n_fft=n_stft,
            return_complex=True,
            window=torch.hann_window(window_length=n_stft).cuda(),
        )

        print(f"shape: {stft_torch.shape}")
        plot_stft(
            stft=stft_torch.cpu().numpy(),
            log=False,
        )
        plot_stft(
            stft=stft_torch.cpu().numpy(),
            log=True,
        )
        # LOGGER.debug("stft_torch dimensions: %s", stft_torch.shape)
        # stft: np.ndarray = librosa.stft(speech_signal.signal,
        #                                 n_fft=256)
        # LOGGER.debug("stft shape: %s", stft.shape)

        # Take the modulus of the complex stft coefficients.
        # LOGGER.debug("device: %s", stft.device)
