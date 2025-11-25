"""
This is a script to make sure I understand the shape of the STFT.
"""

from math import floor
from typing import Optional

import numpy as np
import scipy.signal


def main() -> None:
    ##########
    # SIGNAL #
    ##########
    sampling_freq: int = 16_000
    # sampling_freq: int = 48_000
    print("fs =", sampling_freq)

    # duration_s: float = 1
    # duration_s: float = 0.512
    duration_s: float = 0.1707

    print(f"signal_duration = {duration_s}s")
    num_time_samples: int = int(duration_s * sampling_freq)
    # num_time_samples += randint(0, 2000)
    print("L =", num_time_samples)
    signal: np.ndarray = np.random.random(
        size=num_time_samples,
    )
    print(signal.shape)
    signal = scipy.signal.resample(
        x=signal,
        num=3 * num_time_samples,
    )
    print(signal.shape)

    ########
    # STFT #
    ########

    # nperseg: int = 512
    nperseg: int = 2048
    print("nperseg =", nperseg)
    noverlap: int = nperseg // 2
    print("noverlap =", noverlap)
    hopsize: int = nperseg - noverlap
    print("H =", hopsize)
    boundary: Optional[str] = "zeros"
    boundary = None
    padded: bool = True
    padded = False

    # By default, the signal is padded at the beginning and at the end so that the first window
    # is centered on the first signal sample.
    # stft: np.ndarray = scipy.signal.stft(
    f, t, stft = scipy.signal.stft(
        x=signal,
        nperseg=nperseg,
        fs=sampling_freq,
        padded=padded,
        boundary=boundary,
    )
    # print(np.where(f == 250))
    # print(np.where(f == 2500))

    ######################
    # Bandpass filtering #
    ######################
    min_freq: float = -1
    min_freq_index: int = 0
    max_freq: float = -1
    max_freq_index: Optional[int] = None

    min_freq = 100
    max_freq = 8000

    if min_freq > 0:
        min_freq_index = int(np.argmax(f >= min_freq))
        print("min_freq_index:", min_freq_index)

    if 0 < max_freq < f.max():
        max_freq_index = int(np.argmin(f <= max_freq))
        print("max_freq_index:", max_freq_index)

    stft = stft[min_freq_index:max_freq_index]

    ###############
    # Final shape #
    ###############
    print(f"Actual STFT shape: {stft.shape}")
    # len_stft: int = ceil(num_time_samples / hopsize) + 1
    padded_signal_len: int = num_time_samples
    if boundary is not None:
        padded_signal_len += 2 * (nperseg // 2)
    if padded:
        padded_signal_len += (-(padded_signal_len - nperseg) % (nperseg - noverlap)) % nperseg

    # num_time_frames: int = ceil(num_signal_samples / hopsize) + 1
    len_stft: int = floor((padded_signal_len - (nperseg - 1) - 1) / hopsize) + 1

    num_freq_bins: int = (nperseg // 2) + 1
    print(f"computed shape: {(num_freq_bins, len_stft)}")


if __name__ == "__main__":
    main()
