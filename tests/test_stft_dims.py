import random as rd

import numpy as np

from acoustix import stft


def _compute_stft_and_check_size(
    n_channels: int = 1,
    n_signal_samples: int = 0,
    **kwargs,
) -> None:
    random_signal: np.ndarray = (
        np.random.random(
            size=(
                n_channels,
                n_signal_samples or rd.randint(1_000, 80_000),
            ),
        )
        .astype(np.float32)
        .squeeze()
    )

    stft_module: stft.StftModule = stft.StftModule(**kwargs)

    # The check for stft dimensions is done automatically in the __call__ function
    stft_module(audio_signal=random_signal)


def test_stft() -> None:
    _compute_stft_and_check_size(
        n_channels=1,
    )
    _compute_stft_and_check_size(
        n_signal_samples=7,
        window_length=3,
    )
    _compute_stft_and_check_size(
        n_channels=1,
        padded=False,
        boundary=None,
    )
    _compute_stft_and_check_size(
        n_channels=1,
        boundary=None,
    )
    _compute_stft_and_check_size(
        n_channels=1,
        padded=False,
    )
    _compute_stft_and_check_size(
        n_channels=2,
    )
    _compute_stft_and_check_size(
        n_channels=4,
    )
    _compute_stft_and_check_size(
        n_channels=4,
        n_signal_samples=2_731,
        min_freq=100,
        max_freq=8_000,
    )
    _compute_stft_and_check_size(
        n_channels=4,
        n_signal_samples=2_731,
        min_freq=100,
        max_freq=8_000,
        padded=False,
        boundary=None,
    )
