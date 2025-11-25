from time import time

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

N_SAMPLES: int = 1000
SAMPLE_LEN: int = 240_000
RIR_LEN: int = 4_800


def main() -> None:
    rirs: np.ndarray = np.random.rand(
        N_SAMPLES,
        RIR_LEN,
    )
    print("RIRs.shape:", rirs.shape)
    signals: np.ndarray = np.random.rand(
        N_SAMPLES,
        SAMPLE_LEN,
    )
    print("signals.shape:", signals.shape)

    start_time: float = time()
    for rir, signal in tqdm(
        zip(rirs, signals),
        total=N_SAMPLES,
    ):
        fftconvolve(
            in1=rir,
            in2=signal,
        )

    print("scipy.fftconvolve: {:2f}".format(time() - start_time))


if __name__ == "__main__":
    main()
