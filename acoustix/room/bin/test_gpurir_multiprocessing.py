import os
from multiprocessing import Pool

import numpy as np

from rl_audio_nav.audio_simulator.room import GpuRirRoom, Microphone, Source


def func(cnt):
    import gpuRIR

    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(True)
    return gpuRIR.simulateRIR(
        [4, 4, 4],
        [0.0] * 6,
        np.array([[3, 2, 2]]),
        2 * np.ones((2000, 3)),
        [2] * 3,
        0.0066,
        44100,
        mic_pattern="omni",
    )


def func_room(cnt):
    room: GpuRirRoom = GpuRirRoom()
    source: Source = Source(
        name="source",
        location=np.array(
            [1, 1, 1.7],
        ),
    )

    room.add_source(source)

    mic: Microphone = Microphone(
        name="mic",
        location=np.array(
            [2, 3, 1.7],
        ),
    )
    room.add_microphone(mic)

    room.pre_compute_rir()
    print(f"OK: {cnt}")


if __name__ == "__main__":
    p = Pool(os.cpu_count())
    results = p.map(func_room, range(1000))
