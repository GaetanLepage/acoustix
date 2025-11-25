import multiprocessing as mp
import os

import numpy as np

from acoustix.room import GpuRirRoom, Microphone, Source


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
        position=np.array(
            [1, 1, 1.7],
        ),
    )

    room.add_source(source)

    mic: Microphone = Microphone(
        name="mic",
        position=np.array(
            [2, 3, 1.7],
        ),
    )
    room.add_microphone(mic)

    room.pre_compute_rir()


def test_gpurir_multiprocessing() -> None:
    # Use 'spawn' context to avoid infinite hanging when running pytest
    ctx = mp.get_context("spawn")
    with ctx.Pool(os.cpu_count()) as p:
        p.map(func_room, range(100))
