import random
from time import time
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

import acoustix as acx
from acoustix.microphone_arrays import BinauralArray


def benchmark_simulator(
    backend: Literal["gpurir", "pyroomacoustics"],
    continuous_sources: bool,
    n_steps: int = 100,
    n_sources: int = 4,
    move_at_each_step: bool = True,
    duration_s: float = 6.0,
    rt_60: float = 0.5,
    seed: int = 0,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mic_array: acx.MicArray = BinauralArray(
        position=np.array([2, 2, 1.2]),
        orientation=np.array([0, 1, 0]),
    )

    room: acx.room.Room
    match backend:
        case "gpurir":
            room = acx.room.GpuRirRoom(rt_60=rt_60)
        case "pyroomacoustics":
            room = acx.room.PyRoomAcousticsRoom(rt_60=rt_60)
        case _:
            assert False

    simulator: acx.AudioSimulator = acx.AudioSimulator(
        mic_array=mic_array,
        room=room,
        n_speech_sources=n_sources,
        max_audio_samples=int(duration_s * room.sampling_frequency),
        source_continuous=continuous_sources,
    )

    simulator.reset_positions_random()

    start: float = time()
    for i in tqdm(range(n_steps)):
        simulator.step(listen_audio=False)
        # simulator.plot()
        # plt.close()

        if move_at_each_step:
            simulator.reset_positions_random()

    duration: float = time() - start
    min: int = int(duration // 60)
    sec: int = int(duration % 60)
    print(f"{min}:{sec}")


if __name__ == "__main__":
    benchmark_simulator(
        # backend="gpurir",
        backend="pyroomacoustics",
        continuous_sources=False,
    )
