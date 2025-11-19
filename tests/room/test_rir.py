#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from acoustix.room import Source
from acoustix.room.gpu_rir_room import GpuRirRoom

N_REPETITIONS: int = 2


def test_rir() -> None:
    room: GpuRirRoom = GpuRirRoom()
    room.init_grid(add_grid_mics=True)

    source: Source = Source(
        name="source",
        position=np.array(
            [6, 3, 1.5],
        ),
    )
    room.add_source(
        source=source,
        clip_to_grid=True,
    )

    rir_tensor: np.ndarray = np.zeros(
        shape=(
            N_REPETITIONS,
            112,
            4800,
        ),
    )

    for rep_index in range(N_REPETITIONS):
        room.rir_up_to_date = False

        room.pre_compute_rir()
        rir_tensor[rep_index] = room.rir.squeeze()

    rir_mic_50: np.ndarray = rir_tensor[:, 50, :].squeeze()
    print(rir_mic_50.shape)

    fig, axes = matplotlib.pyplot.subplots()
    # print('min:', rir_tensor.min())
    # print('max:', rir_tensor.max())
    # print('max:', rir_tensor.mean())
    diff = rir_mic_50[0] - rir_mic_50[1]
    # axes.pcolormesh(rir_mic_50)
    axes.plot(rir_mic_50[0])
    axes.plot(diff)

    mean_over_reps: np.ndarray = rir_tensor.mean(axis=0)
    mean_over_reps /= mean_over_reps.mean()
    print("mean =", mean_over_reps.mean(axis=0))
    axes.pcolormesh(mean_over_reps)
    print(mean_over_reps.shape)

    plt.show()
