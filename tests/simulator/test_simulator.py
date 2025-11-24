import random
from time import time

import numpy as np
import torch
from tqdm import tqdm

from acoustix.audio_simulator import AudioSimulator
from acoustix.microphone_arrays import BinauralArray, MicArray
from acoustix.room import GpuRirRoom, Room
from acoustix.stft import plot_stft

SEED: int = 0
N_STEPS: int = 30
DURATION: float = 6.0
N_SOURCES: int = 2


def test_sim_basic() -> None:
    room: Room = GpuRirRoom()
    binaural_array: BinauralArray = BinauralArray(
        mic_dist=4,
        position=np.array([1, 2, 1.8]),
        orientation=np.array([0, -1.0, 0]),
    )
    simulator: AudioSimulator = AudioSimulator(
        room=room,
        mic_array=binaural_array,
        n_speech_sources=1,
        max_audio_samples=4 * room.sampling_frequency,
    )

    simulator.step()

    signal: np.ndarray = simulator.get_agent_audio()
    cross: np.ndarray = np.correlate(
        signal[0],
        signal[1],
    )
    print(cross)
    freqs, times, stft = simulator.get_agent_stft()
    plot_stft(stft=stft[0], freqs=freqs, times=times, log=True)

    # Rotate the agent to the left
    simulator.rotate_agent_left()
    simulator.step()
    freqs, times, stft = simulator.get_agent_stft()
    plot_stft(stft=stft[0], freqs=freqs, times=times, log=True)

    # Add a noise source
    simulator.add_noise_source(noise_source_type="white_noise")
    simulator.step()
    freqs, times, stft = simulator.get_agent_stft()
    plot_stft(stft=stft[0], freqs=freqs, times=times, log=True)


def test_sim_loop() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    array_pos: np.ndarray = np.array([2, 2, 1.2])
    array_ori: np.ndarray = np.array([0, 1, 0])
    # mic_array: sim.MicArray = arrays.MonoArray(
    #     position=array_pos,
    # )
    mic_array: MicArray = BinauralArray(
        mic_dist=10,
        position=array_pos,
        orientation=array_ori,
        mic_pattern="card",
    )
    # mic_array: sim.MicArray = arrays.SquareArray(
    #     center_to_mic_dist=5,
    #     position=array_pos,
    #     orientation=array_ori,
    # )
    # mic_array: sim.MicArray = arrays.TriangleArray(
    #     front_mic_dist=10,
    #     lr_mic_dist=10,
    #     position=array_pos,
    #     orientation=array_ori,
    # )

    rt_60: float = 0.5
    _room: Room = GpuRirRoom(rt_60=rt_60)
    # _room: room.Room = room.PyRoomAcousticsRoom(rt_60=rt_60)

    simulator: AudioSimulator = AudioSimulator(
        mic_array=mic_array,
        room=_room,
        n_speech_sources=N_SOURCES,
        # source=source,
        # noise_source=noise_source,
        max_audio_samples=int(DURATION * _room.sampling_frequency),
        audio_upsampling_freq=48_000,
        # n_mock_samples=1 * 16_000,
        # source_continuous=True,
        # max_audio_samples=4 * 16_000,
    )

    simulator.reset_positions_random()
    simulator.plot()

    start = time()
    for i in tqdm(range(N_STEPS)):
        simulator.move_agent_random()
        simulator.step(listen_audio=False)
        # dist: float = simulator.get_source_array_dist()
        # doa: float = simulator.get_doa()
        # print(f"dist={dist} | doa={doa}")
        # plt.close()
        # simulator.plot()
    duration = time() - start
    min: int = int(duration // 60)
    sec: int = int(duration % 60)
    print(f"{min}:{sec}")
