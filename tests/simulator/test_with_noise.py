import numpy as np

from acoustix.audio_simulator import AudioSimulator
from acoustix.microphone_arrays import SquareArray
from acoustix.room import GpuRirRoom


def test_with_noise() -> None:
    room = GpuRirRoom(
        size_x=12,
        size_y=10,
        height=3,
        rt_60=0.8,
    )
    array = SquareArray(
        center_to_mic_dist=4,
        position=np.array([6, 5, 1.8]),
    )

    simulator = AudioSimulator(
        room=room,
        mic_array=array,
        n_speech_sources=3,
        noise_source=True,
        noise_source_type="white_noise",
        max_audio_samples=4 * room.sampling_frequency,
    )

    # Simulate agent movement through the environment
    positions = [
        np.array([2, 2, 1.8]),
        np.array([6, 5, 1.8]),
        np.array([10, 8, 1.8]),
    ]

    for pos in positions:
        simulator.move_agent(new_position=pos)
        simulator.step()
        audio = simulator.get_agent_audio()
