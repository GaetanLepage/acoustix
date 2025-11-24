import numpy as np

from acoustix.audio_simulator import AudioSimulator
from acoustix.microphone_arrays.square_array import SquareArray
from acoustix.room import GpuRirRoom


def test_doa() -> None:
    audio_simulator: AudioSimulator = AudioSimulator(
        mic_array=SquareArray(
            position=np.array([3.0, 3.0, 1.0]),
            orientation=np.array([-1.0, 1.0, 0.0]),
        ),
        room=GpuRirRoom(
            size_x=7,
            size_y=6,
        ),
        n_speech_sources=2,
    )
    audio_simulator.move_source(
        name="speech_0",
        new_position=np.array([4.0, 4.0, 1.0]),
    )
    audio_simulator.move_source(
        name="speech_1",
        new_position=np.array([5.0, 1.0, 1.0]),
    )

    assert audio_simulator.get_doa(source_name="speech_0") == -np.pi / 2
