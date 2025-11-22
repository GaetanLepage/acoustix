import numpy as np

from acoustix.audio_simulator import AudioSimulator
from acoustix.microphone_arrays import BinauralArray
from acoustix.room import GpuRirRoom, Room
from acoustix.stft import plot_stft


def test_binaural() -> None:
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

    #########
    # ARRAY #
    #########
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
