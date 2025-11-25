import matplotlib.pyplot as plt
import numpy as np

from acoustix.audio_simulator import AudioSimulator
from acoustix.microphone_arrays import SquareArray
from acoustix.room import GpuRirRoom

room = GpuRirRoom(size_x=12, size_y=10, height=4, rt_60=0.8)
array = SquareArray(center_to_mic_dist=4, position=np.array([6, 5, 1.8]))

simulator = AudioSimulator(
    room=room,
    mic_array=array,
    n_speech_sources=3,
    noise_source=True,
    noise_source_type="white_noise",
    source_continuous=True,
    max_audio_samples=2 * room.sampling_frequency,
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

# Run simulation and visualize results
simulator.step()

# Get time-domain signals
audio = simulator.get_agent_audio()

# Plot microphone signals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    if i < audio.shape[0]:
        ax.plot(audio[i])
        ax.set_title(f"Microphone {i + 1}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
plt.tight_layout()
plt.show()
plt.close()

# Get and plot spectrograms
stft = simulator.get_agent_stft()
