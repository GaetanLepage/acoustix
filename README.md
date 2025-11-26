# Acoustix

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Acoustix** is a comprehensive Python library for dynamic acoustic simulation designed specifically for robotics research.
It enables realistic simulation of reverberant acoustic environments with multiple sound sources and microphone arrays, making it ideal for developing and testing sound-driven navigation, source localization, and audio-based robotic perception systems.

This project was developed as part of my PhD project, realized in the [RobotLearn team](https://team.inria.fr/robotlearn), at Inria Grenoble, under the supervision of [Dr. Xavier Alameda](https://xavirema.eu), [Pr. Laurent Girin](https://www.gipsa-lab.grenoble-inp.fr/user/laurent.girin) and [Dr. Chris Reinke](https://www.chris-reinke.com/).
You can learn more about this library, its motivations, its applications and the relevant scientific and technical decisions in my [PhD manuscript](https://theses.fr/s253609):
- Chapter 2: Introduction of acoustics in reverberant environments, and presentation of the _Acoustics_ library
- Chapter 3: Deep-learning-based sound source localization
- Chapter 4: Active sound source localization
- Chapter 5: Deep reinforcement learning for sound-driven navigation

The code for all my experiments can be found [here](https://gitlab.inria.fr/robotlearn/rl-audio-nav).

## ✨ Key Features

- **🏠 Realistic Room Acoustics**: Simulate reverberant environments with customizable room dimensions and acoustic properties (RT60, absorption coefficients)
- **🎤 Multiple Microphone Arrays**: Support for various array geometries including binaural, linear, square, and triangular configurations
- **🔊 Diverse Sound Sources**: Speech sources (LibriSpeech integration), white noise, and custom audio sources with spatial positioning
- **🚀 High-Performance Backends**: Leverages both [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) and [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) for fast Room Impulse Response (RIR) generation
- **🧠 Spatial Audio Processing**: Built-in STFT computation, DOA (Direction of Arrival) estimation, and ILD/IPD analysis
- **🗺️ Egocentric Audio Maps**: Generate spatial representations of the acoustic environment from the agent's perspective
- **🎮 Dynamic Simulation**: Real-time agent movement and source repositioning during simulation
- **📊 Rich Visualization**: Integrated plotting capabilities for room geometry, source positions, and audio signals

## Quick Start

### Installation

```bash
pip install acoustix
```

For speech sources, download the LibriSpeech dataset (optional):
```bash
# Install dependencies
sudo apt install tar curl parallel ffmpeg

# Download LibriSpeech train-clean-100 subset
./acoustix/datasets/download_librispeech.sh
```

### Basic Usage

WARNING! The origin of the coordinate system is always in the top-left!

```python
import numpy as np
from acoustix import GpuRirRoom, AudioSimulator
from acoustix.microphone_arrays import BinauralArray

# Create a reverberant room
room = GpuRirRoom(
    size_x=8.0, # Room dimensions in meters
    size_y=6.0,
    height=3.0,
    rt_60=0.5,              # Reverberation time in seconds
    sampling_freq=16_000,   # Sampling frequency
)

# Set up a binaural microphone array (robot's "ears")
array = BinauralArray(
    mic_dist=10,                        # Distance between microphones in cm
    position=np.array([3.5, 2.0, 1.2]), # Agent position (x, y, z)
    orientation=np.array([0, 1, 0]),    # Agent orientation
    mic_pattern="card",                 # Microphone pattern
)

# Initialize the simulator with multiple speech sources
simulator = AudioSimulator(
    room=room,
    mic_array=array,
    n_speech_sources=2,                             # Number of speech sources
    max_audio_samples=4 * room.sampling_frequency,  # 4 seconds of audio
)

# Run simulation
simulator.step()

# Get the multi-channel audio signal
audio = simulator.get_agent_audio()  # Shape: (n_mics, n_samples)

# Get spectral representation
stft = simulator.get_agent_stft()    # Shape: (n_mics, n_freq, n_frames)

# Extract spatial information
doa = simulator.get_doa(source_name="speech_1")                     # Direction of arrival
distance = simulator.get_source_array_dist(source_name="speech_1")  # Source-array distance
```

## Core Components

### AudioSimulator

The main interface that orchestrates room simulation, source management, and audio processing:

```python
simulator = AudioSimulator(
    room=room,
    mic_array=array,
    n_speech_sources=3,
    source_continuous=True,      # Continuous speech streams
    max_audio_samples=160_000,   # 10 seconds at 16kHz
)

# Dynamic agent movement
simulator.move_agent(
    new_position=np.array([5.0, 3.0, 1.2]),
    new_orientation=np.array([1, 0, 0]),
)

# Step simulation
simulator.step()
```

### Room Models

Choose between two backends:

```python
# GPU-accelerated RIR generation (recommended)
from acoustix import GpuRirRoom
room = GpuRirRoom(size_x=10, size_y=8, height=3], rt_60=0.6)

# CPU-based alternative
from acoustix import PyRoomAcousticsRoom
room = PyRoomAcousticsRoom(size_x=10, size_y=8, height=3], rt_60=0.6)
```

### Microphone Arrays

Multiple array geometries for different robotic platforms:

```python
from acoustix.microphone_arrays import (
    MonoArray              # Single microphone
    BinauralArray,         # 2 microphones (human-like hearing)
    UniformLinearArray,    # Linear array with N microphones
    SquareArray,           # 2x2 square configuration
    TriangleArray,         # 3-microphone triangular setup
)

# Linear array with 4 microphones
linear_array = UniformLinearArray(
    n_mics=4,
    mic_spacing=5,  # 5cm spacing
    position=np.array([2, 2, 1.5]),
)
```

### Sound Sources

Various source types for different scenarios:

```python
from acoustix.room import SpeechSource, WhiteNoiseSource, MusicNoiseSource

# Speech source (uses LibriSpeech)
speech = SpeechSource(
    name="speech_1",
    position=np.array([6, 4, 1.6])
)

# White noise source
noise = WhiteNoiseSource(
    name="ambient_noise",
    position=np.array([1, 1, 2.5]),
    num_samples=160_000,
)
```

### Egocentric Audio Maps

Generate spatial representations from the agent's perspective:

```python
from acoustix.egocentric_map import EgocentricMap, PolarRelativePosition

em: EgocentricMap = EgocentricMap(
    size=6,
    size_pixel=128,
    doa_res=360,
)

doas: list[float] = [
    -np.pi / 2,
    np.pi / 4,
    np.pi / 5,
]
encoded_doas: Tensor = encode_sources(sources_doas=doas)
em.apply_doa(doas=encoded_doas.numpy())
em.sources_positions = [
    PolarRelativePosition(
        dist=0.4 * em.size,
        angle=angle,
    )
    for angle in doas
]
em.plot()
em.move(
    angle=0.1,
    dist=0.5,
)
em.plot()
```

## Applications

Acoustix is particularly well-suited for:

- **🤖 Sound-Driven Navigation**: Training robots to navigate using audio cues
- **🎯 Sound Source Localization**: Developing DOA estimation algorithms
- **🔊 Audio Scene Analysis**: Understanding complex acoustic environments
- **🧠 Machine Learning**: Generating training data for deep learning models
- **📡 Multi-modal Robotics**: Integrating audio with other sensor modalities

## Advanced Examples

### Multi-Source Scenario with Noise

```python
# Complex acoustic scene
room = GpuRirRoom(size_x=12, size_y=10, height=4, rt_60=0.8)
array = SquareArray(center_to_mic_dist=4, position=np.array([6, 5, 1.8]))

simulator = AudioSimulator(
    room=room,
    mic_array=array,
    n_speech_sources=3,
    noise_source=True,
    noise_source_type="white_noise",
    source_continuous=True
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
```

### Real-time Audio Processing

```python
import matplotlib.pyplot as plt

# Run simulation and visualize results
simulator.step()

# Get time-domain signals
audio = simulator.get_agent_audio()

# Plot microphone signals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    if i < audio.shape[0]:
        ax.plot(audio[i])
        ax.set_title(f'Microphone {i+1}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Get and plot spectrograms
stft = simulator.get_agent_stft()
# ... visualization code
```

## Testing

Run the test suite to verify your installation:

```bash
uv run pytest
```

## Citation

If you use Acoustix in your research, please cite:

```bibtex
@phdthesis{acoustix_phd,
  title={From Sound to Action: Deep Learning for Audio-Based Localization and Navigation in Robotics},
  author={Lepage, Gaétan},
  school={Université Grenoble Alpes},
  year={2025},
  url={https://theses.fr/s253609}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was funded by the [SPRING](https://spring-h2020.eu/) European project.
- This simulator is built upon [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) and [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) RIR generation libraries
