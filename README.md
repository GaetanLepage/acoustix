# Acoustix

**Acoustix** is an audio simulation library targeting research in acoustics for robotics.
It allows to simulate a reverberant acoustic environment where multiple sources and microphones can be modeled.
The agent (a microphone array) is placed in a room with specific acoustic properties.
Multiple speech sound sources can then be added to model speaking humans.
**Acoustix** will simulate the multi-channel audio listened by the microphone array.

This project was developed as part of my PhD project, realized in the [RobotLearn team](https://team.inria.fr/robotlearn), at Inria Grenoble, under the supervision of [Dr. Xavier Alameda](https://xavirema.eu), [Pr. Laurent Girin](https://www.gipsa-lab.grenoble-inp.fr/user/laurent.girin) and [Dr. Chris Reinke](https://www.chris-reinke.com/).
You can learn more about this library, its motivations, its applications and the relevant scientific and technical decisions in my [PhD manuscript](https://theses.fr/s253609):
- Chapter 2: Introduction of acoustics in reverberant environments, and presentation of the _Acoustics_ library
- Chapter 3: Deep-learning-based sound source localization
- Chapter 4: Active sound source localization
- Chapter 5: Deep reinforcement learning for sound-driven navigation

The code for all my experiments can be found [here](https://gitlab.inria.fr/robotlearn/rl-audio-nav).


## Install
```bash
pip install acoustix
```

By default, speech sources' input signals are drawn from the [LibriSpeech](https://www.openslr.org/12) corpus.
Acoustix will attempt to load clean speech samples from `./data/LibriSpeech/train-clean-100`.
To download the dataset, please proceed as follows:
```bash
# Get the required dependencies
sudo apt install tar curl parallel ffmpeg

# Run the script
./acoustix/datasets/download_librispeech.sh
```

## Getting started

```python
from acoustix import GpuRirRoom, AudioSimulator
from acoustix.microphone_arrays import BinauralArray

room = GpuRirRoom(
    sampling_freq=16_000,
    rt_60=0.5,
)
array = BinauralArray(
    mic_dist=10, # cm
    position = np.ndarray([3.5, 2.0, 1.0]), # (x, y, z)
    mic_pattern="card",
)
simulator = AudioSimulator(
    room=room,
    mic_array=array,
    n_speech_sources=2,
    max_audio_samples=4 * room.sampling_frequency, # limit simulation to 4s
)

# Run one simulation step (4s)
simulator.step()

# Get the multi-channel listened signal
listened_audio = simulator.get_agent_audio()

# Or directly its spectral representation
stft = simulator.get_agent_stft()

# Get spatial information
doa = simulator.get_doa(source_name="speech_2")
source_array_dist = simulator.get_source_array_dist(source_name="speech_2")
```

## Documentation

TODO

### `AudioSimulator`

The `AudioSimulator` class is the highest level of abstraction and manages the agent (the microphone array),
the room and the sound sources.
The two most important arguments of the constructor are:
- `room`: A `Room` instance, already initialized.
- `mic_array`: A `MicrophoneArray` instance modelling the agent.

You can also provide a number of speech sources (`n_speech_sources`, defaults to `1`) to add to the environment.

**Core features:**
- Agent movements: `AudioSimulator` provides several
- `move_agent`

### `Room`

TODO

### Sound sources

TODO

### Microphone arrays

TODO

- `MonoArray`
- `BinauralArray`
- `TriangleArray`
- `SquareArray`
- `UniformLinearArray`


## Acknowledgments

- This work was funded by the [SPRING](https://spring-h2020.eu/) European project.
- This simulator leverages the [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) and [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) RIR generation libraries
