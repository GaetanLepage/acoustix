import logging
from typing import Any, Optional

import exputils as eu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from .microphone_arrays import MicArray, compute_ild_ipd_from_stft
from .room import IllegalPosition, Room
from .room.audio_objects.source import (
    NoiseSource,
    Source,
    SpeechSource,
    SpeechSourceContinuous,
    WhiteNoiseSource,
)
from .stft import StftModule
from .utils import (
    compute_dist_and_doa,
    get_min_doa_dist,
    play_audio,
    random_orientation,
    rotate_3d_vector,
)


class AudioSimulator:
    """
    The AudioSimulator provides an interface to the RL environments.
    It encompasses both the room, the sources and the "acoustic agent".
    """

    default_config: eu.AttrDict = eu.AttrDict(
        stft=eu.AttrDict(),
    )
    _noise_source_name: str = "noise_source"

    def __init__(
        self,
        room: Room,
        mic_array: MicArray,
        ##################################
        ## Sources
        source_height: float = 1.2,
        # Speech
        n_speech_sources: int = 1,
        source_continuous: bool = False,
        speech_source_base_seed: int = 0,
        muted_source: bool = False,
        # Noise
        noise_source: bool = False,
        noise_source_type: str = "",
        ##################################
        # Audio
        stft: StftModule | None = None,
        max_audio_samples: int = -1,
        min_stft_frames: int = -1,
        audio_upsampling_freq: int = -1,
        n_mock_samples: int = 0,
    ) -> None:
        """
        Initialize an AudioSimulator instance.

        Args:
            room: Room object where the simulation takes place
            mic_array: Microphone array representing the acoustic agent
            source_height: Height at which sources are placed in the room
            n_speech_sources: Number of speech sources to add to the simulation
            source_continuous: Whether to use continuous speech sources
            speech_source_base_seed: Base seed for speech source randomization
            muted_source: Whether to create muted sources (no audio output)
            noise_source: Whether to add a noise source to the simulation
            noise_source_type: Type of noise source (e.g., "white_noise")
            stft: STFT module for audio processing
            max_audio_samples: Maximum number of audio samples to simulate (-1 for unlimited)
            min_stft_frames: Minimum number of STFT frames required
            audio_upsampling_freq: Frequency for audio upsampling (-1 to disable)
            n_mock_samples: Number of mock samples to simulate and discard
        """
        self._logger: logging.Logger = logging.getLogger(__name__)

        self.room: Room = room

        # MICROPHONE ARRAY
        self.mic_array: MicArray = mic_array

        for mic in self.mic_array.microphones:
            self.room.assert_is_in_room(position=mic.position)
            self.room.add_microphone(mic=mic)

        # SIMULATION DURATION
        self.n_mock_samples: int = n_mock_samples
        if self.n_mock_samples > 0:
            self._logger.info(
                "%i extra samples will be simulated and discarded before each simulation (~%.2fs)",
                self.n_mock_samples,
                np.around(
                    self.n_mock_samples / self.room.sampling_frequency,
                    decimals=2,
                ),
            )
        self._n_sim_samples: int = -1
        if max_audio_samples > 0:
            self._logger.info(
                "Simulation input will have a limited duration of %i time samples (~%.2fs)",
                max_audio_samples,
                np.around(
                    max_audio_samples / self.room.sampling_frequency,
                    decimals=2,
                ),
            )
            self._n_sim_samples = n_mock_samples + max_audio_samples
            self._logger.info(
                "Total simulation time: %i samples (~%.2fs)",
                self._n_sim_samples,
                np.around(
                    self._n_sim_samples / self.room.sampling_frequency,
                    decimals=2,
                ),
            )
        self.max_audio_samples: int = max_audio_samples

        # FREQUENCY
        self._logger.info("Room/simulation frequency: %iHz", self.room.sampling_frequency)
        self.audio_upsampling_freq: int = audio_upsampling_freq
        if self.audio_upsampling_freq > 0:
            self._logger.warn("Audio will be upsampled to: %iHz", self.audio_upsampling_freq)
        else:
            self._logger.info("UPSAMPLING DISABLED")
        # The frequency after (eventual upsampling)
        self._audio_final_freq: int = (
            self.audio_upsampling_freq
            if self.audio_upsampling_freq > 0
            else int(self.room.sampling_frequency)
        )

        # STFT
        if stft is None:
            stft = StftModule(
                log_stft=False,
                freq=self._audio_final_freq,
            )
        self.stft: StftModule = stft
        assert self.stft.freq == self._audio_final_freq, (
            f"{self.stft.freq} != {self._audio_final_freq}"
        )

        # SOURCES
        self.source_height: float = source_height
        self.sources: dict[str, Source] = {}
        match n_speech_sources:
            case _ if n_speech_sources < 0:
                self._logger.error("Number of speech source has to be positive")
                raise ValueError
            case 0:
                self._logger.warning("No speech source will be added to the simulator.")
            case _:
                min_duration: float = -1.0
                if min_stft_frames > 0:
                    # WARNING this is a number of samples related to final/upsampling frequency
                    min_audio_samples: int = (self.stft.window_length // 2) * (min_stft_frames + 1)
                    # so to get the duration, we have to use the final frequency
                    min_duration = min_audio_samples / self._audio_final_freq

                for speech_source_index in range(n_speech_sources):
                    name: str = f"speech_{speech_source_index}"
                    if source_continuous:
                        self.sources[name] = SpeechSourceContinuous(
                            name=name,
                            position=self.room.get_random_position(
                                height=self.source_height,
                            ),
                            n_time_samples=self._n_sim_samples,
                            n_overlap_samples=self.n_mock_samples,
                            dataset_seed=speech_source_base_seed + speech_source_index,
                        )
                    elif muted_source:
                        self.sources[name] = Source(
                            name=name,
                            position=self.room.get_random_position(
                                height=self.source_height,
                            ),
                        )
                    else:
                        self.sources[name] = SpeechSource(
                            name=name,
                            position=self.room.get_random_position(
                                height=self.source_height,
                            ),
                            n_time_samples=self._n_sim_samples,
                            min_duration=min_duration,
                            dataset_seed=speech_source_base_seed + speech_source_index,
                        )
                    self.room.add_source(
                        source=self.sources[name],
                    )
        self.n_speech_sources: int = n_speech_sources

        # Noise source
        self.noise_enabled: bool = noise_source
        _noise_source: NoiseSource
        if self.noise_enabled:
            self.add_noise_source(noise_source_type=noise_source_type)

    @property
    def n_sources(self) -> int:
        """
        Get the total number of sources in the simulation.

        Returns:
            Total number of sources (speech + noise)
        """
        return len(self.sources)

    @property
    def speech_sources(self) -> dict[str, SpeechSource]:
        speech_sources: dict[str, SpeechSource] = {
            source_name: source
            for source_name, source in self.sources.items()
            if isinstance(source, SpeechSource)
        }

        assert len(speech_sources) == self.n_speech_sources

        return speech_sources

    def add_noise_source(self, noise_source_type: str) -> None:
        self._logger.info("Adding a `%s` noise source", noise_source_type)
        match noise_source_type:
            case "white_noise":
                _noise_source = WhiteNoiseSource(
                    name="noise_source",
                    position=self.room.get_random_position(
                        height=self.source_height,
                    ),
                    n_time_samples=self._n_sim_samples,
                )
            case _:
                assert False, f"Invalid noise source type `{noise_source_type}`"
        self.sources[self._noise_source_name] = _noise_source
        self.room.add_source(
            source=_noise_source,
        )
        self.noise_enabled = True

    def remove_noise_source(self) -> None:
        assert self.noise_enabled
        self._logger.info("Removing the noise source")

        self.sources.pop(self._noise_source_name)
        self.room.remove_source(source_name=self._noise_source_name)

        self.noise_enabled = False

    @property
    def _single_speech_source_name(self) -> str:
        assert self.n_speech_sources == 1
        return list(self.sources)[0]

    @property
    def noise_source(self) -> NoiseSource:
        assert self.noise_enabled
        noise_source: Source = self.sources[self._noise_source_name]
        assert isinstance(noise_source, NoiseSource | bool)

        return noise_source

    @property
    def agent_position(self) -> np.ndarray:
        return self.mic_array.position

    @property
    def agent_orientation(self) -> np.ndarray:
        return self.mic_array.orientation

    ## POSITION ##

    def move_agent(
        self,
        new_position: Optional[np.ndarray] = None,
        new_orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Move the acoustic agent to a new position and/or orientation.

        Args:
            new_position: New 3D position for the agent (None to keep current)
            new_orientation: New 3D orientation for the agent (None to keep current)

        Raises:
            IllegalPosition: If the new position is outside the room
        """
        if new_position is None:
            new_position = self.agent_position

        if new_orientation is None:
            new_orientation = self.agent_orientation

        assert new_position.shape == (3,)
        assert new_orientation.shape == (3,)

        if not self.room.is_circle_in_room(
            center_position=new_position,
            radius=self.mic_array.radius,
        ):
            raise IllegalPosition("Agent's position is invalid")

        # Get new microphones positions
        new_mics_positions: np.ndarray
        new_mics_orientations: np.ndarray

        new_mics_positions, new_mics_orientations = self.mic_array.get_mic_coordinates(
            position=new_position,
            orientation=new_orientation,
        )

        # Move microphones in the room
        for mic, mic_position, mic_orientation in zip(
            self.mic_array.microphones,
            new_mics_positions,
            new_mics_orientations,
        ):
            self.room.move_microphone(
                mic_name=mic.name,
                new_position=mic_position,
                new_orientation=mic_orientation,
            )

        # Update the agent location
        self.mic_array.position = new_position
        self.mic_array.set_orientation(orientation=new_orientation)

    def move_agent_polar(
        self,
        distance: float,
        angle_rad: float = 0.0,
    ) -> None:
        curr_position: np.ndarray = self.mic_array.position
        curr_ori: np.ndarray = self.mic_array.orientation

        # rotate
        new_ori = rotate_3d_vector(
            vec=curr_ori,
            angle_xy=angle_rad,
        )

        # translate in new direction
        new_position = np.array(
            [
                curr_position[0] + distance * new_ori[0],
                curr_position[1] + distance * new_ori[1],
                curr_position[2],  # height stays the same
            ]
        )

        self.move_agent(
            new_position=new_position,
            new_orientation=new_ori,
        )

    def move_agent_forward(
        self,
        distance: float,
    ) -> None:
        self.move_agent_polar(
            distance=distance,
            angle_rad=0.0,
        )

    def rotate_agent_left(self) -> None:
        self.move_agent_polar(
            distance=0.0,
            angle_rad=-np.pi / 2,
        )

    def rotate_agent_right(self) -> None:
        self.move_agent_polar(
            distance=0.0,
            angle_rad=np.pi / 2,
        )

    def move_agent_random(self) -> None:
        new_position: np.ndarray = self.room.get_random_position(
            height=self.mic_array.height,
            padding=self.mic_array.radius,
        )
        new_orientation: np.ndarray = random_orientation()

        self.move_agent(
            new_position=new_position,
            new_orientation=new_orientation,
        )

    def move_source(
        self,
        name: str = "",
        new_position: Optional[np.ndarray] = None,
        random: bool = False,
    ) -> None:
        if not name:
            assert self.n_sources == 1
            name = list(self.sources)[0]

        if random:
            assert new_position is None
            new_position = self.room.get_random_position(
                height=self.source_height,
            )
        assert name in self.sources
        self.room.move_source(
            source_name=name,
            new_position=new_position,
        )

    def move_speech_source(
        self,
        new_position: Optional[np.ndarray] = None,
        random: bool = False,
    ) -> None:
        assert self.n_speech_sources == 1
        self.move_source(
            name=list(self.speech_sources)[0],
            new_position=new_position,
            random=random,
        )

    def move_noise_source(
        self,
        new_position: Optional[np.ndarray] = None,
        random: bool = False,
    ) -> None:
        assert self.noise_enabled
        self.move_source(
            name=self.noise_source.name,
            new_position=new_position,
            random=random,
        )

    def reset_positions_random(self) -> None:
        """
        Resets the position of the agent and the sources randomly.
        """
        self.move_agent_random()
        for source_name in self.sources:
            self.move_source(
                name=source_name,
                random=True,
            )

    @property
    def min_doa_dist(self) -> float:
        doas: list[float] = [
            self.get_doa(source_name=source_name) for source_name in self.speech_sources
        ]

        return get_min_doa_dist(
            doas=doas,
        )

    ## SIMULATION ##
    def enable_all_sources(self) -> None:
        for source in self.sources.values():
            source.is_active = True

    def get_state_dict(
        self,
        enabled_sources: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        state_dict: dict[str, Any] = {
            "agent": {
                "position": (
                    # Converting to float because int64 is not JSON serializable
                    float(self.agent_position[0]),
                    float(self.agent_position[1]),
                ),
                "height": float(self.agent_position[2]),
                "orientation": (
                    float(self.agent_orientation[0]),
                    float(self.agent_orientation[1]),
                ),
            },
        }
        state_dict["sources"] = {}
        # Warning: n_sources is the total number of sources in the simulator, not necessarily the number
        # of **speech** sources (i.e. invalid when there is some noise source)
        state_dict["n_sources"] = self.n_sources
        state_dict["n_speech_sources"] = self.n_speech_sources
        state_dict["n_active_sources"] = (
            self.n_speech_sources if enabled_sources is None else len(enabled_sources)
        )

        for source_name, source in self.speech_sources.items():
            if enabled_sources is not None and (source_name not in enabled_sources):
                continue
            state_dict["sources"][source_name] = {
                "position": (
                    float(source.position[0]),
                    float(source.position[1]),
                ),
                "height": float(source.position[2]),
                "distance": self.get_source_array_dist(source_name=source_name),
                "doa": self.get_doa(source_name=source_name),
            }
        return state_dict

    def get_agent_audio(
        self,
        max_audio_samples: int = -1,
    ) -> np.ndarray:
        if max_audio_samples <= 0:
            max_audio_samples = self.max_audio_samples

        mic_signals: list[np.ndarray] = []
        for mic in self.mic_array.microphones:
            mic_signals.append(
                self.room.get_audio_at_mic(
                    mic_name=mic.name,
                ),
            )

        assert all(
            [len(mic_signal) == len(mic_signals[0]) for mic_signal in mic_signals],
        )

        mic_signals_array: np.ndarray = np.array(mic_signals)

        # Truncate the mock samples
        if self.n_mock_samples > 0:
            mic_signals_array = mic_signals_array[:, self.n_mock_samples :]

        # If needed, truncate the array
        if 0 < max_audio_samples < mic_signals_array.shape[1]:
            mic_signals_array = mic_signals_array[:, :max_audio_samples]

        if self.audio_upsampling_freq > 0:
            upsampling_ratio: float = self.audio_upsampling_freq / int(self.room.sampling_frequency)
            new_number_of_samples: int = int(upsampling_ratio * mic_signals_array.shape[-1])
            mic_signals_array = scipy.signal.resample(
                mic_signals_array,
                new_number_of_samples,
                axis=1,
            )[:, :-1]

        return mic_signals_array

    def _compute_dist_and_doa(
        self,
        source_name: str = "",
    ) -> tuple[float, float]:
        """
        Computes the distance and the DOA from the agent to the speech source.
        """

        if not source_name:
            source_name = self._single_speech_source_name

        return compute_dist_and_doa(
            agent_2d_pos=self.agent_position[:2],
            agent_2d_ori=self.mic_array.orientation[:2],
            source_2d_pos=self.sources[source_name].position[:2],
        )

    def get_doa(self, source_name: str = "") -> float:
        return self._compute_dist_and_doa(
            source_name=source_name,
        )[1]

    def get_source_array_dist(self, source_name: str = "") -> float:
        return self._compute_dist_and_doa(
            source_name=source_name,
        )[0]

    def get_source_position(self, source_name: str = "") -> np.ndarray:
        if not source_name:
            source_name = self._single_speech_source_name

        return self.sources[source_name].position

    def get_source_position_xy(self, source_name: str = "") -> tuple[float, float]:
        pos: np.ndarray = self.get_source_position(source_name=source_name)
        return (pos[0], pos[1])

    def step(
        self,
        return_stft: bool = True,
        listen_audio: bool = False,
        target_snr_db: int = -1,
        enabled_sources: Optional[list[str]] = None,
        plot: bool = False,
    ) -> (
        np.ndarray
        | tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ):
        disabled_sources: list[str] = []
        if enabled_sources is not None:
            for source_name, source in self.sources.items():
                if source_name not in enabled_sources:
                    disabled_sources.append(source_name)
                    source.is_active = False
        else:
            enabled_sources = [source_name for source_name in self.sources]

        # 1) Load source signal
        for source_name in enabled_sources:
            source = self.sources[source_name]
            source.load_signal()
            self.room.set_source_input_audio_signal(
                input_audio_signal=source.signal,
                source_name=source_name,
            )

        # 2) Simulate listened signal
        self.room.simulate(force=True)

        agent_audio: Optional[np.ndarray] = None

        # Optionally play audio
        if listen_audio:
            agent_audio = self.get_agent_audio()
            play_audio(
                audio_signal=agent_audio[0],
                sample_rate=self._audio_final_freq,
            )

        if plot:
            plt.close()
            self.plot()

        result: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
        # Return relevant features
        if return_stft:
            result = self.get_agent_stft()

        else:
            if agent_audio is None:
                agent_audio = self.get_agent_audio()
            result = agent_audio

        # Add back temporarily removed sources
        for source_name in disabled_sources:
            self.sources[source_name].is_active = True

        return result

    def get_agent_stft(
        self,
        max_audio_samples: int = -1,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Returns (freqs, times, STFT)
        """
        audio_array: np.ndarray = self.get_agent_audio(
            max_audio_samples=max_audio_samples,
        )
        return self.stft(audio_array)

    def get_ild_ipd(
        self,
        max_audio_samples: int = -1,
        ild: bool = True,
        ipd: bool = True,
    ) -> np.ndarray:
        return compute_ild_ipd_from_stft(
            audio_stft=self.get_agent_stft(
                max_audio_samples=max_audio_samples,
            )[-1],
            ild=ild,
            ipd=ipd,
        )

    def plot(
        self,
        show_array: bool = True,
        show: bool = True,
        show_source_labels: bool = True,
        show_grid: bool = False,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes

        fig, ax = self.room.plot(
            show_grid=show_grid,
            show_sources=False,
            show_mics=False,
        )
        fig.tight_layout()

        # Sources
        for source in self.sources.values():
            source.plot(
                ax=ax,
                show_label=show_source_labels,
            )

        # Mic array
        if show_array:
            self.mic_array.plot(ax=ax)

        if show:
            plt.show()

        return fig, ax
