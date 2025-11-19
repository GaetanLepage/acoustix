import random as rd
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ...audio_signal import AudioSignalObject, SpeechSignal
from ...datasets.cbf import get_cf_dataset
from .audio_object import AudioObject

DEFAULT_PATTERN: str = "omni"


class Source(AudioObject):
    """
    An audio source.

    Attributes:
        name (str):             A name for this source to easily identify it.
        position (np.ndarray):  Location array (x, y, z) coordinates.
                                    shape: (3,)
        signal (np.ndarray):    The audio signal associated to this source.
        delay (float):          A time delay until the source signal starts in the simulation.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        pattern: str = DEFAULT_PATTERN,
        orientation: Optional[np.ndarray] = None,
        n_time_samples: int = -1,
    ) -> None:
        """
        Init method.

        Args:
            name (str):             A name for this source to easily identify it.
            location (np.ndarray):  Location array (x, y, z) coordinates.
        """
        super().__init__(
            name=name,
            position=position,
            pattern=pattern,
            orientation=orientation,
            default_plot_color="green",
        )

        self.signal: np.ndarray
        self.delay: float = 0

        self.num_samples: int = n_time_samples

    def load_signal(
        self,
        signal: Optional[np.ndarray] = None,
        *args,
    ) -> None:
        assert signal is not None
        self.signal = signal


class NoiseSource(Source, ABC):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        n_time_samples: int = -1,
        pattern: str = DEFAULT_PATTERN,
        orientation: Optional[np.ndarray] = None,
        target_snr_db: float = -1,
    ) -> None:
        self.target_snr_db: float = target_snr_db

        super().__init__(
            name=name,
            position=position,
            pattern=pattern,
            orientation=orientation,
            n_time_samples=n_time_samples,
        )

    @abstractmethod
    def _load_raw_noise_signal(
        self,
        num_samples: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def load_signal(  # type: ignore
        self,
        num_samples: int = -1,
        speech_signal: Optional[np.ndarray] = None,
        target_snr_db: float = -1,
    ) -> None:
        if target_snr_db < 0:
            target_snr_db = self.target_snr_db

        if (num_samples < 0) and (self.num_samples > 0):
            num_samples = self.num_samples

        elif speech_signal is not None:
            num_samples = len(speech_signal)

        noise_signal: np.ndarray = self._load_raw_noise_signal(
            num_samples=num_samples,
        )

        if (speech_signal is not None) and (target_snr_db > 0):
            noise_power: float = np.mean(noise_signal**2)
            noise_power_db: float = 10 * np.log10(noise_power)

            signal_power: float = np.mean(speech_signal**2)
            signal_power_db: float = 10 * np.log10(signal_power)

            raw_snr_db: float = signal_power_db - noise_power_db

            coeff: float = 10 ** ((raw_snr_db - target_snr_db) / 20)

            noise_signal = coeff * noise_signal

        self.signal = noise_signal


class WhiteNoiseSource(NoiseSource):
    def _load_raw_noise_signal(
        self,
        num_samples: int,
    ) -> np.ndarray:
        assert num_samples > 0, "WhiteNoiseSource requires a positive `num_samples`"
        noise_signal = np.random.normal(
            loc=0,
            scale=1,
            size=(num_samples,),
        )

        return noise_signal


class MusicNoiseSource(NoiseSource):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        pattern: str = DEFAULT_PATTERN,
        orientation: Optional[np.ndarray] = None,
        n_time_samples: int = -1,
        target_snr_db: float = -1,
        n_dataset_samples: int = -1,
        dataset_path: str = "",
        dataset_seed: int = -1,
    ) -> None:
        super().__init__(
            name=name,
            position=position,
            pattern=pattern,
            orientation=orientation,
            n_time_samples=n_time_samples,
            target_snr_db=target_snr_db,
        )

        self.noise_dataset: list[AudioSignalObject] = get_cf_dataset(
            random=True,
            n_samples=n_dataset_samples,
            dataset_path=dataset_path,
            seed=dataset_seed,
            load_audio_array=True,
        )

    def _load_raw_noise_signal(
        self,
        num_samples: int,
    ) -> np.ndarray:
        noise_signal: AudioSignalObject = rd.choice(self.noise_dataset)

        noise_signal.load_audio(
            # Do not keep the whole audio recording but only `num_samples` frames.
            n_samples=num_samples,
            # Select those `num_samples` frames randomly
            random_start_index=True,
        )
        noise_signal_array: np.ndarray = noise_signal.normalized_signal.squeeze()

        return noise_signal_array


class SpeechSource(Source):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        n_dataset_samples: int = -1,
        n_time_samples: int = -1,
        dataset_path: str = "",
        dataset_seed: int = -1,
        min_duration: float = -1.0,
        pattern: str = DEFAULT_PATTERN,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        from ...datasets.librispeech import get_speech_data_set

        super().__init__(
            name=name,
            position=position,
            pattern=pattern,
            orientation=orientation,
            n_time_samples=n_time_samples,
        )

        self.speech_dataset: list[SpeechSignal] = get_speech_data_set(
            random=True,
            n_samples=n_dataset_samples,
            dataset_path=dataset_path,
            seed=dataset_seed,
            min_duration=min_duration,
        )
        self._n_dataset_samples: int = len(self.speech_dataset)

    def load_signal(self, *args) -> None:
        speech_signal: SpeechSignal = rd.choice(self.speech_dataset)

        speech_signal.load_audio(
            # Do not keep the whole audio recording but only `num_samples` frames.
            n_samples=self.num_samples,
            # Select those `num_samples` frames randomly
            random_start_index=True,
        )
        self.signal = speech_signal.normalized_signal.copy()
        speech_signal.unload_audio()


class SpeechSourceContinuous(SpeechSource):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        n_dataset_samples: int = -1,
        n_time_samples: int = -1,
        dataset_path: str = "",
        dataset_seed: int = -1,
        pattern: str = DEFAULT_PATTERN,
        orientation: Optional[np.ndarray] = None,
        n_overlap_samples: int = 0,
    ) -> None:
        super().__init__(
            name=name,
            position=position,
            pattern=pattern,
            orientation=orientation,
            n_dataset_samples=n_dataset_samples,
            n_time_samples=n_time_samples,
            dataset_path=dataset_path,
            dataset_seed=dataset_seed,
        )

        assert self.num_samples > 0, (
            "SpeechSourceContinuous does not work without a fix n_time_samples"
        )

        self.n_overlap_samples: int = n_overlap_samples

        self._current_sample_index: int = 0
        self._current_sample: SpeechSignal = self.speech_dataset[self._current_sample_index]
        # Load the full recording
        self._current_sample.load_audio()
        self._time_frame: int = 0

    def _load_next_signal(self) -> None:
        start: int = self._time_frame
        end: int = start + self.num_samples

        assert end - start == self.num_samples
        assert len(self._current_sample.normalized_signal[start:]) >= self.num_samples
        self.signal = self._current_sample.normalized_signal[start:end].copy()
        assert self.signal.shape == (self.num_samples,), (
            f"{self.signal.shape} != ({self.num_samples},)"
        )

        self._time_frame += self.num_samples - self.n_overlap_samples

        # If no more chunk to extract from this audio sample, set up next one
        next_end: int = self._time_frame + self.num_samples
        if next_end > self._current_sample.n_samples:
            self._current_sample.unload_audio()
            self._current_sample_index += 1
            self._current_sample_index %= self._n_dataset_samples

            while True:
                self._current_sample = self.speech_dataset[self._current_sample_index]
                self._current_sample.load_audio()

                if len(self._current_sample.signal) > self.num_samples:
                    break
                self._current_sample_index += 1
                self._current_sample_index %= self._n_dataset_samples
                self._current_sample.unload_audio()

            self._time_frame = 0

    def load_signal(self, *args) -> None:
        self._load_next_signal()
        while np.abs(self.signal).max() < 1e-3:
            self._load_next_signal()
