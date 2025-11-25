"""
LibriSpeech data set loader.
"""

import logging
import os
import random as rd
from os import path
from typing import Iterator

import matplotlib.pyplot as plt
from tqdm import tqdm

from ..audio_signal import SpeechSignal

LOGGER: logging.Logger = logging.getLogger(__name__)

NUM_SAMPLES: int = 28_539


def data_set_length() -> int:
    """
    Returns:
        data_set_length (int):  The number of samples in the complete data set.
    """
    return len(get_speech_data_set(random=False))


def _data_set_loading_loop(
    load_audio_array: bool,
    load_audio_tensor: bool,
    dataset_path: str,
    min_duration: float,
    n_samples: int = -1,
) -> list[SpeechSignal]:
    """
    Internal function to load LibriSpeech dataset.

    Args:
        load_audio_array: Whether to load audio as numpy arrays
        load_audio_tensor: Whether to load audio as torch tensors
        dataset_path: Path to the LibriSpeech dataset
        min_duration: Minimum duration for audio samples
        n_samples: Maximum number of samples to load (-1 for all)

    Returns:
        List of SpeechSignal objects
    """
    LOGGER.info("Loading dataset '%s'", dataset_path)
    if n_samples < 0:
        n_samples = NUM_SAMPLES

    data_set: list[SpeechSignal] = []

    sample_global_index: int = 0

    with tqdm(total=NUM_SAMPLES) as progress_bar:
        # Loop through speakers
        for speaker_dir in os.listdir(dataset_path):
            speaker_id: str = path.basename(speaker_dir)

            speaker_dir = path.join(dataset_path, speaker_dir)

            # Loop through chapters
            for chapter_dir in os.listdir(speaker_dir):
                chapter_id: str = path.basename(chapter_dir)

                chapter_dir = path.join(speaker_dir, chapter_dir)

                with open(
                    path.join(
                        chapter_dir,
                        f"{speaker_id}-{chapter_id}.trans.txt",
                    ),
                    "r",
                ) as trans_file:
                    trans_file_lines: list[str] = trans_file.readlines()

                audio_file_names: list[str] = [
                    filename for filename in os.listdir(chapter_dir) if filename.endswith(".wav")
                ]
                audio_file_names.sort()

                # Loop through samples
                for sample_index, (trans_line, audio_filename) in enumerate(
                    zip(trans_file_lines, audio_file_names)
                ):
                    trans_line_split: list[str] = trans_line.split()

                    base_name: str = f"{speaker_id}-{chapter_id}-{sample_index:04d}"

                    assert trans_line_split[0] == base_name

                    assert path.basename(audio_filename) == base_name + ".wav"

                    audio_filename = path.join(chapter_dir, audio_filename)

                    transcript: str = " ".join(trans_line_split[1:])

                    speech_signal: SpeechSignal = SpeechSignal(
                        file_path=audio_filename,
                        transcript=transcript,
                        load_audio_array=load_audio_array,
                        load_audio_tensor=load_audio_tensor,
                    )

                    if min_duration > 0:
                        speech_signal.load_audio()
                        duration: float = speech_signal.duration
                        speech_signal.unload_audio()

                        # This recording os too short -> skip it
                        if duration < min_duration:
                            continue

                    data_set.append(speech_signal)

                    sample_global_index += 1

                    progress_bar.update()

                    if sample_global_index >= n_samples:
                        return data_set

    return data_set


def get_speech_data_set(
    random: bool = True,
    seed: int = -1,
    n_samples: int = -1,
    min_duration: float = -1.0,
    load_audio_array: bool = False,
    load_audio_tensor: bool = False,
    dataset_path: str = "",
) -> list[SpeechSignal]:
    """
    Returns `n_samples` items from the LibriSpeech data set.

    Args:
        random (bool, default=False):               If true, samples randomly from the data set.
        seed (int, default=-1):                     The seed for the random sampling process.
        n_samples (int):                            The number of samples to draw from the data set.
        min_duration (float, default=-1)            Only load sumples of a duration greater than
                                                        this value.
        load_audio_array (bool, default=False):     If true, loads the audio signals in memory as
                                                        numpy arrays.
        load_audio_tensor (bool, default=False):    If true, loads the audio signals in memory as
                                                        torch tensors.
        dataset_path (str):                         The path to the dataset.

    Returns:
        dataset (list[SpeechSignal]):               The requested dataset. A list of `n_samples`
                                                        emements from the original dataset.
    """

    if seed >= 0:
        rd.seed(seed)
    LOGGER.info("num_samples: %i", n_samples)
    LOGGER.info("min_duration: %f", min_duration)
    LOGGER.info("random: %s" + (f" (seed = {seed})" if random else ""), random)
    LOGGER.info("load_audio_array: %s", load_audio_array)
    LOGGER.info("load_audio_tensor: %s", load_audio_tensor)
    if n_samples > 0:
        LOGGER.debug("Loading %i/%i samples", n_samples, NUM_SAMPLES)

    if not dataset_path:
        dataset_path = "data/LibriSpeech/train-clean-100"
    assert os.path.isdir(dataset_path), "cannot find librispeech dataset."

    dataset: list[SpeechSignal] = _data_set_loading_loop(
        load_audio_array=load_audio_array,
        load_audio_tensor=load_audio_tensor,
        dataset_path=dataset_path,
        min_duration=min_duration,
        n_samples=-1 if (random or (n_samples == -1)) else n_samples,
    )

    if random:
        rd.shuffle(dataset)

    if n_samples > 0:
        assert len(dataset) >= n_samples
        return dataset[:n_samples]

    LOGGER.info("final number of samples: %i", len(dataset))

    return dataset


def speech_data_set_iterator(random: bool = True) -> Iterator[SpeechSignal]:
    for speech_signal in get_speech_data_set(random=random):
        yield speech_signal


def main() -> None:
    """
    Simply loop through the data set and print information about each sample.
    """
    durations: list[float] = []
    dataset: list[SpeechSignal] = get_speech_data_set(
        random=False,
        load_audio_tensor=False,
        # min_duration=6.0,
    )
    for speech_signal in tqdm(dataset):
        # LOGGER.info("audio filename: %s", speech_signal.filename)
        # LOGGER.info("transcript: %s", speech_signal.transcript)
        speech_signal.load_audio()
        durations.append(
            len(speech_signal.signal) / 16_000,
        )
        # assert durations[-1] >= 6.0
        speech_signal.unload_audio()

    plt.hist(durations)
    plt.show()


if __name__ == "__main__":
    main()
