import logging
import os
import random as rd
from fnmatch import fnmatch

from tqdm import tqdm

from ..audio_signal import AudioSignalObject

LOGGER: logging.Logger = logging.getLogger(__name__)

NUM_SAMPLES: int = 80


def _data_set_loading_loop(
    keep_indices: list[int],
    load_audio_array: bool,
    load_audio_tensor: bool,
    dataset_path: str,
) -> list[AudioSignalObject]:
    LOGGER.info("Loading dataset '%s'", dataset_path)

    data_set: list[AudioSignalObject] = []

    next_sample_to_keep: int = keep_indices.pop(0)
    sample_global_index: int = 0

    with tqdm(total=len(keep_indices)) as progress_bar:
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                if fnmatch(name, "*.wav"):
                    if sample_global_index == next_sample_to_keep:
                        data_set.append(
                            AudioSignalObject(
                                file_path=os.path.join(path, name),
                                load_audio_array=load_audio_array,
                                load_audio_tensor=load_audio_tensor,
                            )
                        )
                        if len(keep_indices) == 0:
                            return data_set
                        next_sample_to_keep = keep_indices.pop(0)

                    sample_global_index += 1
                    progress_bar.update()

    return data_set


# print('Number of audio files:', format(len(wav_files)))
#
# # save file names for convenient feature extraction by matlab
# with open('file_names.txt', 'w') as f:
#     for item in wav_files:
#         f.write("%s\n" % item)
#
# # check duration of the dataset
# total_len = 0
# for k in range(len(wav_files)):
#     x, sr = soundfile.read( wav_files[k])
#     total_len = total_len + x.shape[0]/sr

# print("Total duration of the dataset: %.2f h." % (total_len/3600))


def get_cf_dataset(
    random: bool = True,
    seed: int = -1,
    # n_samples: int = NUM_SAMPLES,
    n_samples: int = -1,
    load_audio_array: bool = False,
    load_audio_tensor: bool = False,
    dataset_path: str = "",
) -> list[AudioSignalObject]:
    if seed >= 0:
        rd.seed(seed)
    LOGGER.info("num_samples: %i", n_samples)
    LOGGER.info("random: %s" + (f" (seed = {seed})" if random else ""), random)
    LOGGER.info("load_audio_array: %s", load_audio_array)
    LOGGER.info("load_audio_tensor: %s", load_audio_tensor)

    if n_samples < 0:
        n_samples = NUM_SAMPLES

    if not dataset_path:
        dataset_path = "data/CBFdataset"

    keep_indices: list[int] = list(range(NUM_SAMPLES))
    if n_samples != NUM_SAMPLES:
        if random:
            LOGGER.debug("Loading %i/%i samples", n_samples, NUM_SAMPLES)
            keep_indices = rd.sample(
                keep_indices,
                k=n_samples,
            )
            keep_indices.sort()
        else:
            keep_indices = list(range(n_samples))
    else:
        LOGGER.info("Loading all the %i samples", NUM_SAMPLES)

    data_set: list[AudioSignalObject] = _data_set_loading_loop(
        keep_indices=keep_indices,
        load_audio_array=load_audio_array,
        load_audio_tensor=load_audio_tensor,
        dataset_path=dataset_path,
    )

    assert len(data_set) == n_samples

    if random:
        rd.shuffle(data_set)

    return data_set
