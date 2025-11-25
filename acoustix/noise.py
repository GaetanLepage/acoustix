import numpy as np

from .room.audio_objects import Source


class RandomNoiseSource(Source):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        snr: float,
    ) -> None:
        super().__init__(
            name=name,
            position=position,
        )

        self.snr = snr

    @property
    def signal(self):
        pass

        # Scale noise


# def noise_data_set_iterator() -> Iterator[AudioSignal]:
#     data_set_path: str = 'data/fma/'
#
#     with open(join(data_set_path, 'fma_metadata/tracks.csv'), 'r') as tracks_metadata:
#         csv_reader = csv.reader(tracks_metadata, delimiter=',')
#         yield next(csv_reader)
#         yield next(csv_reader)
#         yield next(csv_reader)
# for track in csv_reader:
#     yield track

#     for speaker_dir in os.listdir(data_set_path):
#
#         speaker_id: str = path.basename(speaker_dir)
#
#         speaker_dir = path.join(data_set_path, speaker_dir)
#
#         for chapter_dir in os.listdir(speaker_dir):
#
#             chapter_id: str = path.basename(chapter_dir)
#
#             chapter_dir = path.join(speaker_dir, chapter_dir)
#
#             with open(path.join(chapter_dir,
#                                 f"{speaker_id}-{chapter_id}.trans.txt"), 'r') as trans_file:
#
#                 trans_file_lines: List[str] = trans_file.readlines()
#
#             audio_file_names: List[str] = [filename
#                                            for filename in os.listdir(chapter_dir)
#                                            if filename.endswith('.wav')]
#             audio_file_names.sort()
#
#             for sample_index, (trans_line, audio_filename) \
#                     in enumerate(zip(trans_file_lines, audio_file_names)):
#
#                 trans_line_split: List[str] = trans_line.split()
#
#                 base_name: str = f"{speaker_id}-{chapter_id}-{sample_index:04d}"
#
#                 assert trans_line_split[0] == base_name
#
#                 assert path.basename(audio_filename) == base_name + '.flac.wav'
#
#                 audio_filename = path.join(chapter_dir, audio_filename)
#
#                 transcript: str = ' '.join(trans_line_split[1:])
#
#                 yield SpeechSignal(file_path=audio_filename,
#                                    transcript=transcript)

# return


def compute_snr(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:
    signal_power: float = np.sum(signal**2) / len(signal)
    noise_power: float = np.sum(noise**2) / len(noise)
    return 10 * np.log10(signal_power / noise_power)


def scale_noise(
    speech_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr: float,
) -> np.ndarray:
    power_speech: float = np.sum(speech_signal**2) / len(speech_signal)
    power_noise: float = np.sum(noise_signal**2) / len(noise_signal)
    # np.exp is wrong here ? We should raise to 10^
    coefficient: float = np.exp(np.log10(power_speech / power_noise) - target_snr / 10)

    return (noise_signal * coefficient).astype("int16")
