import numpy as np

from acoustix.room.audio_objects import SpeechSourceContinuous


def test_continuous_source() -> None:
    speech_source: SpeechSourceContinuous = SpeechSourceContinuous(
        name="",
        position=np.zeros(3),
        n_time_samples=16_000,
    )
    for _ in range(200):
        speech_source.load_signal()
