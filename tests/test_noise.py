from typing import Iterator

import numpy as np
from scipy.io import wavfile

from acoustix.audio_signal import SpeechSignal
from acoustix.datasets.librispeech import speech_data_set_iterator
from acoustix.noise import compute_snr, scale_noise


def test_noise_scaling() -> None:
    speech_dataset: Iterator[SpeechSignal] = speech_data_set_iterator()

    speech_signal_obj = next(speech_dataset)
    speech_signal_obj.load_audio()
    speech_signal: np.ndarray = speech_signal_obj.signal

    _, noise_signal = wavfile.read(filename="data/guitar_16k.wav")

    print("speech signal type:", speech_signal.dtype)
    print("noise signal type:", noise_signal.dtype)

    print("no scaling:")
    # play_audio(audio_signal=noise_signal, sample_rate=sampling_freq)
    print(
        "snr =",
        compute_snr(
            signal=speech_signal,
            noise=noise_signal,
        ),
    )

    import matplotlib.pyplot as plt

    plt.plot(speech_signal)
    plt.show()
    plt.close()

    for snr in (-5, 0, 5):
        print("target_snr = ", snr)
        noise_signal = scale_noise(
            speech_signal=speech_signal,
            noise_signal=noise_signal,
            target_snr=snr,
        )
        print(noise_signal.dtype)
        # wavfile.write(
        #     filename=f"output_{snr}.wav",
        #     rate=16000,
        #     data=noise_signal,
        # )
        # play_audio(audio_signal=noise_signal)
        print(
            "computed snr =",
            compute_snr(
                signal=speech_signal,
                noise=noise_signal,
            ),
        )
