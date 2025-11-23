import numpy as np

from acoustix.microphone_arrays import (
    BinauralArray,
    MicArray,
    MonoArray,
    SquareArray,
    TriangleArray,
    UniformLinearArray,
)

POSITION: np.ndarray = np.array([12, 4, 1.8])


def _test_mic_pattern(array: MicArray) -> None:
    for mic in array.microphones:
        assert mic.pattern == array.mic_pattern


def test_mono_array() -> None:
    array: MonoArray = MonoArray(
        position=POSITION,
    )
    _test_mic_pattern(array=array)


def test_binaural_array() -> None:
    array: BinauralArray = BinauralArray(
        position=POSITION,
    )
    _test_mic_pattern(array=array)

    array = BinauralArray(
        position=POSITION,
        mic_pattern="subcard",
    )
    _test_mic_pattern(array=array)


def test_triangle_array() -> None:
    array: TriangleArray = TriangleArray(
        position=POSITION,
    )
    _test_mic_pattern(array=array)

    array = TriangleArray(
        position=np.array(
            [12, 4, 1.8],
        ),
        mic_pattern="subcard",
    )
    _test_mic_pattern(array=array)


def test_square_array() -> None:
    array: SquareArray = SquareArray(
        position=POSITION,
    )
    _test_mic_pattern(array=array)


def test_ula() -> None:
    array: UniformLinearArray = UniformLinearArray(
        n_mics=6,
        position=POSITION,
    )
    _test_mic_pattern(array=array)
