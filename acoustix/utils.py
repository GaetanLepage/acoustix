import numpy as np
import scipy
import soundcard as sc
import torch
from scipy.io import wavfile
from torch import Tensor


def to_float32(
    signal: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Cast data (typically from WAV) to float32.

    Source:
    https://github.com/LCAV/pyroomacoustics/blob/218c0ec3e8422f1ede30de684520c782a500f9ff/pyroomacoustics/utilities.py#L129

    Args:
        signal (np.ndarray):    Real signal in time domain, typically obtained from WAV file.

    Returns:
        signal (np.ndarray):    `signal` as float32.
    """
    max_val: float = np.abs(signal).max()

    if np.issubdtype(signal.dtype, np.integer):
        # max_val: int = abs(np.iinfo(signal.dtype).min)
        if max_val > 0:
            signal = signal.astype(np.float32) / max_val

    if normalize:
        signal /= max_val

    return signal


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def random_orientation() -> np.ndarray:
    """
    Return a random orientation i.e. a vector that:
        - is 3 dimensional (x, y, z)
        - has norm = 1 (unit vector)
        - is confined to the (x, y) plane: orientation[z] = 0
    """

    orientation = np.append(
        np.random.uniform(
            low=-1,
            high=1,
            size=2,
        ),
        0,
    )

    return normalize_vector(vec=orientation)


def rotate_2d_vector(
    vec: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    https://matthew-brett.github.io/teaching/rotation_2d.html
    https://stackoverflow.com/questions/14607640/rotating-a-vector-in-3d-space
    https://en.wikipedia.org/wiki/Euler_angles
    """
    assert -np.pi <= angle <= np.pi

    assert vec.shape == (2,)

    cos = np.cos(angle)
    sin = np.sin(angle)

    rotation_matrix: np.ndarray = np.array(
        [
            [cos, -sin],
            [sin, cos],
        ]
    )

    rotated_vector: np.ndarray = rotation_matrix @ vec

    return rotated_vector


def play_audio(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
    num_channels: int = -1,
    bytes_per_sample: int = 2,
) -> None:
    """
    Play the audio signal on the computer sound sink.
    """
    assert 1 <= audio_signal.ndim <= 2
    audio_signal = audio_signal.squeeze()

    sc.default_speaker().play(
        data=audio_signal.T,
        samplerate=sample_rate,
        blocking=True,
    )


def save_audio(
    audio_signal: np.ndarray,
    filename: str,
    sample_rate: int = 16_000,
) -> None:
    audio_signal = to_float32(audio_signal)
    print(audio_signal.dtype)
    print(audio_signal.min())
    print(audio_signal.max())
    wavfile.write(
        filename=filename,
        rate=sample_rate,
        data=audio_signal.T,
    )


def rotate_3d_vector(
    vec: np.ndarray,
    angle_xy: float,
) -> np.ndarray:
    """
    Rotate a RD vector around the z-axis
    """
    assert -np.pi <= angle_xy <= np.pi
    # assert - np.pi <= angle_yz <= np.pi

    assert vec.shape == (3,)

    cos = np.cos(angle_xy)
    sin = np.sin(angle_xy)

    rotation_matrix: np.ndarray = np.array(
        [
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1],
        ],
    )

    rotated_vector: np.ndarray = rotation_matrix @ vec

    return rotated_vector


def angle_between_two_vectors(
    vec_1: np.ndarray,
    vec_2: np.ndarray,
) -> np.float32:
    assert vec_1.shape == vec_2.shape == (2,)

    det: float = np.linalg.det(
        [
            vec_1,
            vec_2,
        ],
    )
    dot: float = np.dot(
        vec_1,
        vec_2,
    )

    return np.arctan2(det, dot)


def compute_dist_and_doa(
    agent_2d_pos: np.ndarray,
    agent_2d_ori: np.ndarray,
    source_2d_pos: np.ndarray,
) -> tuple[float, float]:
    agent_to_source_vector: np.ndarray = source_2d_pos - agent_2d_pos

    dist_to_source: float = float(np.linalg.norm(agent_to_source_vector))

    # Compute the DOA
    agent_to_source_unit_vector: np.ndarray = agent_to_source_vector / dist_to_source
    agent_direction_unit_vector: np.ndarray = agent_2d_ori
    doa: float = float(
        angle_between_two_vectors(
            vec_1=agent_direction_unit_vector,
            vec_2=agent_to_source_unit_vector,
        )
    )

    return dist_to_source, doa


def angular_dist_torch(
    theta_1: Tensor,
    theta_2: Tensor,
) -> Tensor:
    """
    Symmetric angular distance

    d(θ_1, θ_2) = π - ||θ_2 - θ_1|[2π] - π|
    """
    delta: Tensor = torch.abs(
        theta_2 - theta_1,
    )
    delta = torch.remainder(delta, 2 * torch.pi)
    return torch.pi - torch.abs(delta - torch.pi)


def angular_dist_numpy(
    doa_1: float,
    doa_2: float,
) -> float:
    """
    Symmetric angular distance

    d(θ_1, θ_2) = π - ||θ_2 - θ_1|[2π] - π|
    """
    return np.pi - np.abs(
        np.abs(
            doa_2 - doa_1,
        )
        - np.pi
    )


def get_min_doa_dist(doas: list[float]) -> float:
    """
    Returns the smallest DOA difference (in radians)
    """
    doas_array: np.ndarray = np.array(doas)
    assert doas_array.ndim == 1
    assert len(doas_array) > 1, "Min doa dist has no meaning if there is less than two sources"
    doas_array = np.expand_dims(doas_array, axis=1)

    dist_matrix: np.ndarray = scipy.spatial.distance.cdist(
        doas_array,
        doas_array,
        metric=angular_dist_numpy,
    )
    np.fill_diagonal(
        dist_matrix,
        val=10,
    )
    min_doa_dist: float = dist_matrix.min()

    return min_doa_dist
