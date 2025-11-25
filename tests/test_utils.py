import numpy as np

from acoustix.utils import angle_between_two_vectors, rotate_2d_vector, rotate_3d_vector


def test_rotate_vector():
    np.testing.assert_equal(
        rotate_2d_vector(
            vec=np.array(
                [10, 10],
            ),
            angle=-np.pi / 2,
        ),
        np.array(
            [10, -10],
        ),
    )
    np.testing.assert_array_almost_equal(
        rotate_2d_vector(
            vec=np.array(
                [1, 0],
            ),
            angle=np.pi / 2,
        ),
        np.array(
            [0, 1],
        ),
    )
    np.testing.assert_array_almost_equal(
        rotate_3d_vector(
            vec=np.array(
                [1, 0, 0.5],
            ),
            angle_xy=np.pi / 2,
        ),
        np.array(
            [0, 1, 0.5],
        ),
    )


def test_doa() -> None:
    agent_ori = np.array([0, 1])
    agent_to_source = np.array([-1, 1])
    doa = angle_between_two_vectors(agent_ori, agent_to_source)
    assert doa == np.pi / 4

    agent_ori = np.array([0, 1])
    agent_to_source = np.array([1, 1])
    doa = angle_between_two_vectors(agent_ori, agent_to_source)
    assert doa == -np.pi / 4

    agent_ori = np.array([0, 1])
    agent_to_source = np.array([-1, -1])
    doa = angle_between_two_vectors(agent_ori, agent_to_source)
    assert doa == 3 * np.pi / 4

    agent_ori = np.array([0, 1])
    agent_to_source = np.array([1, -1])
    doa = angle_between_two_vectors(agent_ori, agent_to_source)
    assert doa == -3 * np.pi / 4
