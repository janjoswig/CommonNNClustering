import numpy as np

import core.cnn as cnn


class TestLoops:
    def test_check_similarity_array(self):
        assert cnn.CNN.check_similarity_array(
            np.array([0, 1, 2, 4]),
            np.array([1, 2, 5, 6]),
            2
            )

        assert cnn.CNN.check_similarity_array(
            np.array([]),
            np.array([]),
            0
            )

        assert not cnn.CNN.check_similarity_array(
            np.array([0, 1, 2, 4]),
            np.array([1, 2, 5, 6]),
            3
            )

    def test_check_similarity_set(self):
        assert cnn.CNN.check_similarity_set(
            {0, 1, 2, 4},
            {1, 2, 5, 6},
            2
            )

        assert cnn.CNN.check_similarity_set(
            set(),
            set(),
            0
            )

        assert not cnn.CNN.check_similarity_set(
            {0, 1, 2, 4},
            {1, 2, 5, 6},
            3
            )

    def test_check_similarity_sequence(self):
        assert cnn.CNN.check_similarity_sequence(
            [0, 1, 2, 4],
            [1, 2, 5, 6],
            2
            )

        assert cnn.CNN.check_similarity_sequence(
            [],
            [],
            0
            )

        assert not cnn.CNN.check_similarity_sequence(
            [0, 1, 2, 4],
            [1, 2, 5, 6],
            3
            )
