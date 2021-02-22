"""Testing functions calculating data point distances"""

import math

import numpy as np

import cnnclustering._cfits as cfits


def ref_distance_euclidean(a, b):
    total = 0
    for component_a, component_b in zip(a, b):
        total += (component_a - component_b)**2
    return math.sqrt(total)


def test_ref_distance_euclidean():
    assert ref_distance_euclidean((0, 0), (0, 0)) == 0
    assert ref_distance_euclidean((0, 0), (0, 1)) == 1
    assert ref_distance_euclidean((0, -1), (0, 1)) == 2
    assert ref_distance_euclidean((1, 1), (1, 1)) == 0


CASES = [  # p1, p2, result
    ([0, 0], [0, 1]),
    ([0, 0], [0, 0]),
    ([1., 2.], [3., 4.]),
    ([11.4, 12.2, 7.5], [3.3, 4.8, 1.2]),
    ]


class TestPython:
    pass


class TestEuclideanCython:
    def test_get_distance_squared_euclidean_PointsArray(self):
        for p1, p2 in CASES:
            p1 = np.asarray(p1, dtype=np.float_)
            p2 = np.asarray(p2, dtype=np.float_)

            ref_result = ref_distance_euclidean(p1, p2)**2

            result = cfits._get_distance_squared_euclidean_PointsArray(
                p1, p2, parallel=False
                )

            assert np.isclose(result, ref_result)

            # result = cfits._get_distance_squared_euclidean_PointsArray(
            #     p1, p2, parallel=True
            #     )

            # assert np.isclose(result, ref_result)
