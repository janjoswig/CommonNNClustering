import math

import numpy as np
import pytest

from cnnclustering._primitive_types import P_AVALUE
from cnnclustering._types import (
    InputDataExtPointsMemoryview,
    MetricEuclidean,
    MetricEuclideanReduced,
    MetricExtEuclidean,
    MetricExtEuclideanReduced,
    MetricExtEuclideanPeriodicReduced
)


def ref_distance_euclidean(a, b):
    total = 0
    for component_a, component_b in zip(a, b):
        total += (component_a - component_b) ** 2
    return math.sqrt(total)


def test_ref_distance_euclidean():
    assert ref_distance_euclidean((0, 0), (0, 0)) == 0
    assert ref_distance_euclidean((0, 0), (0, 1)) == 1
    assert ref_distance_euclidean((0, -1), (0, 1)) == 2
    assert ref_distance_euclidean((1, 1), (1, 1)) == 0


class TestMetric:
    @pytest.mark.parametrize(
        "metric,metric_args,metric_is_ext,ref_func",
        [
            (
                MetricEuclidean, (), False,
                lambda a, b: ref_distance_euclidean(a, b),
            ),
            (
                MetricEuclideanReduced, (), False,
                lambda a, b: ref_distance_euclidean(a, b) ** 2,
            ),
            (
                MetricExtEuclidean, (), True,
                lambda a, b: ref_distance_euclidean(a, b),
            ),
            (
                MetricExtEuclideanReduced, (), True,
                lambda a, b: ref_distance_euclidean(a, b) ** 2,
            ),
        ]
    )
    @pytest.mark.parametrize(
        "input_data_type,data,other_data,input_is_ext",
        [
            (
                InputDataExtPointsMemoryview,
                np.array([
                    [0, 0, 0],
                    [1, 1, 1],
                    [1.2, 1.5, 1.3],
                    ], order="C", dtype=P_AVALUE
                ),
                np.array([
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [1.2, 1.5, 1.3],
                    [4.3, 2.5, 0.7],
                    ], order="C", dtype=P_AVALUE
                ),
                True
            ),
        ],
    )
    def test_calc_distance(
            self, metric, metric_args, metric_is_ext,
            input_data_type, data, other_data, input_is_ext, ref_func):

        if metric_is_ext and (not input_is_ext):
            # pytest.skip("Bad combination of component types.")
            return

        _metric = metric()

        input_data = input_data_type(data)

        for i in range(input_data.n_points):
            for j in range(i + 1, input_data.n_points):
                a, b = zip(
                    *(
                        (
                            input_data.get_component(i, d),
                            input_data.get_component(j, d),
                        )
                        for d in range(input_data.n_dim)
                    )
                )
                ref_distance = ref_func(a, b)

                distance = _metric.calc_distance(i, j, input_data)

                np.testing.assert_approx_equal(
                    distance, ref_distance, significant=12
                    )

        other_input_data = input_data_type(other_data)

        for i in range(other_input_data.n_points):
            for j in range(input_data.n_points):
                a, b = zip(
                    *(
                        (
                            other_input_data.get_component(i, d),
                            input_data.get_component(j, d),
                        )
                        for d in range(input_data.n_dim)
                    )
                )
                ref_distance = ref_func(a, b)

                distance = _metric.calc_distance_other(
                    i, j, input_data, other_input_data
                    )

                np.testing.assert_approx_equal(
                    distance, ref_distance, significant=12
                    )
