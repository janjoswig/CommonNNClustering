import math

import numpy as np
import pytest

from cnnclustering._primitive_types import P_AVALUE
from cnnclustering._types import (
    InputDataExtComponentsMemoryview,
    Metric,
    MetricDummy,
    MetricPrecomputed,
    MetricEuclidean,
    MetricEuclideanReduced,
    MetricExtInterface,
    MetricExtDummy,
    MetricExtPrecomputed,
    MetricExtEuclidean,
    MetricExtEuclideanReduced,
    MetricExtEuclideanPeriodicReduced,
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
        "metric,metric_args,metric_kwargs,isinstance_of",
        [
            (MetricDummy, (), {}, [Metric]),
            (MetricPrecomputed, (), {}, [Metric]),
            (MetricEuclidean, (), {}, [Metric]),
            (MetricEuclideanReduced, (), {}, [Metric]),
            (MetricExtDummy, (), {}, [Metric, MetricExtInterface]),
            (MetricExtPrecomputed, (), {}, [Metric, MetricExtInterface]),
            (MetricExtEuclidean, (), {}, [Metric, MetricExtInterface]),
            (MetricExtEuclideanReduced, (), {}, [Metric, MetricExtInterface]),
            (
                MetricExtEuclideanPeriodicReduced,
                (np.ones(2),), {},
                [Metric, MetricExtInterface]
            ),
        ]
    )
    def test_inheritance(self, metric, metric_args, metric_kwargs, isinstance_of):
        _metric = metric(*metric_args, **metric_kwargs)
        assert all([isinstance(_metric, x) for x in isinstance_of])

    @pytest.mark.parametrize(
        "metric,metric_args,metric_kwargs,metric_is_ext,ref_func",
        [
            (
                MetricEuclidean, (), {}, False,
                ref_distance_euclidean,
            ),
            (
                MetricEuclideanReduced, (), {}, False,
                ref_distance_euclidean,
            ),
            (
                MetricExtEuclidean, (), {}, True,
                ref_distance_euclidean,
            ),
            (
                MetricExtEuclideanReduced, (), {}, True,
                ref_distance_euclidean,
            ),
        ]
    )
    @pytest.mark.parametrize(
        "input_data_type,data,other_data,input_is_ext",
        [
            (
                InputDataExtComponentsMemoryview,
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
            self, metric, metric_args, metric_kwargs, metric_is_ext,
            input_data_type, data, other_data, input_is_ext, ref_func):

        if metric_is_ext and (not input_is_ext):
            # pytest.skip("Bad combination of component types.")
            return

        _metric = metric(*metric_args, **metric_kwargs)

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
                ref_distance = _metric.adjust_radius(ref_func(a, b))

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
                ref_distance = _metric.adjust_radius(ref_func(a, b))

                distance = _metric.calc_distance_other(
                    i, j, input_data, other_input_data
                    )

                np.testing.assert_approx_equal(
                    distance, ref_distance, significant=12
                    )

    @pytest.mark.parametrize(
        "metric,metric_args,metric_is_ext",
        [
            (
                MetricPrecomputed, (), False,
            ),
            (
                MetricExtPrecomputed, (), True,
            ),
        ]
    )
    @pytest.mark.parametrize(
        "input_data_type,data,other_data,input_is_ext",
        [
            (
                InputDataExtComponentsMemoryview,
                np.array([
                    [0., 0.2, 0.3],
                    [0.2, 0., 1.],
                    [0.3, 1., 0.],
                    ], order="C", dtype=P_AVALUE
                ),
                np.array([
                    [0., 1., 2.],
                    [1., 0., 1.],
                    [2., 1., 0.],
                    [3., 2., 1.],
                    ], order="C", dtype=P_AVALUE
                ),
                True
            ),
        ],
    )
    def test_precomputed_distance(
            self, metric, metric_args, metric_is_ext,
            input_data_type, data, other_data, input_is_ext):

        if metric_is_ext and (not input_is_ext):
            # pytest.skip("Bad combination of component types.")
            return

        _metric = metric()

        input_data = input_data_type(data)

        for i in range(input_data.n_points):
            for j in range(i + 1, input_data.n_points):
                ref_distance = _metric.adjust_radius(input_data.get_component(i, j))

                distance = _metric.calc_distance(i, j, input_data)

                np.testing.assert_approx_equal(
                    distance, ref_distance, significant=12
                    )

        other_input_data = input_data_type(other_data)

        for i in range(other_input_data.n_points):
            for j in range(input_data.n_points):
                ref_distance = _metric.adjust_radius(
                    other_input_data.get_component(i, j)
                    )

                distance = _metric.calc_distance_other(
                    i, j, input_data, other_input_data
                    )

                np.testing.assert_approx_equal(
                    distance, ref_distance, significant=12
                    )
