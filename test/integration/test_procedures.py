import numpy as np
import pytest
from sklearn import neighbors

try:
    from sklearn.neighbors import KDTree
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from cnnclustering import cluster
from cnnclustering._types import (
    InputDataNeighboursSequence,
    InputDataExtPointsMemoryview,
    NeighboursGetterBruteForce,
    NeighboursGetterLookup,
    NeighboursGetterExtBruteForce,
    NeighboursList,
    NeighboursSet,
    NeighboursExtVector,
    MetricDummy,
    MetricEuclidean,
    MetricExtDummy,
    MetricExtEuclidean,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    QueueFIFODeque,
    QueueExtFIFOQueue,
)
from cnnclustering._fit import FitterExtBFS, FitterBFS


pytestmark = pytest.mark.sklearn


class LabelTracker(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counter = 1

    def __missing__(self, key):
        self.__setitem__(key, self._counter)
        self._counter += 1
        return self.__getitem__(key)


def equalise_labels(labels):
    label_map = LabelTracker()

    for index, l in enumerate(labels):
        if l == 0:
            continue

        new_label = label_map[l]
        labels[index] = new_label

    return labels


def convert_points_to_neighbours_array_array(points, r, c):
    tree = KDTree(points)
    return tree.query_radius(
        points, r=r, return_distance=False
        )

def convert_points_to_neighbours_list_set(points, r, c):
    tree = KDTree(points)
    points = tree.query_radius(
        points, r=r, return_distance=False
        )
    return [set(neighbours) for neighbours in points]


def no_convert(points, r, c):
    return points


@pytest.mark.parametrize(
    (
        "components,converter"
    ),
    [
        (
            (
                ("input_data", InputDataNeighboursSequence, (), {}),
                (
                    "neighbours_getter", NeighboursGetterLookup,
                    (), {"is_selfcounting": True}
                ),
                ("neighbours", NeighboursList, (), {}),
                ("neighbour_neighbours", NeighboursList, (), {}),
                ("metric", MetricDummy, (), {}),
                ("similarity_checker", SimilarityCheckerContains, (), {}),
                ("queue", QueueFIFODeque, (), {}),
                ("fitter", FitterBFS, (), {}),
            ),
            convert_points_to_neighbours_array_array
        ),
        pytest.param(
            (
                ("input_data", InputDataExtPointsMemoryview, (), {}),
                ("neighbours_getter", NeighboursGetterBruteForce, (), {}),
                ("neighbours", NeighboursExtVector, (500,), {}),
                ("neighbour_neighbours", NeighboursExtVector, (500,), {}),
                ("metric", MetricEuclidean, (), {}),
                ("similarity_checker", SimilarityCheckerContains, (), {}),
                ("queue", QueueFIFODeque, (), {}),
                ("fitter", FitterBFS, (), {}),
            ),
            no_convert,
            marks=[pytest.mark.heavy]
        ),
        pytest.param(
            (
                ("input_data", InputDataExtPointsMemoryview, (), {}),
                ("neighbours_getter", NeighboursGetterExtBruteForce, (), {}),
                ("neighbours", NeighboursExtVector, (500,), {}),
                ("neighbour_neighbours", NeighboursExtVector, (500,), {}),
                ("metric", MetricExtEuclidean, (), {}),
                ("similarity_checker", SimilarityCheckerExtContains, (), {}),
                ("queue", QueueExtFIFOQueue, (), {}),
                ("fitter", FitterExtBFS, (), {}),
            ),
            no_convert,
            marks=[pytest.mark.heavy]
        ),
    ]
)
@pytest.mark.parametrize(
    "n_samples,gen_func,gen_kwargs,r,c",
    [(1500, "moons", {}, 0.2, 5)]
)
def test_cluster_toy_data_with_reference(
        n_samples, gen_func, gen_kwargs, toy_data_points, converter,
        r, c, components):
    if not SKLEARN_FOUND:
        pytest.skip("Test module requires scikit-learn.")

    points, reference_labels = toy_data_points
    points = converter(points, r, c)

    prepared_components = {}
    for component_kw, component_type, args, kwargs in components:
        if component_kw == "input_data":
            args = (points, *args)

        if component_type is not None:
            prepared_components[component_kw] = component_type(
                *args, **kwargs
            )

    clustering = cluster.Clustering(
        **prepared_components
    )

    clustering.fit(r, c)

    labels_found = equalise_labels(clustering._labels.labels)
    labels_expected = equalise_labels(reference_labels)
    np.testing.assert_array_equal(
        labels_found,
        labels_expected,
        )
