import numpy as np
import pytest

try:
    from sklearn.neighbors import KDTree
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from cnnclustering import cluster
from cnnclustering._types import (
    InputDataNeighboursSequence,
    InputDataExtPointsMemoryview,
    NeighboursGetterLookup,
    NeighboursGetterExtBruteForce,
    NeighboursList,
    NeighboursExtVector,
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


@pytest.mark.parametrize(
    (
        "input_data_t,neighbours_getter_t,neighbours_t,neighbour_neighbours_t,"
        "metric_t,similarity_checker_t,queue_t,fitter_t"
    ),
    [
        (
            InputDataNeighboursSequence,
            NeighboursGetterLookup,
            NeighboursList,
            NeighboursList,
            None,
            SimilarityCheckerContains,
            QueueFIFODeque,
            FitterBFS
        )
    ]
)
@pytest.mark.parametrize(
    "n_samples,gen_func,gen_kwargs,convert_to,r,c",
    [(1500, "moons", {}, "neighbours", 0.2, 5)]
)
def test_cluster_toy_data_with_reference(
        n_samples, gen_func, gen_kwargs, toy_data_points, convert_to,
        input_data_t, neighbours_getter_t, neighbours_t,
        neighbour_neighbours_t,
        metric_t, similarity_checker_t, queue_t, fitter_t,
        r, c):
    if not SKLEARN_FOUND:
        pytest.skip("Test module requires scikit-learn.")

    points, reference_labels = toy_data_points

    if convert_to == "neighbours":
        tree = KDTree(points)
        input_data_raw = tree.query_radius(
            points, r=r, return_distance=False
            )
    else:
        input_data_raw = points

    input_data = input_data_t(input_data_raw)
    neighbours_getter = neighbours_getter_t()
    neighbours = neighbours_t()
    neighbour_neighbours = neighbour_neighbours_t()

    if metric_t is not None:
        metric = metric_t()
    else:
        metric = None

    similarity_checker = similarity_checker_t()
    queue = queue_t()
    fitter = fitter_t()

    clustering = cluster.Clustering(
        input_data=input_data,
        neighbours_getter=neighbours_getter,
        neighbours=neighbours,
        neighbour_neighbours=neighbour_neighbours,
        metric=metric,
        similarity_checker=similarity_checker,
        queue=queue,
        fitter=fitter,
    )

    clustering.fit(r, c)

    labels_found = equalise_labels(clustering._labels.labels)
    labels_expected = equalise_labels(reference_labels)
    np.testing.assert_array_equal(
        labels_found,
        labels_expected,
        )
