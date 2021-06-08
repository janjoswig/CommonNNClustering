import numpy as np
import matplotlib as mpl
import pytest

try:
    from sklearn.neighbors import KDTree
    from sklearn.metrics import pairwise_distances
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from cnnclustering import cluster
from cnnclustering._primitive_types import P_AVALUE
from cnnclustering._types import (
    Labels,
    InputDataNeighboursSequence,
    InputDataExtPointsMemoryview,
    NeighboursGetterBruteForce,
    NeighboursGetterLookup,
    NeighboursGetterExtBruteForce,
    NeighboursList,
    NeighboursSet,
    NeighboursExtVector,
    MetricDummy,
    MetricPrecomputed,
    MetricEuclidean,
    MetricExtDummy,
    MetricExtPrecomputed,
    MetricExtEuclidean,
    MetricExtEuclideanPeriodicReduced,
    SimilarityCheckerContains,
    SimilarityCheckerExtContains,
    QueueFIFODeque,
    QueueExtFIFOQueue,
)
from cnnclustering._fit import FitterExtBFS, FitterBFS
from cnnclustering._fit import PredictorFirstmatch


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

    for index, label in enumerate(labels):
        if label == 0:
            continue

        new_label = label_map[label]
        labels[index] = new_label

    return labels


def convert_points_to_neighbours_array_array(points, r, c):
    tree = KDTree(points)
    return tree.query_radius(
        points, r=r, return_distance=False
        )


def convert_points_to_distances_array2d(points, r, c):
    return pairwise_distances(points)


def no_convert(points, r, c):
    return points


def convert_points_to_neighbours_array_array_other(points, other_points, r, c):
    tree = KDTree(points)

    neighbours = tree.query_radius(
        points, r=r, return_distance=False
        )

    neighbours_other = tree.query_radius(
        other_points, r=r, return_distance=False
        )

    return neighbours, neighbours_other


def no_convert_other(points, other_points, r, c):
    return points, other_points


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
                ("neighbours_getter", NeighboursGetterExtBruteForce, (), {}),
                ("neighbours", NeighboursExtVector, (250,), {}),
                ("neighbour_neighbours", NeighboursExtVector, (250,), {}),
                ("metric", MetricExtPrecomputed, (), {}),
                ("similarity_checker", SimilarityCheckerExtContains, (), {}),
                ("queue", QueueExtFIFOQueue, (), {}),
                ("fitter", FitterExtBFS, (), {}),
            ),
            convert_points_to_distances_array2d,
            marks=[pytest.mark.heavy]
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
    "n_samples,gen_func,gen_kwargs,r,c,fit_kwargs,max_diff",
    [
        (1500, "moons", {}, 0.2, 5, {}, 0),
        (
            1500, "blobs", {
                "cluster_std": [1.0, 2.5, 0.5],
                "random_state": 170,
                },
            0.25, 20,
            {"member_cutoff": 20},
            12
        )
    ]
)
def test_fit_toy_data_with_reference(
        n_samples, gen_func, gen_kwargs, toy_data_points, converter,
        r, c, fit_kwargs, max_diff, components):

    if not SKLEARN_FOUND:
        pytest.skip("Test function requires scikit-learn.")

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

    clustering.fit(r, c, **fit_kwargs)

    labels_found = equalise_labels(clustering._labels.labels)

    reference_labels[labels_found == 0] = 0
    labels_expected = equalise_labels(reference_labels)

    try:
        np.testing.assert_array_equal(
            labels_found,
            labels_expected,
            )
    except AssertionError as e:
        for line in str(e).splitlines():
            if not line.startswith("Mismatched elements:"):
                continue

            diff = int(line.split()[2])
            if diff > max_diff:
                raise e

            break


@pytest.mark.image_regression
def test_fit_evaluate_regression(datadir, image_regression):

    mpl.use("agg")
    data = np.load(datadir / "backbone_dihedrals.npy")
    clustering = cluster.prepare_clustering(data)

    clustering._metric = MetricExtEuclideanPeriodicReduced(
        np.array([360, 360], dtype=float)
    )

    fig, *_ = clustering.evaluate()
    fig.tight_layout()
    figname_original = datadir / "backbone_dihedrals_original.png"
    fig.savefig(figname_original)
    image_regression.check(
        figname_original.read_bytes(),
        basename="test_fit_evaluate_regression_backbone_dihedrals_original",
        diff_threshold=5,
        )

    clustering.fit(
        10, 15, member_cutoff=50,
        info=False, v=False, record=False, record_time=False
        )

    fig, *_ = clustering.evaluate(annotate=False)
    fig.tight_layout()
    figname_clustered = datadir / "backbone_dihedrals_clustered.png"
    fig.savefig(figname_clustered)
    image_regression.check(
        figname_clustered.read_bytes(),
        basename="test_fit_evaluate_regression_backbone_dihedrals_clustered",
        diff_threshold=5,
        )


@pytest.mark.parametrize(
    (
        "components,other_components,converter"
    ),
    [
        pytest.param(
            (
                ("input_data", InputDataNeighboursSequence, (), {}),
                (
                    "neighbours_getter", NeighboursGetterLookup,
                    (), {"is_selfcounting": True}
                ),
                ("predictor", PredictorFirstmatch, (), {})
            ),
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
            ),
            convert_points_to_neighbours_array_array_other,
        ),
        pytest.param(
            (
                ("input_data", InputDataExtPointsMemoryview, (), {}),
                ("neighbours_getter", NeighboursGetterExtBruteForce, (), {}),
                ("predictor", PredictorFirstmatch, (), {})
            ),
            (
                ("input_data", InputDataExtPointsMemoryview, (), {}),
                ("neighbours_getter", NeighboursGetterExtBruteForce, (), {}),
                ("neighbours", NeighboursExtVector, (500,), {}),
                ("neighbour_neighbours", NeighboursExtVector, (500,), {}),
                ("metric", MetricExtEuclidean, (), {}),
                ("similarity_checker", SimilarityCheckerExtContains, (), {}),
            ),
            no_convert_other,
            marks=[pytest.mark.heavy]
        )
    ]
)
@pytest.mark.parametrize(
    "n_samples,gen_func,gen_kwargs,stride,r,c",
    [(1500, "moons", {}, 2, 0.2, 2)]
)
def test_predict_for_toy_data_from_reference(
        n_samples, gen_func, gen_kwargs, toy_data_points, stride, converter,
        r, c, components, other_components):

    if not SKLEARN_FOUND:
        pytest.skip("Test function requires scikit-learn.")

    other_points, other_reference_labels = toy_data_points
    points = np.array(other_points[::stride], order="C", dtype=P_AVALUE)
    reference_labels = other_reference_labels[::stride]

    points, other_points = converter(points, other_points, r, c)

    prepared_components = {}
    for component_kw, component_type, args, kwargs in components:
        if component_kw == "input_data":
            args = (points, *args)

        if component_type is not None:
            prepared_components[component_kw] = component_type(
                *args, **kwargs
            )
    prepared_components["labels"] = Labels.from_sequence(reference_labels)

    other_prepared_components = {}
    for component_kw, component_type, args, kwargs in other_components:
        if component_kw == "input_data":
            args = (other_points, *args)

        if component_type is not None:
            other_prepared_components[component_kw] = component_type(
                *args, **kwargs
            )

    clustering = cluster.Clustering(
        **prepared_components
    )

    other_clustering = cluster.Clustering(
        **other_prepared_components
        )

    clustering.predict(other_clustering, r, c)

    labels_found = equalise_labels(other_clustering._labels.labels)
    labels_expected = equalise_labels(other_reference_labels)
    np.testing.assert_array_equal(
        labels_found,
        labels_expected,
        )
