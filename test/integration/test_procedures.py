import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

try:
    from sklearn.neighbors import KDTree
    from sklearn.metrics import pairwise_distances
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from cnnclustering import cluster, hooks
from cnnclustering._primitive_types import P_AVALUE
from cnnclustering import _types, _fit


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
        "recipe,converter"
    ),
    [
        (
            {
                "input_data": _types.InputDataNeighbourhoodsSequence,
                "fitter.getter": (
                    _types.NeighboursGetterLookup,
                    (), {"is_selfcounting": True}
                ),
                "fiiter.na": _types.NeighboursList,
                "fitter.checker": _types.SimilarityCheckerContains,
                "queue": _types.QueueFIFODeque,
                "fitter": _fit.FitterBFS,
            },
            convert_points_to_neighbours_array_array
        ),
        pytest.param(
            {
                "input_data": _types.InputDataExtComponentsMemoryview,
                "fitter.getter": _types.NeighboursGetterExtBruteForce,
                "fiiter.na": _types.NeighboursExtVector,
                "fitter.checker": _types.SimilarityCheckerExtContains,
                "fitter.getter.dgetter": _types.DistanceGetterExtMetric,
                "fitter.getter.dgetter.metric": _types.MetricExtPrecomputed,
                "queue": _types.QueueExtFIFOQueue,
                "fitter": _fit.FitterExtBFS,
            },
            convert_points_to_distances_array2d,
            marks=[pytest.mark.heavy]
        ),
        pytest.param(
            {
                "input_data": _types.InputDataExtComponentsMemoryview,
                "fitter.getter": _types.NeighboursGetterExtBruteForce,
                "fiiter.na": _types.NeighboursExtVector,
                "fitter.checker": _types.SimilarityCheckerExtContains,
                "fitter.getter.dgetter": _types.DistanceGetterExtMetric,
                "fitter.getter.dgetter.metric": _types.MetricExtEuclidean,
                "queue": _types.QueueExtFIFOQueue,
                "fitter": _fit.FitterExtBFS,
            },
            no_convert,
            marks=[pytest.mark.heavy]
        )
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
        r, c, fit_kwargs, max_diff, recipe):

    if not SKLEARN_FOUND:
        pytest.skip("Test function requires scikit-learn.")

    points, reference_labels = toy_data_points
    points = converter(points, r, c)

    builder = cluster.ClusteringBuilder(
        points, preparation_hook=hooks.prepare_pass, **recipe)
    clustering = builder.build()

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
    _metric = (
        _types.MetricExtEuclideanPeriodicReduced,
        (np.array([360, 360], dtype=float),), {}
    )
    builder = cluster.ClusteringBuilder(
        data,
        fitter__getter__dgetter__metric=_metric
        )
    clustering = builder.build()

    fig, ax = plt.subplots()
    clustering.evaluate(ax=ax)
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

    fig, ax = plt.subplots()
    clustering.evaluate(ax=ax, annotate=False)
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
        "recipe,other_recipe,converter"
    ),
    [
        pytest.param(
            {
                "input_data": _types.InputDataNeighbourhoodsSequence,
                "predictor.na": _types.NeighboursList,
                "predictor.getter": (
                    _types.NeighboursGetterLookup,
                    (), {"is_selfcounting": True}
                ),
                "predictor.checker": _types.SimilarityCheckerContains,
                "predictor": _fit.PredictorFirstmatch
            },
            {
                "input_data": _types.InputDataNeighbourhoodsSequence,
            },
            convert_points_to_neighbours_array_array_other,
        ),
    ]
)
@pytest.mark.parametrize(
    "n_samples,gen_func,gen_kwargs,stride,r,c",
    [(1500, "moons", {}, 2, 0.2, 2)]
)
def test_predict_for_toy_data_from_reference(
        n_samples, gen_func, gen_kwargs, toy_data_points, stride, converter,
        r, c, recipe, other_recipe):

    if not SKLEARN_FOUND:
        pytest.skip("Test function requires scikit-learn.")

    other_points, other_reference_labels = toy_data_points
    points = np.array(other_points[::stride], order="C", dtype=P_AVALUE)
    reference_labels = other_reference_labels[::stride]

    points, other_points = converter(points, other_points, r, c)

    builder = cluster.ClusteringBuilder(
        points,
        preparation_hook=hooks.prepare_pass,
        registered_recipe_key="None",
        labels=(_types.Labels.from_sequence, (reference_labels,), {}),
        **recipe
        )
    clustering = builder.build()

    builder = cluster.ClusteringBuilder(
        other_points,
        preparation_hook=hooks.prepare_pass,
        registered_recipe_key="None",
        **other_recipe
        )
    other_clustering = builder.build()

    clustering.predict(other_clustering, r, c)

    labels_found = equalise_labels(other_clustering._labels.labels)
    labels_expected = equalise_labels(other_reference_labels)
    np.testing.assert_array_equal(
        labels_found,
        labels_expected,
        )
