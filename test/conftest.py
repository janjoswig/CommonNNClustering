import pytest

try:
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False


@pytest.fixture
def toy_data_points(request):
    n_samples = request.node.funcargs.get("n_samples")
    gen_func = request.node.funcargs.get("gen_func")
    gen_kwargs = request.node.funcargs.get("gen_kwargs")

    generation_functions = {
        "moons": datasets.make_moons,
    }

    points, reference_labels = generation_functions.get(gen_func)(
        n_samples=n_samples, **gen_kwargs
        )

    points = StandardScaler().fit_transform(points)
    reference_labels += 1
    return points, reference_labels
