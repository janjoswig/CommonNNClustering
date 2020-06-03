import pytest
from sklearn import datasets
import cnnclustering.cnn as cnn


@pytest.fixture
def circles():
    noisy_circles, _ = datasets.make_circles(
        n_samples=2000,
        factor=.5,
        noise=.05
        )
    return noisy_circles

@pytest.mark.skip(reason="Deprecated API")
def test_fit_circles(circles):
    C = cnn.CNN(circles)
    C.fit(0.25, 20)
    assert C.summary.iloc[-1]["n_cluster"] == 2
