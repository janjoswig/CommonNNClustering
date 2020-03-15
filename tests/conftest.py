import pytest
from sklearn import datasets

import core.cmsm
import core.cnn


@pytest.fixture
def empty_cobj():
    return core.cnn.CNN(alias="empty")


@pytest.fixture
def random_circles_cobj():
    return core.cnn.CNN(
        alias="random_circles",
        data=[
            datasets.make_circles(n_samples=2000, factor=.5, noise=.05),
            datasets.make_circles(n_samples=1500, factor=.6, noise=.04),
            ]
        )