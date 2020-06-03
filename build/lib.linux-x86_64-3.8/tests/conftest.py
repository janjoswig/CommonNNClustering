import pytest
from sklearn import datasets

import core.cmsm as cmsm
import core.cnn as cnn


@pytest.fixture
def empty_cobj():
    return cnn.CNN(alias="empty")


@pytest.fixture
def std_cobj():
    return cnn.CNN(
        alias="std",
        points=[[[1, 2, 3],
                [4, 5, 6]],
               [[7, 8, 9],
                [10, 11, 12]]]
        )


@pytest.fixture
def hierarchical_cobj():
    cobj = cnn.CNN(
        alias="hierarchical",
        )
                 # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    cobj.labels = [0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 1, 2, 2, 1, 0]

    cobj._children = {
        0: cnn.CNNChild(cobj),
        1: cnn.CNNChild(cobj),
        2: cnn.CNNChild(cobj),
    }

    cobj._children[1].labels = [0, 1, 0, 2, 2, 2, 1]
    cobj._children[1]._refindex_relative = [2, 3, 6, 8, 9, 10, 13]

    cobj._children[1]._children = {
        0: cnn.CNNChild(cobj),
        1: cnn.CNNChild(cobj),
        2: cnn.CNNChild(cobj),
    }

    cobj._children[1]._children[2].labels = [2, 1, 0]
    cobj._children[1]._children[2]._refindex_relative = [3, 4, 5]

    return cobj

@pytest.fixture
def random_circles_cobj():
    return cnn.CNN(
        alias="random_circles",
        data=[
            datasets.make_circles(n_samples=2000, factor=.5, noise=.05),
            datasets.make_circles(n_samples=1500, factor=.6, noise=.04),
            ]
        )
