import pytest
from sklearn import datasets

# import cnnclustering.cmsm as cmsm
import cnnclustering.cnn as cnn


BASE_POINTS = [
    [0, 0],       # 0
    [1, 1],       # 1
    [1, 0],       # 2
    [0, -1],      # 3
    [0.5, -0.5],  # 4
    [2,  1.5],    # 5
    [2.5, -0.5],  # 6
    [4, 2],       # 7
    [4.5, 2.5],   # 8
    [5, -1],      # 9
    [5.5, -0.5],  # 10
    [5.5, -1.5],  # 11
    ]

BASE_NEIGHBOURHOODS_e15 = [
    [1, 2, 3, 4],  # 0
    [0, 2, 5],     # 1
    [0, 1, 3, 4],  # 2
    [0, 2, 4],     # 3
    [0, 2, 3],     # 4
    [1],           # 5
    [],            # 6
    [8],           # 7
    [7],           # 8
    [10, 11],      # 9
    [9, 11],       # 10
    [9, 10],       # 11
    ]

BASE_LABELS_e15_0 = [1, 1, 1, 1, 1, 1, 0, 2, 2, 3, 3, 3]
BASE_LABELS_e15_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2]
BASE_LABELS_e15_2 = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


@pytest.fixture
def base_points():
    return BASE_POINTS


@pytest.fixture
def base_neighbourhoods_e15():
    return BASE_NEIGHBOURHOODS_e15


@pytest.fixture
def base_labels_e15_0():
    return BASE_LABELS_e15_0


@pytest.fixture
def base_labels_e15_1():
    return BASE_LABELS_e15_1


@pytest.fixture
def base_labels_e15_2():
    return BASE_LABELS_e15_2


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
