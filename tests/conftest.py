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

BASE_DISTANCES = [
    [0.       , 1.41421356, 1.        , 1.        , 0.70710678,
    2.5       , 2.54950976, 4.47213595, 5.14781507, 5.09901951,
    5.52268051, 5.70087713],
    [1.41421356, 0.        , 1.        , 2.23606798, 1.58113883,
    1.11803399, 2.12132034, 3.16227766, 3.80788655, 4.47213595,
    4.74341649, 5.14781507],
    [1.        , 1.        , 0.        , 1.41421356, 0.70710678,
    1.80277564, 1.58113883, 3.60555128, 4.30116263, 4.12310563,
    4.52769257, 4.74341649],
    [1.        , 2.23606798, 1.41421356, 0.        , 0.70710678,
    3.20156212, 2.54950976, 5.        , 5.70087713, 5.        ,
    5.52268051, 5.52268051],
    [0.70710678, 1.58113883, 0.70710678, 0.70710678, 0.        ,
    2.5       , 2.        , 4.30116263, 5.        , 4.52769257,
    5.        , 5.09901951],
    [2.5       , 1.11803399, 1.80277564, 3.20156212, 2.5       ,
    0.        , 2.06155281, 2.06155281, 2.6925824 , 3.90512484,
    4.03112887, 4.60977223],
    [2.54950976, 2.12132034, 1.58113883, 2.54950976, 2.        ,
    2.06155281, 0.        , 2.91547595, 3.60555128, 2.54950976,
    3.        , 3.16227766],
    [4.47213595, 3.16227766, 3.60555128, 5.        , 4.30116263,
    2.06155281, 2.91547595, 0.        , 0.70710678, 3.16227766,
    2.91547595, 3.80788655],
    [5.14781507, 3.80788655, 4.30116263, 5.70087713, 5.        ,
    2.6925824 , 3.60555128, 0.70710678, 0.        , 3.53553391,
    3.16227766, 4.12310563],
    [5.09901951, 4.47213595, 4.12310563, 5.        , 4.52769257,
    3.90512484, 2.54950976, 3.16227766, 3.53553391, 0.        ,
    0.70710678, 0.70710678],
    [5.52268051, 4.74341649, 4.52769257, 5.52268051, 5.        ,
    4.03112887, 3.        , 2.91547595, 3.16227766, 0.70710678,
    0.        , 1.        ],
    [5.70087713, 5.14781507, 4.74341649, 5.52268051, 5.09901951,
    4.60977223, 3.16227766, 3.80788655, 4.12310563, 0.70710678,
    1.        , 0.        ]
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

BASE_DENSITYGRAPH_e15_0 = (
    [1, 2, 3, 4,  # 0
     0, 2, 5,     # 1
     0, 1, 3, 4,  # 2
     0, 2, 4,     # 3
     0, 2, 3,     # 4
     1,           # 5
     8,           # 7
     7,           # 8
     10, 11,      # 9
     9, 11,       # 10
     9, 10],     # 11
    [0, 4, 7, 11, 14, 17, 18, 18, 19, 20, 22, 24, 26]
)

BASE_DENSITYGRAPH_e15_1 = (
    [1, 2, 3, 4,  # 0
     0, 2,        # 1
     0, 1, 3, 4,  # 2
     0, 2, 4,     # 3
     0, 2, 3,     # 4
     10, 11,      # 9
     9, 11,       # 10
     9, 10],      # 11
    [0, 4, 6, 10, 13, 16, 16, 16, 16, 16, 18, 20, 22]
)

BASE_DENSITYGRAPH_e15_2 = (
    [2, 3, 4,  # 0
     0, 3, 4,  # 2
     0, 2, 4,  # 3
     0, 2, 3],  # 4
    [0, 3, 3, 6, 9, 12, 12, 12, 12, 12, 12, 12, 12]
)

BASE_LABELS_e15_0 = [1, 1, 1, 1, 1, 1, 0, 2, 2, 3, 3, 3]
BASE_LABELS_e15_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2]
BASE_LABELS_e15_2 = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


@pytest.fixture
def base_points():
    return BASE_POINTS


@pytest.fixture
def base_distances():
    return BASE_DISTANCES


@pytest.fixture
def base_neighbourhoods_e15():
    return BASE_NEIGHBOURHOODS_e15


@pytest.fixture
def base_densitygraph_e15_0():
    return BASE_DENSITYGRAPH_e15_0


@pytest.fixture
def base_densitygraph_e15_1():
    return BASE_DENSITYGRAPH_e15_1


@pytest.fixture
def base_densitygraph_e15_2():
    return BASE_DENSITYGRAPH_e15_2


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
