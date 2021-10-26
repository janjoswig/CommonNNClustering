import numpy as np
import pytest

from cnnclustering import hooks


@pytest.mark.parametrize(
    "data,expected_data,expected_meta",
    [
        pytest.param(
            1, None, None,
            marks=pytest.mark.raises(exception=TypeError)
        ),
        ([], [[]], {"edges": [1]}),
        ([1, 2, 3], [[1, 2, 3]], {"edges": [1]}),
        pytest.param(
            [[1, 2, 3], [4, 5]], None, None,
            marks=pytest.mark.raises(exception=ValueError)
        ),
        (
            [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
            {"edges": [2]}
        ),
        pytest.param(
            [[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11], [13, 14, 15]]], None, None,
            marks=pytest.mark.raises(exception=ValueError)
        ),
        (
            [[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12], [13, 14, 15]]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            {"edges": [2, 3]}
        ),
    ],
    ids=[
        "invalid", "empty", "1d", "2d_invalid", "2d", "1d2d_invalid",
        "1d2d"
        ]
)
def test_prepare_points_from_parts(
        data, expected_data, expected_meta):
    data_args, data_kwargs = hooks.prepare_points_from_parts(data)
    reformatted_data = data_args[0]
    np.testing.assert_array_equal(
        expected_data,
        reformatted_data
        )
    assert data_kwargs["meta"] == expected_meta


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [0], [0, 3], [0, 2]]
    ]
)
def test_prepare_neighbourhoods(data):
    (padded_data, n_neighbours), data_kwargs = hooks.prepare_neighbourhoods(data)

    assert isinstance(data_kwargs.get("meta"), dict)
    assert padded_data.shape[0] == len(data)
    assert n_neighbours.shape[0] == len(data)
    assert padded_data.shape[1] == n_neighbours.max()
