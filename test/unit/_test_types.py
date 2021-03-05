import numpy as np
import pytest

from cnnclustering import _types


class TestPoints:
    @pytest.mark.parametrize("data", [None, np.random.normal(size=(100, 2))])
    def test_create(self, data):
        points = _types.Points(data)
        assert isinstance(points.__str__(), str)
        assert isinstance(points.shape, tuple)
