import numpy as np
import pytest

import core.cnn as cnn


class TestReel:
    """Tests for child cluster merging after hierarchical clustering"""

    def test_reel_deep_None(self, hierarchical_cobj):
        hierarchical_cobj.reel(deep=None)
        np.testing.assert_array_equal(
            hierarchical_cobj.labels,
            cnn.Labels([0, 0, 0, 3, 0, 0, 0, 2, 6, 5, 0, 2, 2, 3, 0])
            )

    def test_reel_deep_1(self, hierarchical_cobj):
        hierarchical_cobj.reel(deep=1)
        np.testing.assert_array_equal(
            hierarchical_cobj.labels,
            cnn.Labels([0, 0, 0, 3, 0, 0, 0, 2, 4, 4, 4, 2, 2, 3, 0])
            )
