try:
    import pandas as pd
    PANDAS_FOUND = True
except ModuleNotFoundError:
    PANDAS_FOUND = False

import pytest

from cnnclustering import cluster


pytestmark = pytest.mark.pandas


def test_make_typed_DataFssrame():
    if not PANDAS_FOUND:
        pytest.skip("Test function requires pandas.")

    tdf = cluster.make_typed_DataFrame(
        columns=["a", "b"],
        dtypes=[int, str],
        content=[[0, 1, 2], ["None", "True", "foo"]],
    )
    assert list(tdf.columns) == ["a", "b"]
