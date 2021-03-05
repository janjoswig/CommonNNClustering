import cnnclustering.cnn as cnn


class TestPandas:
    """Tests for pandas related operations"""

    def test_TypedDataFrame(self):
        tdf = cnn.TypedDataFrame(
            columns=["a", "b"],
            dtypes=[int, str],
            content=[[0, 1, 2], ["None", "True", "foo"]],
        )
        assert list(tdf.columns) == ["a", "b"]
