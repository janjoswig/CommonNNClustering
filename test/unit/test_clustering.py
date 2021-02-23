from cnnclustering import cluster


class TestCommonNNClustering:
    def test_create(self):
        clustering = cluster.CommonNNClustering(None)
        assert clustering

    def test_fit(self):
        clustering = cluster.CommonNNClustering(None)
        clustering.fit(None, None)
        assert clustering._labels == []
