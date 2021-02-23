class CommonNNClustering:

    def __init__(self, data):
        self._data = data

    def fit(self, radius_cutoff, cnn_cutoff):
       self._labels = []