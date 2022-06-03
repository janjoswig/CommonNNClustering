from collections.abc import MutableSequence

import numpy as np

try:
    import pandas as pd
    PANDAS_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    PANDAS_FOUND = False


class Summary(MutableSequence):
    """List like container for cluster results

    Stores instances of :obj:`cnnclustering.cluster.Record`.
    """

    def __init__(self, iterable=None):
        if iterable is None:
            iterable = []

        self._list = []
        for i in iterable:
            self.append(i)

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, item):
        if isinstance(item, Record):
            self._list.__setitem__(key, item)
        else:
            raise TypeError(
                "Summary can only contain records of type `Record`"
            )

    def __delitem__(self, key):
        self._list.__delitem__(key)

    def __len__(self):
        return self._list.__len__()

    def __str__(self):
        return self._list.__str__()

    def insert(self, index, item):
        if isinstance(item, Record):
            self._list.insert(index, item)
        else:
            raise TypeError(
                "Summary can only contain records of type `Record`"
            )

    def to_DataFrame(self):
        """Convert list of records to (typed) :obj:`pandas.DataFrame`

        Returns:
            :obj:`pandas.DataFrame`
        """

        if not PANDAS_FOUND:
            raise ModuleNotFoundError("No module named 'pandas'")

        _record_dtypes = [
            pd.Int64Dtype(),  # points
            np.float64,       # r
            pd.Int64Dtype(),  # n
            pd.Int64Dtype(),  # min
            pd.Int64Dtype(),  # max
            pd.Int64Dtype(),  # clusters
            np.float64,       # largest
            np.float64,       # noise
            np.float64,       # time
            ]

        content = []
        for field in Record.__slots__:
            content.append([
                record.__getattribute__(field)
                for record in self._list
                ])

        return make_typed_DataFrame(
            columns=Record.__slots__,
            dtypes=_record_dtypes,
            content=content,
            )


class Record:
    """Cluster result container

    :obj:`cnnclustering.cluster.Record` instances can created during
    :meth:`cnnclustering.cluster.Clustering.fit` and
    are collected in :obj:`cnnclustering.cluster.Summary`.
    """

    __slots__ = [
        "n_points",
        "radius_cutoff",
        "cnn_cutoff",
        "member_cutoff",
        "max_clusters",
        "n_clusters",
        "ratio_largest",
        "ratio_noise",
        "execution_time",
        ]

    def __init__(
            self,
            n_points=None,
            radius_cutoff=None,
            cnn_cutoff=None,
            member_cutoff=None,
            max_clusters=None,
            n_clusters=None,
            ratio_largest=None,
            ratio_noise=None,
            execution_time=None):

        self.n_points = n_points
        self.radius_cutoff = radius_cutoff
        self.cnn_cutoff = cnn_cutoff
        self.member_cutoff = member_cutoff
        self.max_clusters = max_clusters
        self.n_clusters = n_clusters
        self.ratio_largest = ratio_largest
        self.ratio_noise = ratio_noise
        self.execution_time = execution_time

    def __repr__(self):
        attrs_str = ", ".join(
            [
                f"{attr}={getattr(self, attr)!r}"
                for attr in self.__slots__
            ]
            )
        return f"{type(self).__name__}({attrs_str})"

    def __str__(self):
        attr_str = ""
        for attr in self.__slots__:
            if attr == "execution_time":
                continue

            value = getattr(self, attr)
            if value is None:
                attr_str += f"{value!r:<10}"
            elif isinstance(value, float):
                attr_str += f"{value:<10.3f}"
            else:
                attr_str += f"{value:<10}"

        if self.execution_time is not None:
            hours, rest = divmod(self.execution_time, 3600)
            minutes, seconds = divmod(rest, 60)
            execution_time_str = f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:.3f}"
        else:
            execution_time_str = None

        printable = (
            f'{"-" * 95}\n'
            f"#points   "
            f"r         "
            f"c         "
            f"min       "
            f"max       "
            f"#clusters "
            f"%largest  "
            f"%noise    "
            f"time     \n"
            f"{attr_str}"
            f"{execution_time_str}\n"
            f'{"-" * 95}\n'
        )
        return printable

    def to_dict(self):
        return {
            "n_points": self.n_points,
            "radius_cutoff": self.radius_cutoff,
            "cnn_cutoff": self.cnn_cutoff,
            "member_cutoff": self.member_cutoff,
            "max_clusters": self.max_clusters,
            "n_clusters": self.n_clusters,
            "ratio_largest": self.ratio_largest,
            "ratio_noise": self.ratio_noise,
            "execution_time": self.execution_time,
            }


def make_typed_DataFrame(columns, dtypes, content=None):
    """Construct :obj:`pandas.DataFrame` with typed columns"""

    if not PANDAS_FOUND:
        raise ModuleNotFoundError("No module named 'pandas'")

    assert len(columns) == len(dtypes)

    if content is None:
        content = [[] for i in range(len(columns))]

    df = pd.DataFrame({
        k: pd.array(c, dtype=v)
        for k, v, c in zip(columns, dtypes, content)
        })

    return df
