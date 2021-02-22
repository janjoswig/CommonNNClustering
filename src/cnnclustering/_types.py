from abc import ABC, abstractmethod
import pathlib
import pickle
from typing import Any, Optional, Union
from typing import Iterator, Sequence

import numpy as np

from . import _cfits


class PointsABC(ABC):
    """Abstract base class for points classes

    Points containers store coordinates of *n* points in *d* dimensions.

    To qualify as a points container, the following
    attributs and methods should be implemented in any case:

        shape: A tuple (*n*, *m*).
        __str__: A useful str-representation.

    """

    @abstractmethod
    def __str__(self):
        """Reveal type of points"""

    @property
    @abstractmethod
    def shape(self):
        """Return tuple (#points, #dimensions)"""


class Points(PointsABC):
    """Concrete points class for generic data"""

    def __init__(self, points=None):
        self._data = points
        self.params = {
            "radius_cutoff": None,
            "cnn_cutoff": None
            }

    @property
    def parameter(self):
        return self._params

    @property
    def shape(self):
        if self._data is None:
            return (0, 0)
        return self._data.shape

    def get_neighbours(self, point):
        neighbours = set()
        p = self._data[point]

    def __str__(self):
        return self._data.__str__()


class PointsArray(np.ndarray):
    """Concrete points class based on NumPy array"""

    def __new__(
            cls,
            p: Optional[Any] = None,
            edges: Optional[Sequence] = None,
            tree: Optional[Any] = None):
        if p is None:
            p = []

        p = np.asarray(p, dtype=np.float_)
        assert len(p.shape) <= 2  # Accept structure of 2 or less D.
        obj = np.atleast_2d(p).view(cls)

        if edges is None:
            edges = []
        obj._edges = np.atleast_1d(np.asarray(edges,
                                              dtype=_cfits.ARRAYINDEX_DTYPE))
        obj._tree = tree
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._edges = getattr(obj, "edges", None)
        self._tree = getattr(obj, "tree", None)

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, x):
        if x is None:
            x = []
        else:
            sum_edges = sum(x)
            n, d = self.shape

            if (d != 0) and (sum_edges != n):
                raise ValueError(
                    f"Part edges ({sum_edges} points) do not match data points "
                    f"({n} points)"
                    )

        self._edges = np.asarray(x, dtype=_cfits.ARRAYINDEX_DTYPE)

    @property
    def tree(self):
        return self._tree

    @classmethod
    def from_parts(cls, p: Optional[Sequence]):
        """Alternative constructor

        Use if data is passed as collection of parts, as

            >>> p = Points.from_parts([[[0, 0], [1, 1]],
            ...                        [[2, 2], [3,3]]])
            ... p
            Points([[0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3]])

        Recognised input formats are:
            * Sequence
            * 2D Sequence (sequence of sequences all of same length)
            * Sequence of 2D sequences all of same second dimension

        In this way, part edges are taken from the input shape and do
        not have to be specified explicitly. Calls :meth:`get_shape`.

        Args:
            p: File name as string or :obj:`pathlib.Path` object

        Return:
            Instance of :obj:`Points`
        """

        return cls(*cls.get_shape(p))

    @classmethod
    def from_file(
            cls,
            f: Union[str, pathlib.Path],
            *args,
            from_parts: bool = False,
            **kwargs):
        """Alternative constructor

        Load file content to be interpreted as points. Uses :meth:`load`
        to read files.

        Recognised input formats are:
            * Sequence
            * 2D Sequence (sequence of sequences all of same length)
            * Sequence of 2D sequences all of same second dimension

        Args:
            f: File name as string or :obj:`pathlib.Path` object.
            *args: Arguments passed to :meth:`load`.
            from_parts: If `True` uses :meth:`from_parts` constructor.
               If `False` uses default constructor.

        Return:
            Instance of :obj:`Points`
        """

        if from_parts:
            return cls(*cls.get_shape(cls.load(f, *args, **kwargs)))
        return cls(cls.load(f, *args, **kwargs))

    @staticmethod
    def get_shape(data: Any):
        r"""Maintain data in universal shape (2D NumPy array)

        Analyses the format of given data and fits it into the standard
        format (parts, points, dimensions).  Creates a
        :obj:`numpy.ndarray` vstacked along the parts componenent that
        can be passed to the `Points` constructor alongside part edges.
        This may not be able to deal with all possible kinds of input
        structures correctly, so check the outcome carefully.

        Recognised input formats are:
            * Sequence
            * 2D Sequence (sequence of sequences all of same length)
            * Sequence of 2D sequences all of same second dimension

        Args:
            data: Either `None`
                or:

                * a 1D sequence of length *d*,
                  interpreted as 1 point in *d* dimension
                * a 2D sequence of length *n* (rows) and width
                  *d* (columns),
                  interpreted as *n* points in *d* dimensions
                * a list of 2D sequences,
                  interpreted as groups (parts) of points

        Returns:
            Tuple of

                * NumPy array of shape (:math:`\sum n, d`)
                * Part edges list, marking the end points of the
                  parts
        """

        if data is None:
            return None, None

        data_shape = np.shape(data[0])
        # raises a type error if data is not subscribable

        if np.shape(data_shape)[0] == 0:
            # 1D Sequence passed
            data = [np.array([data])]

        elif np.shape(data_shape)[0] == 1:
            # 2D Sequence of sequences passed"
            assert len({len(s) for s in data}) == 1
            data = [np.asarray(data)]

            # TODO Does not catch edge case in which sequence of 2D
            #    sequences is passed, not all having the same dimension

        elif np.shape(data_shape)[0] == 2:
            assert len({len(s) for s_ in data for s in s_}) == 1
            data = [np.asarray(x) for x in data]

        else:
            raise ValueError(
                f"Data shape {data_shape} not allowed"
                )

        edges = [x.shape[0] for x in data]

        return np.vstack(data), edges

    def by_parts(self) -> Iterator:
        """Yield data by parts

        Returns:
            Generator of 2D :obj:`numpy.ndarray` s (parts)
        """

        if self.size > 0:
            _edges = self.edges
            if _edges.shape[0] == 0:
                _edges = np.asarray([self.shape[0]], dtype=_cfits.ARRAYINDEX_DTYPE)

            start = 0
            for end in _edges:
                yield self[start:(start + end), :]
                start += end

        else:
            yield from ()

    @staticmethod
    def load(f: Union[pathlib.Path, str], *args, **kwargs) -> None:
        """Loads file content

        Depending on the filename extension, a suitable loader is
        called:

            * .p: :func:`pickle.load`
            * .npy: :func:`numpy.load`
            * None: :func:`numpy.loadtxt`
            * .xvg, .dat: :func:`numpy.loadtxt`

        Sets :attr:`data` and :attr:`shape`.

        Args:
            f: File
            *args: Passed to loader

        Keyword Args:
            **kwargs: Passed to loader

        Returns:
            Return value of the loader
        """

        extension = pathlib.Path(f).suffix

        case_ = {
            '.p': lambda: pickle.load(
                open(f, 'rb'),
                *args,
                **kwargs
                ),
            '.npy': lambda: np.load(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
            '': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
            '.xvg': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
            '.dat': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                *args,
                **kwargs
                ),
             }

        return case_.get(
            extension,
            lambda: print(f"Unknown filename extension {extension}")
            )()

    def cKDTree(self, **kwargs):
        """Wrapper for :meth:`scipy.spatial.cKDTree`

        Sets :attr:`Points.tree`.

        Args:
            **kwargs: Passed to :meth:`scipy.spatial.cKDTree`
        """

        self._tree = cKDTree(self, **kwargs)