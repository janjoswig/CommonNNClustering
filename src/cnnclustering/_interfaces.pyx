from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from typing import Container, Iterator, Sequence


from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


class InputData(ABC):
    """Defines the input data interface"""

    @property
    @abstractmethod
    def data(self):
        """Return underlying data (only for user convenience, not to be relied on)"""

    @property
    @abstractmethod
    def meta(self):
        """Return meta-information"""

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Return total number of points"""

    @abstractmethod
    def get_subset(self, indices: Container) -> Type['InputData']:
        """Return input data subset"""


class InputDataComponents(InputData):
    """Extends the input data interface"""

    @property
    @abstractmethod
    def n_dim(self) -> int:
        """Return total number of dimensions"""

    @abstractmethod
    def get_component(self, point: int, dimension: int) -> float:
        """Return one component of point coordinates"""

    @abstractmethod
    def to_components_array(self):
        """Return input data as NumPy array of shape (#points, #components)"""


class InputDataPairwiseDistances(InputData):
    """Extends the input data interface"""

    @abstractmethod
    def get_distance(self, point_a: int, point_b: int) -> float:
        """Return the pairwise distance between two points"""


class InputDataPairwiseDistancesComputer(InputDataPairwiseDistances):
    """Extends the input data interface"""

    @abstractmethod
    def compute_distances(self, input_data: Type["InputData"]) -> None:
        """Pre-compute pairwise distances"""


class InputDataNeighbourhoods(InputData):
    """Extends the input data interface"""

    @abstractmethod
    def get_n_neighbours(self, point: int) -> int:
        """Return number of neighbours for point"""

    @abstractmethod
    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""


class InputDataNeighbourhoodsComputer(InputDataNeighbourhoods):
    """Extends the input data interface"""

    @abstractmethod
    def compute_neighbourhoods(
            self,
            input_data: Type["InputData"], r: float,
            is_sorted: bool = False, is_selfcounting: bool = True) -> None:
        """Pre-compute neighbourhoods at radius"""


cdef class InputDataExtInterface:
    """Defines the input data interface for Cython extension types"""

    cdef AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil: ...

    def get_component(self, point: int, dimension: int) -> int:
        return self._get_component(point, dimension)

    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil: ...

    def get_n_neighbours(self, point: int) -> int:
        return self._get_n_neighbours(point)

    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil: ...

    def get_neighbour(self, point: int, member: int) -> int:
        return self._get_neighbour(point, member)

    cdef AVALUE _get_distance(self, const AINDEX point_a, const AINDEX point_b) nogil: ...

    def get_distance(self, point_a: int, point_b: int) -> int:
        return self._get_distance(point_a, point_b)

    cdef void _compute_distances(self, InputDataExtInterface input_data) nogil: ...

    def compute_distances(self, InputDataExtInterface input_data):
        self._compute_distances(input_data)

    cdef void _compute_neighbourhoods(
            self,
            InputDataExtInterface input_data, AVALUE r,
            ABOOL is_sorted, ABOOL is_selfcounting) nogil: ...

    def compute_neighbourhoods(
            self,
            InputDataExtInterface input_data, AVALUE r,
            ABOOL is_sorted, ABOOL is_selfcounting):
        self._compute_neighbourhoods(input_data, r, is_sorted, is_selfcounting)


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def assign(self, member: int) -> None:
       """Add a member to this container"""

    @abstractmethod
    def reset(self) -> None:
       """Reset/empty this container"""

    @abstractmethod
    def enough(self, member_cutoff: int) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def contains(self, member: int) -> bool:
       """Return True if member is in neighbours container"""


class NeighboursGetter(ABC):
    """Defines the neighbours-getter interface"""

    @property
    @abstractmethod
    def is_sorted(self) -> bool:
       """Return True if neighbour indices are sorted"""

    @property
    @abstractmethod
    def is_selfcounting(self) -> bool:
       """Return True if points count as their own neighbour"""

    @abstractmethod
    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            distance_getter: Type['DistanceGetter'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours for point in input data"""

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            distance_getter: Type['DistanceGetter'],
            metric: Type['Metric'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours in input data for point in other input data"""


class DistanceGetter(ABC):
    """Defines the distance getter interface"""

    @abstractmethod
    def get_single(
            self,
            point_a: int, point_b: int,
            input_data: Type['InputData'],
            metric: Type["Metric"]) -> float:
        """Get distance between two points in input data"""

    @abstractmethod
    def get_single_other(
            self,
            point_a: int, point_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            metric: Type["Metric"]) -> float:
        """Get distance between two points in input data and other input data"""


class Metric(ABC):
    """Defines the metric-interface"""

    @abstractmethod
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        """Return distance between two points in input data"""

    @abstractmethod
    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:
        """Return distance between two points in input data and other input data"""


class SimilarityChecker(ABC):
    """Defines the similarity checker interface"""

    @abstractmethod
    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:
        """Retrun True if a and b have sufficiently many common neighbours"""


class Queue(ABC):
    """Defines the queue interface"""

    @abstractmethod
    def push(self, value):
        """Put value into the queue"""

    @abstractmethod
    def pop(self):
        """Retrieve value from the queue"""

    @abstractmethod
    def is_empty(self) -> bool:
        """Return True if there are no values in the queue"""


class Fitter(ABC):
    """Defines the fitter interface"""

    @abstractmethod
    def fit(
            self,
            input_data: Type['InputData'],
            neighbours_getter: Type['NeighboursGetter'],
            distance_getter: Type['DistanceGetter'],
            neighbours: Type['Neighbours'],
            neighbour_neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            similarity_checker: Type['SimilarityChecker'],
            queue: Type['Queue'],
            labels: Type['Labels'],
            cluster_params: Type['ClusterParameters']):
        """Generic clustering"""


class Predictor(ABC):
    """Defines the predictor interface"""

    @abstractmethod
    def predict(
            self,
            input_data: Type['InputData'],
            predictand_input_data: Type['InputData'],
            neighbours_getter: Type['NeighboursGetter'],
            predictand_neighbours_getter: Type['NeighboursGetter'],
            distance_getter: Type['DistanceGetter'],
            predictand_distance_getter: Type['DistanceGetter'],
            neighbours: Type['Neighbours'],
            neighbour_neighbours: Type['Neighbours'],
            metric: Type['Metric'],
            similarity_checker: Type['SimilarityChecker'],
            labels: Type['Labels'],
            predictand_labels: Type['Labels'],
            cluster_params: Type['ClusterParameters']):
        """Generic cluster label prediction"""
