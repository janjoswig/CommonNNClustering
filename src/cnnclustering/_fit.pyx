from collections import deque

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE
from cnnclustering._types cimport INPUT_DATA


cpdef void fit_deque(
        INPUT_DATA input_data,
        labels,
        consider):
    """Generic clustering

    Features of this variant:
        V1 (Queue): Python :obj:`collections.deque`

    Args:
        input_data: Data source implementing the input data interface.
        labels: Indexable sequence as cluster label assignment container.
        consider: Indexable sequence of 0 and 1 to track point inclusion.
    """

    cdef AINDEX n, m

    with nogil(INPUT_DATA is object):
        n = input_data.n_points

        current = 1
        q = deque()  # V1 (Queue)

        for init_point in range(n):
            if consider[init_point] == 0:
                continue
            consider[init_point] = 0

            neighbours = input_data.get_neighbours(
                init_point,
                )

            if not neighbours.enough():
                continue

            labels[init_point] = current

            while True:

                neighbours_indexable = neighbours.get_indexable()
                m = neighbours.n_points

                for member_index in range(m):
                    member = neighbours_indexable[member_index]

                    if consider[member] == 0:
                        continue

                    neighbour_neighbours = input_data.get_neighbours(
                        member
                        )

                    if not neighbour_neighbours.enough():
                        consider[member] = 0
                        continue

                    if neighbours.check_similarity(
                            neighbour_neighbours):
                        consider[member] = 0
                        labels[member] = current
                        q.append(member)

                if not q:
                    break

                point = q.popleft()
                neighbours = input_data.get_neighbours(
                    point
                    )

            current += 1