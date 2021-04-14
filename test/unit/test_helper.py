import time

from cnnclustering import cluster


def test_timed_decorator():
    def some_function():
        time.sleep(0.01)

    decorated = cluster.timed(some_function)
    decorated_result = decorated()

    assert isinstance(decorated_result, tuple)
    assert decorated_result[0] is None
    assert isinstance(decorated_result[1], float)
