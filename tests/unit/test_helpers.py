import time

from cnnclustering import cnn


class TestTimedDecorator:

    def test_decorate(self):
        def some_function():
            time.sleep(0.1)

        decorated = cnn.timed(some_function)
        decorated_result = decorated()
        assert isinstance(decorated_result, tuple)
