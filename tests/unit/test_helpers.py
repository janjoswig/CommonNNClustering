import time

from cnnclustering import cnn


class TestTimedDecorator:

    def test_decorate(self):
        def some_function():
            time.sleep(0.1)

        decorated = cnn.timed(some_function)
        decorated_result = decorated()
        print(decorated_result)
